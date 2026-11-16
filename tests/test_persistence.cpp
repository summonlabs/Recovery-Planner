#include "test_util.hpp"
#include "scenario.hpp"
#include <filesystem>
#include <fstream>
#include <chrono>

using namespace recovery_planner;
using namespace scenario;
namespace fs = std::filesystem;

namespace {
std::string temp_dir() {
  static long counter = 0;
  auto ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return (fs::temp_directory_path() / ("rplanner_test_" + std::to_string(ts) +
          "_" + std::to_string(counter++))).string();
}
void cleanup(const std::string& dir) {
  std::error_code ec;
  fs::remove_all(dir, ec);
}
std::string frame_path(const std::string& dir, const std::string& key) {
  return (fs::path(dir) / (key + ".rpp")).string();
}
void make_world(RecoveryPlanner& planner) {
  planner.register_worker(kWorkerA, kBootA);
  planner.register_worker(kWorkerB, kBootB);
  planner.set_resource("gpu_mem", 10000);
  planner.set_recompute_available(kWorkloadGen, true);
}
}

// ---------------------------------------------------------------------------
RP_TEST(persistence_save_load_roundtrip) {
  std::string dir = temp_dir();
  auto store = make_file_persistence_store(dir);
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, store);
  make_world(planner);

  RecoveryRequest req = make_request();
  auto decision = planner.plan(req, {make_candidate(RecoveryStrategy::RESTORE, 1)});
  RP_CHECK(decision.outcome == PlanOutcome::PLAN_FOUND);
  planner.save_state();
  std::string digest = decision.plan->decision_digest;

  RecoveryPlanner planner2(clock, store);
  planner2.load_state();
  make_world(planner2);  // workers must republish after load (dynamic evidence)
  RP_CHECK(planner2.context().epoch == kEpoch);
  auto plans = planner2.plans();
  RP_CHECK(plans.size() == 1);
  RP_CHECK(plans[0].decision_digest == digest);

  // Replaying the same durable request + candidate reconstructs the same identity.
  auto d2 = planner2.plan(req, {make_candidate(RecoveryStrategy::RESTORE, 1)});
  RP_CHECK(d2.plan->decision_digest == digest);
  cleanup(dir);
}

// ---------------------------------------------------------------------------
RP_TEST(persistence_corruption_rejected) {
  std::string dir = temp_dir();
  auto store = make_file_persistence_store(dir);
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, store);
  make_world(planner);
  planner.plan(make_request(), {make_candidate(RecoveryStrategy::RESTORE, 1)});
  planner.save_state();

  // Corrupt one payload byte.
  std::string path = frame_path(dir, "planner_state");
  std::vector<std::uint8_t> raw;
  {
    std::ifstream ifs(path, std::ios::binary);
    raw.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
  }
  RP_CHECK(raw.size() > 16);
  raw[raw.size() / 2] ^= 0xFF;  // flip a byte in the payload
  {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    ofs.write(reinterpret_cast<const char*>(raw.data()), (std::streamsize)raw.size());
  }
  auto bad = store->load("planner_state");
  RP_CHECK(!bad.has_value());
  cleanup(dir);
}

// ---------------------------------------------------------------------------
RP_TEST(persistence_truncation_rejected) {
  std::string dir = temp_dir();
  auto store = make_file_persistence_store(dir);
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, store);
  make_world(planner);
  planner.plan(make_request(), {make_candidate(RecoveryStrategy::RESTORE, 1)});
  planner.save_state();

  std::string path = frame_path(dir, "planner_state");
  std::vector<std::uint8_t> raw;
  {
    std::ifstream ifs(path, std::ios::binary);
    raw.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
  }
  raw.resize(raw.size() / 2);
  {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    ofs.write(reinterpret_cast<const char*>(raw.data()), (std::streamsize)raw.size());
  }
  auto bad = store->load("planner_state");
  RP_CHECK(!bad.has_value());
  cleanup(dir);
}

// ---------------------------------------------------------------------------
RP_TEST(coordinator_restart_invalidates_inflight_and_advances_epoch) {
  std::string dir = temp_dir();
  auto store = make_file_persistence_store(dir);
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, store);
  make_world(planner);
  RecoveryRequest req = make_request();
  auto decision = planner.plan(req, {make_candidate(RecoveryStrategy::RESTORE, 1)});
  auto authorized = planner.authorize_plan(*decision.plan);
  RP_CHECK(authorized.state == PlanLifecycleState::AUTHORIZED);
  planner.save_state();

  CoordinatorEpoch before_epoch = planner.current_epoch();
  planner.restart_coordinator();
  RP_CHECK(planner.current_epoch().value() == before_epoch.value() + 1);

  // The authorized-but-not-completed plan must not be resumed; it is invalidated.
  auto plans = planner.plans();
  bool found_invalidated = false;
  for (const auto& p : plans) {
    if (p.plan_id == decision.plan->plan_id) {
      RP_CHECK(p.state == PlanLifecycleState::INVALIDATED);
      found_invalidated = true;
    }
  }
  RP_CHECK(found_invalidated);

  // After restart, dynamic evidence requires re-registration. New work can plan
  // once workers republish under the new epoch.
  planner.register_worker(kWorkerA, kBootA);
  planner.register_worker(kWorkerB, kBootB);
  planner.set_resource("gpu_mem", 10000);
  planner.set_recompute_available(kWorkloadGen, true);
  CoordinatorEpoch now_epoch = planner.current_epoch();
  auto c = make_candidate(RecoveryStrategy::RESTORE, 1);
  c.authority->coordinator_epoch = now_epoch;
  c.authority->meta.coordinator_epoch = now_epoch;
  RecoveryRequest req2 = make_request();
  auto d2 = planner.plan(req2, {c});
  RP_CHECK(d2.outcome == PlanOutcome::PLAN_FOUND);
  cleanup(dir);
}

// ---------------------------------------------------------------------------
// Regression tests: recovery of identities, rejected snapshots, restart durability.
// ---------------------------------------------------------------------------
namespace {
// An executor that records each dispatch and never reports a completion.
class RecordingExecutor final : public IRecoveryExecutor {
 public:
  RecoveryReceipt dispatch(const RecoveryDispatch& d) override {
    seen.push_back(d);
    RecoveryReceipt r;
    r.dispatch_id = d.dispatch_id;
    r.status = DispatchStatus::DISPATCHED;
    return r;
  }
  const char* name() const noexcept override { return "recording"; }
  std::vector<RecoveryDispatch> seen;
};
}

// ---------------------------------------------------------------------------
RP_TEST(identities_are_not_reused_after_load) {
  auto store = make_memory_persistence_store();
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecordingExecutor exec;
  RecoveryPlanId old_id;
  RecoveryPlanGeneration old_generation;
  {
    RecoveryPlanner planner(clock, store);
    make_world(planner);
    auto d = planner.plan(make_request(), {make_candidate(RecoveryStrategy::RESTORE, 1)});
    auto dispatched = planner.dispatch_plan(planner.authorize_plan(*d.plan), exec);
    RP_CHECK(dispatched.state == PlanLifecycleState::EXECUTING);
    old_id = dispatched.plan_id;
    old_generation = dispatched.generation;
    planner.save_state();
  }

  RecoveryPlanner planner2(clock, store);
  RP_CHECK(planner2.load_state());
  make_world(planner2);
  RecoveryRequest req2 = make_request();
  req2.request_id = RecoveryRequestId(2);
  auto d2 = planner2.plan(req2, {make_candidate(RecoveryStrategy::RESTORE, 2)});
  RP_CHECK(d2.outcome == PlanOutcome::PLAN_FOUND);
  RP_CHECK(d2.plan->plan_id != old_id);
  RP_CHECK(d2.plan->generation != old_generation);
  RP_CHECK(planner2.plans().size() == 2);   // the recovered plan was not overwritten

  planner2.dispatch_plan(planner2.authorize_plan(*d2.plan), exec);
  RP_CHECK(exec.seen.size() == 2);
  RP_CHECK(exec.seen[0].dispatch_id != exec.seen[1].dispatch_id);
}

// ---------------------------------------------------------------------------
RP_TEST(rejected_snapshot_leaves_the_planner_unchanged) {
  auto store = make_memory_persistence_store();
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, store);
  make_world(planner);
  auto d = planner.plan(make_request(), {make_candidate(RecoveryStrategy::RESTORE, 1)});
  RP_CHECK(planner.save_state());
  auto good = store->load("planner_state");
  RP_CHECK(good.has_value());

  // Cut in the middle of the plan records.
  auto cut = *good;
  cut.resize(cut.size() / 2);
  store->save("planner_state", cut);
  RP_CHECK(!planner.load_state());
  auto plans = planner.plans();
  RP_CHECK(plans.size() == 1);
  RP_CHECK(plans[0].decision_digest == d.plan->decision_digest);
  RP_CHECK(planner.current_epoch() == kEpoch);

  // Bytes after a complete snapshot.
  auto padded = *good;
  padded.push_back(0);
  store->save("planner_state", padded);
  RP_CHECK(!planner.load_state());
  RP_CHECK(planner.plans().size() == 1);

  // The intact snapshot still loads.
  store->save("planner_state", *good);
  RP_CHECK(planner.load_state());
  RP_CHECK(planner.plans().size() == 1);
}

// ---------------------------------------------------------------------------
RP_TEST(load_state_reports_a_rejected_frame) {
  std::string dir = temp_dir();
  auto store = make_file_persistence_store(dir);
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  {
    RecoveryPlanner planner(clock, store);
    make_world(planner);
    planner.plan(make_request(), {make_candidate(RecoveryStrategy::RESTORE, 1)});
    planner.save_state();
  }
  std::string path = frame_path(dir, "planner_state");
  std::vector<std::uint8_t> raw;
  {
    std::ifstream ifs(path, std::ios::binary);
    raw.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
  }
  RP_CHECK(raw.size() > 16);
  raw[raw.size() / 2] ^= 0xFF;
  {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    ofs.write(reinterpret_cast<const char*>(raw.data()), (std::streamsize)raw.size());
  }
  // A frame exists but was rejected: that is not the same as an empty store.
  RecoveryPlanner fresh(clock, store);
  RP_CHECK(!fresh.load_state());
  RP_CHECK(fresh.plans().empty());

  // No stored state at all is not an error.
  std::string dir2 = temp_dir();
  RecoveryPlanner empty(clock, make_file_persistence_store(dir2));
  RP_CHECK(empty.load_state());
  cleanup(dir);
  cleanup(dir2);
}

// ---------------------------------------------------------------------------
RP_TEST(restart_persists_the_new_epoch_and_the_invalidations) {
  auto store = make_memory_persistence_store();
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanId id;
  {
    RecoveryPlanner planner(clock, store);
    make_world(planner);
    auto d = planner.plan(make_request(), {make_candidate(RecoveryStrategy::RESTORE, 1)});
    planner.authorize_plan(*d.plan);
    id = d.plan->plan_id;
    planner.restart_coordinator();
    RP_CHECK(planner.current_epoch() == CoordinatorEpoch(2));
  }
  // The next incarnation starts from what the restart left behind.
  RecoveryPlanner next(clock, store);
  RP_CHECK(next.load_state());
  RP_CHECK(next.current_epoch() == CoordinatorEpoch(2));
  auto plans = next.plans();
  RP_CHECK(plans.size() == 1);
  RP_CHECK(plans[0].plan_id == id);
  RP_CHECK(plans[0].state == PlanLifecycleState::INVALIDATED);
}

// ---------------------------------------------------------------------------
RP_TEST(hard_objective_verdict_survives_save_and_load) {
  auto store = make_memory_persistence_store();
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlan fallback;
  RecoveryPlan timed;
  {
    RecoveryPlanner planner(clock, store);
    make_world(planner);
    RecoveryRequest req = make_request();
    req.recovery_deadline = Timestamp(500);   // the candidate needs 5000
    auto d = planner.plan(req, {make_candidate(RecoveryStrategy::RESTORE, 1)});
    RP_CHECK(d.outcome == PlanOutcome::NO_DEADLINE_FEASIBLE_PLAN);
    fallback = *d.plan;

    RecoveryRequest req2 = make_request();
    req2.request_id = RecoveryRequestId(2);
    req2.recovery_deadline = Timestamp(100000);
    auto d2 = planner.plan(req2, {make_candidate(RecoveryStrategy::RESTORE, 2)});
    RP_CHECK(d2.outcome == PlanOutcome::PLAN_FOUND);
    timed = *d2.plan;
    planner.save_state();
  }

  RecoveryPlanner next(clock, store);
  RP_CHECK(next.load_state());
  make_world(next);
  RP_EXPECT_THROW(next.authorize_plan(fallback), ErrorCode::INFEASIBLE_CANDIDATE);
  clock->set(Timestamp(200000));
  RP_EXPECT_THROW(next.authorize_plan(timed), ErrorCode::STALE_AUTHORITY);
}

// ---------------------------------------------------------------------------
RP_TEST(snapshot_without_objective_records_still_loads) {
  auto store = make_memory_persistence_store();
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, store);
  make_world(planner);
  auto d = planner.plan(make_request(), {make_candidate(RecoveryStrategy::RESTORE, 1)});
  planner.save_state();
  auto blob = *store->load("planner_state");
  // One plan without deadline or tolerance: marker (4) + count (4) + plan id (8) + 4 flag bytes.
  RP_CHECK(blob.size() > 20);
  blob.resize(blob.size() - 20);
  store->save("planner_state", blob);

  RecoveryPlanner next(clock, store);
  RP_CHECK(next.load_state());
  auto plans = next.plans();
  RP_CHECK(plans.size() == 1);
  RP_CHECK(plans[0].decision_digest == d.plan->decision_digest);
}
