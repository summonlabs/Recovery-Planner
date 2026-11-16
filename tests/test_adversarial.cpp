#include "test_util.hpp"
#include "scenario.hpp"
#include <filesystem>
#include <fstream>
#include <limits>
#include <chrono>

using namespace recovery_planner;
using namespace scenario;
namespace fs = std::filesystem;

namespace {
std::string temp_dir() {
  static long counter = 0;
  auto ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return (fs::temp_directory_path() / ("rplanner_adv_" + std::to_string(ts) + "_" +
          std::to_string(counter++))).string();
}
void cleanup(const std::string& dir) { std::error_code ec; fs::remove_all(dir, ec); }
void write_frame(const std::string& path, const std::vector<std::uint8_t>& payload,
                 std::uint32_t version = kPersistenceFormatVersion,
                 bool append_garbage = false) {
  ByteWriter w;
  w.u32(kPersistenceMagic);
  w.u32(version);
  w.u32(static_cast<std::uint32_t>(payload.size()));
  for (std::uint8_t b : payload) w.u8(b);
  w.u32(crc32c(payload));
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  ofs.write(reinterpret_cast<const char*>(w.data().data()), (std::streamsize)w.size());
  if (append_garbage) ofs.write("Þ­¾ï", 4);
}
void make_world(RecoveryPlanner& planner) {
  planner.register_worker(kWorkerA, kBootA);
  planner.register_worker(kWorkerB, kBootB);
  planner.set_resource("gpu_mem", 10000);
  planner.set_recompute_available(kWorkloadGen, true);
}
}

// ---------------------------------------------------------------------------
RP_TEST(malformed_strategy_rejected) {
  auto ctx = make_context();
  auto c = make_candidate(RecoveryStrategy::RESTORE, 1);
  c.strategy = static_cast<RecoveryStrategy>(200);
  auto ev = evaluate_feasibility(c, make_request(), ctx);
  RP_CHECK(!ev.feasible());
}

// ---------------------------------------------------------------------------
RP_TEST(huge_duration_no_overflow) {
  auto ctx = make_context();
  auto c = make_candidate(RecoveryStrategy::RESTORE, 1);
  c.cost->expected_time = Duration(std::numeric_limits<std::int64_t>::max());
  RecoveryRequest req = make_request();
  req.recovery_deadline = Timestamp(std::numeric_limits<std::int64_t>::max() - 100);
  auto de = evaluate_deadline(c, req, ctx, Timestamp(0));
  // Must not crash; the outcome is a deterministic, bounded decision.
  RP_CHECK(de.outcome == DeadlineOutcome::HARD_FEASIBLE ||
           de.outcome == DeadlineOutcome::VIOLATED);
}

// ---------------------------------------------------------------------------
RP_TEST(negative_duration_clamped) {
  auto ctx = make_context();
  auto c = make_candidate(RecoveryStrategy::RESTORE, 1);
  c.cost->expected_time = Duration(-5000);
  RecoveryRequest req = make_request();
  req.recovery_deadline = Timestamp(1000);
  auto de = evaluate_deadline(c, req, ctx, Timestamp(0));
  // Negative time is clamped to 0, so completion==now <= deadline -> feasible.
  RP_CHECK(de.outcome == DeadlineOutcome::HARD_FEASIBLE);
}

// ---------------------------------------------------------------------------
RP_TEST(absurd_required_bytes_rejected) {
  auto ctx = make_context();
  auto c = make_candidate(RecoveryStrategy::RESTORE, 1);
  c.resource->required = std::numeric_limits<std::uint64_t>::max();
  auto ev = evaluate_feasibility(c, make_request(), ctx);
  RP_CHECK(!ev.feasible());
}

// ---------------------------------------------------------------------------
RP_TEST(frame_trailing_garbage_rejected) {
  std::string dir = temp_dir();
  auto store = make_file_persistence_store(dir);
  write_frame((fs::path(dir) / "g.rpp").string(), {1, 2, 3, 4}, kPersistenceFormatVersion, true);
  auto got = store->load("g");
  RP_CHECK(!got.has_value());
  cleanup(dir);
}

// ---------------------------------------------------------------------------
RP_TEST(frame_version_mismatch_rejected) {
  std::string dir = temp_dir();
  auto store = make_file_persistence_store(dir);
  write_frame((fs::path(dir) / "v.rpp").string(), {1, 2, 3, 4}, 9999);
  auto got = store->load("v");
  RP_CHECK(!got.has_value());
  cleanup(dir);
}

// ---------------------------------------------------------------------------
RP_TEST(valid_frame_roundtrips) {
  std::string dir = temp_dir();
  auto store = make_file_persistence_store(dir);
  write_frame((fs::path(dir) / "ok.rpp").string(), {9, 8, 7}, kPersistenceFormatVersion);
  auto got = store->load("ok");
  RP_CHECK(got.has_value());
  RP_CHECK(got->size() == 3 && (*got)[0] == 9 && (*got)[2] == 7);
  cleanup(dir);
}

// ---------------------------------------------------------------------------
RP_TEST(state_generation_lineage_mismatch_rejected) {
  auto ctx = make_context();
  auto c = make_candidate(RecoveryStrategy::RESTORE, 1);
  c.checkpoint->state_generation = StateGeneration(999);  // lineage mismatch
  auto ev = evaluate_feasibility(c, make_request(), ctx);
  RP_CHECK(!ev.feasible());
}

// ---------------------------------------------------------------------------
RP_TEST(forbidden_strategy_rejected) {
  auto ctx = make_context();
  auto c = make_candidate(RecoveryStrategy::RECOMPUTE, 1);
  RecoveryRequest req = make_request();
  req.forbidden_strategies.push_back(RecoveryStrategy::RECOMPUTE);
  auto ev = evaluate_feasibility(c, req, ctx);
  RP_CHECK(!ev.feasible());
  RP_CHECK(ev.result == FeasibilityResult::INFEASIBLE);
}

// ---------------------------------------------------------------------------
RP_TEST(deadline_in_past_is_violated) {
  auto ctx = make_context();
  auto c = make_candidate(RecoveryStrategy::RESTORE, 1);
  c.cost->expected_time = Duration(1000);
  RecoveryRequest req = make_request();
  req.recovery_deadline = Timestamp(-5000);
  auto de = evaluate_deadline(c, req, ctx, Timestamp(0));
  RP_CHECK(de.outcome == DeadlineOutcome::VIOLATED);
}

// ---------------------------------------------------------------------------
RP_TEST(ambiguous_completion_does_not_fabricate_success) {
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, make_memory_persistence_store());
  make_world(planner);
  RecoveryRequest req = make_request();
  auto d = planner.plan(req, {make_candidate(RecoveryStrategy::RESTORE, 1)});
  auto auth = planner.authorize_plan(*d.plan);
  ReferenceExecutor exec([](DispatchId, ExecutionOutcome, ProgressUnits, Duration, std::string) {});
  auto dispatched = planner.dispatch_plan(auth, exec);
  RP_CHECK(dispatched.state == PlanLifecycleState::EXECUTING);

  ExecutionReport rep;
  rep.plan_id = dispatched.plan_id;
  rep.plan_generation = dispatched.generation;
  rep.reporter_boot = dispatched.target_boot;
  rep.outcome = ExecutionOutcome::AMBIGUOUS;
  planner.report_execution(rep);
  auto _plans = planner.plans(); auto& p = _plans[0];
  RP_CHECK(p.state == PlanLifecycleState::FAILED);
  RP_CHECK(p.execution_outcome == ExecutionOutcome::AMBIGUOUS);
  RP_CHECK(p.outcome_published == false);
}

// ---------------------------------------------------------------------------
// Regression tests: evidence that was read but never enforced.
// ---------------------------------------------------------------------------
RP_TEST(invalid_authority_evidence_is_rejected) {
  auto ctx = make_context();
  for (int i = 0; i < 4; ++i) {
    auto c = make_candidate(RecoveryStrategy::RESTORE, 1);
    if (i == 0) c.authority->coordinator_authority_valid = false;
    if (i == 1) c.authority->policy_authority_valid = false;
    if (i == 2) c.authority->candidate_authority_valid = false;
    if (i == 3) c.authority->execution_authority_valid = false;
    auto ev = evaluate_feasibility(c, make_request(), ctx);
    RP_CHECK(ev.result == FeasibilityResult::INFEASIBLE);
  }
  // Authority evidence that is not current is not sufficient either.
  auto c = make_candidate(RecoveryStrategy::RESTORE, 1);
  c.authority->meta.freshness = Freshness::STALE;
  RP_CHECK(evaluate_feasibility(c, make_request(), ctx).result ==
           FeasibilityResult::INSUFFICIENT_EVIDENCE);
  c.authority->meta.freshness = Freshness::REVALIDATION_REQUIRED;
  RP_CHECK(evaluate_feasibility(c, make_request(), ctx).result ==
           FeasibilityResult::REVALIDATION_REQUIRED);
}

// ---------------------------------------------------------------------------
RP_TEST(availability_evidence_must_describe_the_target) {
  auto ctx = make_context();
  // The plan would run on worker B, but the only availability evidence is about worker A.
  auto c = make_candidate(RecoveryStrategy::RESTORE, 1);
  c.target_worker = kWorkerB;
  c.target_boot = kBootB;
  RP_CHECK(evaluate_feasibility(c, make_request(), ctx).result == FeasibilityResult::INFEASIBLE);

  // Same worker, another boot.
  auto c2 = make_candidate(RecoveryStrategy::RESTORE, 2);
  c2.target_boot = WorkerBootId(101);
  RP_CHECK(evaluate_feasibility(c2, make_request(), ctx).result == FeasibilityResult::INFEASIBLE);

  // Evidence about the target itself is accepted.
  auto c3 = make_candidate(RecoveryStrategy::RESTORE, 3);
  c3.target_worker = kWorkerB;
  c3.target_boot = kBootB;
  c3.availability->worker = kWorkerB;
  c3.availability->boot = kBootB;
  RP_CHECK(evaluate_feasibility(c3, make_request(), ctx).result == FeasibilityResult::FEASIBLE);
}

// ---------------------------------------------------------------------------
RP_TEST(restore_requires_a_usable_checkpoint) {
  auto ctx = make_context();
  auto corrupt = make_candidate(RecoveryStrategy::RESTORE, 1);
  corrupt.checkpoint->integrity = StateIntegrity::CORRUPT;
  RP_CHECK(evaluate_feasibility(corrupt, make_request(), ctx).result ==
           FeasibilityResult::INFEASIBLE);

  auto expired = make_candidate(RecoveryStrategy::RESTORE, 2);
  expired.checkpoint->meta.freshness = Freshness::EXPIRED;
  RP_CHECK(evaluate_feasibility(expired, make_request(), ctx).result ==
           FeasibilityResult::INFEASIBLE);

  auto unverified = make_candidate(RecoveryStrategy::RESTORE, 3);
  unverified.checkpoint->integrity = StateIntegrity::NOT_VERIFIED;
  RP_CHECK(evaluate_feasibility(unverified, make_request(), ctx).result ==
           FeasibilityResult::INSUFFICIENT_EVIDENCE);

  // A strategy that does not read the checkpoint is not affected by it.
  auto restart = make_candidate(RecoveryStrategy::RESTART, 4);
  restart.checkpoint->integrity = StateIntegrity::CORRUPT;
  RP_CHECK(evaluate_feasibility(restart, make_request(), ctx).result ==
           FeasibilityResult::FEASIBLE);
}
