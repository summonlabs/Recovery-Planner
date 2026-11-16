#include "test_util.hpp"
#include "scenario.hpp"

using namespace recovery_planner;
using namespace scenario;

namespace {
void make_world(RecoveryPlanner& planner) {
  planner.register_worker(kWorkerA, kBootA);
  planner.register_worker(kWorkerB, kBootB);
  planner.set_resource("gpu_mem", 10000);
  planner.set_recompute_available(kWorkloadGen, true);
}
}

// ---------------------------------------------------------------------------
RP_TEST(authorize_revalidate_dispatch_success) {
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, make_memory_persistence_store());
  make_world(planner);

  RecoveryRequest req = make_request();
  auto restore = make_candidate(RecoveryStrategy::RESTORE, 1);
  auto decision = planner.plan(req, {restore});
  RP_CHECK(decision.outcome == PlanOutcome::PLAN_FOUND);
  RP_CHECK(decision.plan->state == PlanLifecycleState::PLANNED);
  auto plan = *decision.plan;

  auto authorized = planner.authorize_plan(plan);
  RP_CHECK(authorized.state == PlanLifecycleState::AUTHORIZED);

  auto revalidated = planner.revalidate_plan(authorized);
  RP_CHECK(revalidated.state == PlanLifecycleState::AUTHORIZED);

  ReferenceExecutor exec(
      [&planner](DispatchId d, ExecutionOutcome o, ProgressUnits r, Duration e, std::string s) {
        planner.on_adapter_completion(d, o, r, e, std::move(s));
      });
  auto dispatched = planner.dispatch_plan(revalidated, exec);
  RP_CHECK(dispatched.state == PlanLifecycleState::SUCCEEDED);
  RP_CHECK(dispatched.execution_outcome == ExecutionOutcome::SUCCEEDED);
  RP_CHECK(dispatched.outcome_published == true);
}

// ---------------------------------------------------------------------------
RP_TEST(duplicate_completion_rejected) {
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, make_memory_persistence_store());
  make_world(planner);

  RecoveryRequest req = make_request();
  auto decision = planner.plan(req, {make_candidate(RecoveryStrategy::RESTORE, 1)});
  auto authorized = planner.authorize_plan(*decision.plan);
  ReferenceExecutor exec(
      [&planner](DispatchId d, ExecutionOutcome o, ProgressUnits r, Duration e, std::string s) {
        planner.on_adapter_completion(d, o, r, e, std::move(s));
      });
  auto dispatched = planner.dispatch_plan(authorized, exec);
  RP_CHECK(dispatched.state == PlanLifecycleState::SUCCEEDED);

  // Conflicting duplicate success from the correct identity must be ignored.
  ExecutionReport rep;
  rep.dispatch = DispatchId(99);
  rep.plan_id = dispatched.plan_id;
  rep.plan_generation = dispatched.generation;
  rep.reporter_boot = dispatched.target_boot;
  rep.outcome = ExecutionOutcome::SUCCEEDED;
  planner.report_execution(rep);
  auto after = *planner.plans().begin();
  RP_CHECK(after.state == PlanLifecycleState::SUCCEEDED);
  RP_CHECK(after.execution_outcome == ExecutionOutcome::SUCCEEDED);
}

// ---------------------------------------------------------------------------
RP_TEST(stale_completion_rejected) {
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, make_memory_persistence_store());
  make_world(planner);

  RecoveryRequest req = make_request();
  auto decision = planner.plan(req, {make_candidate(RecoveryStrategy::RESTORE, 1)});
  auto authorized = planner.authorize_plan(*decision.plan);
  // An executor that does NOT report (dispatch stays EXECUTING); use a no-op.
  ReferenceExecutor exec([](DispatchId, ExecutionOutcome, ProgressUnits, Duration, std::string) {});
  auto dispatched = planner.dispatch_plan(authorized, exec);
  RP_CHECK(dispatched.state == PlanLifecycleState::EXECUTING);

  // Stale reporter boot id -> rejected, plan unchanged.
  ExecutionReport rep;
  rep.plan_id = dispatched.plan_id;
  rep.plan_generation = dispatched.generation;
  rep.reporter_boot = WorkerBootId(999999);  // stale
  rep.outcome = ExecutionOutcome::SUCCEEDED;
  planner.report_execution(rep);
  auto _plans = planner.plans(); auto& after = _plans[0];
  RP_CHECK(after.state == PlanLifecycleState::EXECUTING);
}

// ---------------------------------------------------------------------------
RP_TEST(stale_plan_rejected_on_revalidate) {
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, make_memory_persistence_store());
  make_world(planner);

  RecoveryRequest req = make_request();
  auto decision = planner.plan(req, {make_candidate(RecoveryStrategy::RESTORE, 1)});
  auto authorized = planner.authorize_plan(*decision.plan);

  // The target worker disappears / restarts under a new boot.
  planner.unregister_worker(kWorkerA);
  RP_EXPECT_THROW(planner.revalidate_plan(authorized), ErrorCode::STALE_AUTHORITY);
}

// ---------------------------------------------------------------------------
RP_TEST(cancel_plan_before_dispatch) {
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, make_memory_persistence_store());
  make_world(planner);

  RecoveryRequest req = make_request();
  auto decision = planner.plan(req, {make_candidate(RecoveryStrategy::RESTART, 1)});
  auto plan = *decision.plan;
  bool ok = planner.cancel_plan(plan.plan_id);
  RP_CHECK(ok);
  auto _plans = planner.plans(); auto& after = _plans[0];
  RP_CHECK(after.state == PlanLifecycleState::CANCELLED);
}

// ---------------------------------------------------------------------------
RP_TEST(late_completion_from_invalidated_plan_rejected) {
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, make_memory_persistence_store());
  make_world(planner);

  RecoveryRequest req = make_request();
  auto decision = planner.plan(req, {make_candidate(RecoveryStrategy::RESTORE, 1)});
  auto authorized = planner.authorize_plan(*decision.plan);
  planner.invalidate_plan(authorized.plan_id);

  ReferenceExecutor exec(
      [&planner](DispatchId d, ExecutionOutcome o, ProgressUnits r, Duration e, std::string s) {
        planner.on_adapter_completion(d, o, r, e, std::move(s));
      });
  // Dispatch a now-invalidated plan must refuse.
  RP_EXPECT_THROW(planner.dispatch_plan(authorized, exec), ErrorCode::STALE_AUTHORITY);
}

// ---------------------------------------------------------------------------
RP_TEST(new_plan_supersedes_older_plan) {
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, make_memory_persistence_store());
  make_world(planner);

  RecoveryRequest req = make_request();
  auto d1 = planner.plan(req, {make_candidate(RecoveryStrategy::RESTORE, 1)});
  auto d2 = planner.plan(req, {make_candidate(RecoveryStrategy::RESTART, 2)});
  RP_CHECK(d1.plan->candidate_id != d2.plan->candidate_id);

  // The older plan is superseded and can no longer be authorized.
  RP_EXPECT_THROW(planner.authorize_plan(*d1.plan), ErrorCode::INVALID_INPUT);
  auto _plans = planner.plans(); auto& after = _plans[0];
  RP_CHECK(after.state == PlanLifecycleState::SUPERSEDED);
}

// ---------------------------------------------------------------------------
// Regression tests: dispatch identity, hard objectives, expiry, adapter failure.
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

// An executor whose backend call fails.
class ThrowingExecutor final : public IRecoveryExecutor {
 public:
  RecoveryReceipt dispatch(const RecoveryDispatch& d) override {
    last = d.dispatch_id;
    throw std::runtime_error("backend unavailable");
  }
  const char* name() const noexcept override { return "throwing"; }
  DispatchId last;
};

RecoveryPlan plan_by_id(const RecoveryPlanner& planner, RecoveryPlanId id) {
  for (const auto& p : planner.plans())
    if (p.plan_id == id) return p;
  throw rptest::Fail("plan not found");
}
}

// ---------------------------------------------------------------------------
RP_TEST(dispatch_ids_are_unique_across_plans) {
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, make_memory_persistence_store());
  make_world(planner);

  RecoveryRequest r1 = make_request();
  RecoveryRequest r2 = make_request();
  r2.request_id = RecoveryRequestId(2);
  auto d1 = planner.plan(r1, {make_candidate(RecoveryStrategy::RESTORE, 1)});
  auto d2 = planner.plan(r2, {make_candidate(RecoveryStrategy::RESTORE, 2)});
  auto p1 = planner.authorize_plan(*d1.plan);
  auto p2 = planner.authorize_plan(*d2.plan);

  RecordingExecutor exec;
  planner.dispatch_plan(p1, exec);
  planner.dispatch_plan(p2, exec);
  RP_CHECK(exec.seen.size() == 2);
  RP_CHECK(exec.seen[0].dispatch_id != exec.seen[1].dispatch_id);

  // The completion of the first dispatch belongs to the first plan only.
  planner.on_adapter_completion(exec.seen[0].dispatch_id, ExecutionOutcome::SUCCEEDED,
                                ProgressUnits(90), Duration(10), "done");
  RP_CHECK(plan_by_id(planner, p1.plan_id).state == PlanLifecycleState::SUCCEEDED);
  RP_CHECK(plan_by_id(planner, p2.plan_id).state == PlanLifecycleState::EXECUTING);
}

// ---------------------------------------------------------------------------
RP_TEST(fallback_plan_violating_a_hard_objective_is_not_executable) {
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, make_memory_persistence_store());
  make_world(planner);

  // Every candidate misses the deadline: the fallback is a labeled alternative.
  RecoveryRequest req = make_request();
  req.recovery_deadline = Timestamp(500);   // the candidate needs 5000
  auto d = planner.plan(req, {make_candidate(RecoveryStrategy::RESTORE, 1)});
  RP_CHECK(d.outcome == PlanOutcome::NO_DEADLINE_FEASIBLE_PLAN);
  RP_CHECK(d.plan.has_value());
  RP_EXPECT_THROW(planner.authorize_plan(*d.plan), ErrorCode::INFEASIBLE_CANDIDATE);
  RP_CHECK(plan_by_id(planner, d.plan->plan_id).state == PlanLifecycleState::PLANNED);

  // Same rule for the tolerated progress loss.
  RecoveryRequest req2 = make_request();
  req2.request_id = RecoveryRequestId(2);
  req2.max_tolerated_progress_loss = ProgressUnits(10);
  auto lossy = make_candidate(RecoveryStrategy::RESTART, 2);
  lossy.cost->expected_lost_progress = ProgressUnits(50);
  auto d2 = planner.plan(req2, {lossy});
  RP_CHECK(d2.outcome == PlanOutcome::NO_VALID_PLAN);
  RP_CHECK(d2.plan.has_value());
  RP_EXPECT_THROW(planner.authorize_plan(*d2.plan), ErrorCode::INFEASIBLE_CANDIDATE);
  RP_EXPECT_THROW(planner.revalidate_plan(*d2.plan), ErrorCode::STALE_AUTHORITY);
  RP_CHECK(plan_by_id(planner, d2.plan->plan_id).state == PlanLifecycleState::PLANNED);
}

// ---------------------------------------------------------------------------
RP_TEST(unknown_progress_loss_does_not_meet_a_tolerance) {
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, make_memory_persistence_store());
  make_world(planner);

  RecoveryRequest req = make_request();
  req.max_tolerated_progress_loss = ProgressUnits(10);
  auto c = make_candidate(RecoveryStrategy::RESTORE, 1);
  c.cost->expected_lost_progress.reset();   // nobody measured the loss
  auto d = planner.plan(req, {c});
  RP_CHECK(d.outcome == PlanOutcome::NO_VALID_PLAN);
  RP_CHECK(d.ranked.size() == 1);
  RP_CHECK(!d.ranked[0].economics.progress_loss_within_tolerance);
}

// ---------------------------------------------------------------------------
RP_TEST(plan_expires_once_its_deadline_has_passed) {
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, make_memory_persistence_store());
  make_world(planner);

  RecoveryRequest req = make_request();
  req.recovery_deadline = Timestamp(100000);
  auto d = planner.plan(req, {make_candidate(RecoveryStrategy::RESTORE, 1)});
  RP_CHECK(d.outcome == PlanOutcome::PLAN_FOUND);
  auto authorized = planner.authorize_plan(*d.plan);

  // Nothing was dispatched before the deadline passed.
  clock->set(Timestamp(200000));
  RP_EXPECT_THROW(planner.revalidate_plan(authorized), ErrorCode::STALE_AUTHORITY);
  RP_CHECK(plan_by_id(planner, authorized.plan_id).state == PlanLifecycleState::AUTHORIZED);
  RecordingExecutor exec;
  RP_EXPECT_THROW(planner.dispatch_plan(authorized, exec), ErrorCode::STALE_AUTHORITY);
  RP_CHECK(exec.seen.empty());
  RP_CHECK(plan_by_id(planner, authorized.plan_id).state == PlanLifecycleState::EXPIRED);

  // A plan that was never authorized expires the same way.
  clock->set(Timestamp(0));
  RecoveryRequest req2 = make_request();
  req2.request_id = RecoveryRequestId(2);
  req2.recovery_deadline = Timestamp(100000);
  auto d2 = planner.plan(req2, {make_candidate(RecoveryStrategy::RESTORE, 2)});
  RP_CHECK(d2.outcome == PlanOutcome::PLAN_FOUND);
  clock->set(Timestamp(200000));
  RP_EXPECT_THROW(planner.authorize_plan(*d2.plan), ErrorCode::STALE_AUTHORITY);
  RP_CHECK(plan_by_id(planner, d2.plan->plan_id).state == PlanLifecycleState::EXPIRED);
}

// ---------------------------------------------------------------------------
RP_TEST(executor_exception_does_not_strand_the_plan) {
  auto clock = std::make_shared<TestClock>(Timestamp(0));
  RecoveryPlanner planner(clock, make_memory_persistence_store());
  make_world(planner);

  auto d = planner.plan(make_request(), {make_candidate(RecoveryStrategy::RESTORE, 1)});
  auto authorized = planner.authorize_plan(*d.plan);
  ThrowingExecutor exec;
  bool threw = false;
  try {
    planner.dispatch_plan(authorized, exec);
  } catch (const std::exception&) {
    threw = true;
  }
  RP_CHECK(threw);
  auto after = plan_by_id(planner, authorized.plan_id);
  RP_CHECK(after.state == PlanLifecycleState::FAILED);
  RP_CHECK(after.outcome_published == false);

  // A completion for the dispatch the adapter never took changes nothing.
  planner.on_adapter_completion(exec.last, ExecutionOutcome::SUCCEEDED, ProgressUnits(90),
                                Duration(1), "late");
  RP_CHECK(plan_by_id(planner, authorized.plan_id).state == PlanLifecycleState::FAILED);
}
