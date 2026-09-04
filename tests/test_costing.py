"""Cost-accounting reconciliation tests.

These exist because the solver, the baselines and the simulator each used
to price a schedule differently. The result was that "savings vs best
baseline" divided a discounted optimiser cost by an undiscounted baseline
cost, overstating the advantage by 5-16 percentage points.

The contract these tests lock in:

  1. The CP-SAT objective equals ``deterministic_cost`` of the schedule it
     returned. If someone changes the objective without changing the cost
     model (or the reverse), this fails.
  2. Every ``SolverResult`` -- optimiser or baseline -- has a cost
     breakdown whose four components sum to its own ``objective_value``.
  3. All schedules are priced on one basis, so comparing them is valid.
"""

import pytest

from core.baseline import ALL_STRATEGIES, fixed_interval_schedule
from core.costing import (
    CostBreakdown,
    attribute_chain_costs,
    deterministic_cost,
    savings_vs,
)
from core.instance import MachineSpec, ProblemInstance, ProductionChain
from core.solver import solve
from utils.generator import generate_small, generate_tiny

SEEDS = [0, 1, 2]


def rounding_tolerance(instance, num_tasks: int) -> float:
    """Slack for CP-SAT's integer cost scaling.

    ``core.solver`` stores every failure-cost table entry as
    ``int(cost * COST_SCALE)`` and charges each gap as T[a + g] - T[a], so
    the objective can sit up to two cents below the exact figure per gap
    (gaps = tasks + machines). Chain costs are exact integers. This bounds
    the accumulated truncation without being loose enough to hide a real
    accounting mismatch (the defect this guards against was worth
    thousands of dollars).
    """
    return 0.01 * (2 * (num_tasks + instance.num_machines) + 2)


class TestObjectiveReconciliation:
    """The headline guard: the objective and the cost model agree."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_solver_objective_matches_recomputed_cost(self, seed):
        inst = generate_tiny(seed=seed)
        res = solve(inst, time_limit_seconds=10)
        assert res.status in ("OPTIMAL", "FEASIBLE")

        recomputed = deterministic_cost(inst, res.machine_schedules).total
        tol = rounding_tolerance(inst, len(res.tasks))

        if res.status == "OPTIMAL":
            assert abs(recomputed - res.objective_value) <= tol, (
                f"objective ${res.objective_value:,.2f} but the same "
                f"schedule prices at ${recomputed:,.2f}"
            )
        else:
            # A time-limited incumbent may leave an outage indicator raised
            # that no task justifies, so the exact price can only be lower.
            assert recomputed <= res.objective_value + tol

    def test_solver_breakdown_sums_to_objective(self):
        """The defect that shipped in baseline_comparison.csv."""
        inst = generate_small(seed=0)
        res = solve(inst, time_limit_seconds=20)
        parts = (res.total_pm_cost + res.total_production_loss
                 + res.total_retooling_cost + res.total_failure_cost)
        tol = rounding_tolerance(inst, len(res.tasks))
        assert abs(parts - res.objective_value) <= tol

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_baseline_breakdown_sums_to_objective(self, strategy):
        inst = generate_small(seed=0)
        res = fixed_interval_schedule(inst, strategy)
        parts = (res.total_pm_cost + res.total_production_loss
                 + res.total_retooling_cost + res.total_failure_cost)
        assert parts == pytest.approx(res.objective_value)

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_baseline_objective_matches_cost_model(self, strategy):
        """Baselines are priced by the same function as the optimiser."""
        inst = generate_small(seed=0)
        res = fixed_interval_schedule(inst, strategy)
        recomputed = deterministic_cost(inst, res.machine_schedules).total
        assert recomputed == pytest.approx(res.objective_value)


class TestCostModel:

    def test_total_is_the_sum_of_its_parts(self):
        bd = CostBreakdown(pm_cost=10, production_loss=20,
                           retooling_cost=5, failure_cost=2.5)
        assert bd.total == pytest.approx(37.5)

    def test_empty_schedule_costs_only_failure_risk(self):
        inst = generate_tiny(seed=0)
        empty = {m.id: [] for m in inst.machines}
        bd = deterministic_cost(inst, empty)
        assert bd.num_tasks == 0
        assert bd.pm_cost == 0
        assert bd.production_loss == 0
        assert bd.retooling_cost == 0
        expected = sum(m.expected_failure_cost(inst.horizon)
                       for m in inst.machines)
        assert bd.failure_cost == pytest.approx(expected)

    def test_missing_machines_are_treated_as_unmaintained(self):
        inst = generate_tiny(seed=0)
        assert (deterministic_cost(inst, {}).total
                == pytest.approx(deterministic_cost(
                    inst, {m.id: [] for m in inst.machines}).total))

    def test_full_basis_is_never_cheaper_than_grouped_basis(self):
        inst = generate_small(seed=0)
        sched = fixed_interval_schedule(inst, "half_max").machine_schedules
        grouped = deterministic_cost(inst, sched, chain_grouping=True).total
        full = deterministic_cost(inst, sched, chain_grouping=False).total
        assert full >= grouped

    def test_grouping_credit_requires_a_real_overlap(self):
        """Two chain machines maintained at the same time share the outage."""
        inst = generate_small(seed=0)
        if not inst.chains:
            pytest.skip("instance has no production chain")
        chain = inst.chains[0]
        if len(chain.machine_ids) < 2:
            pytest.skip("chain has fewer than two machines")
        a, b = chain.machine_ids[0], chain.machine_ids[1]

        together = {m.id: [] for m in inst.machines}
        together[a] = [2]
        together[b] = [2]

        apart = {m.id: [] for m in inst.machines}
        apart[a] = [2]
        apart[b] = [2 + inst.machines[a].maintenance_duration
                    + inst.machines[b].maintenance_duration + 2]

        cost_together = deterministic_cost(inst, together)
        cost_apart = deterministic_cost(inst, apart)
        assert cost_together.num_grouped_tasks == 2
        assert cost_apart.num_grouped_tasks == 0
        assert (cost_together.retooling_cost
                < cost_apart.retooling_cost)


def _chain_instance(num_machines=3, d=1, V=900, R=400, horizon=12):
    machines = [
        MachineSpec(id=i, name=f"M{i}", maintenance_duration=d, pm_cost=100,
                    cm_cost=5000, production_value=50, weibull_beta=2.5,
                    weibull_eta=6.0, max_interval=horizon, min_gap=1)
        for i in range(num_machines)
    ]
    chain = ProductionChain(0, "line", list(range(num_machines)),
                            chain_value=V, retooling_cost=R)
    return ProblemInstance("chain", num_machines, num_machines, horizon,
                           machines, chains=[chain])


class TestSharedOutageModel:
    """A chain outage is paid once per period and once per restart."""

    def test_three_machines_down_together_pay_one_outage(self):
        inst = _chain_instance(3)
        sched = {0: [4], 1: [4], 2: [4]}
        bd = deterministic_cost(inst, sched)
        assert bd.production_loss == pytest.approx(900)     # not 3 × 450
        assert bd.retooling_cost == pytest.approx(400)      # not 3 × 200
        assert bd.chain_costs[0]["down_periods"] == 1
        assert bd.chain_costs[0]["num_runs"] == 1
        assert bd.num_grouped_tasks == 3

    def test_back_to_back_windows_form_one_run(self):
        inst = _chain_instance(2, d=2)
        sched = {0: [3], 1: [5]}              # [3,5) then [5,7): no gap
        bd = deterministic_cost(inst, sched)
        assert bd.chain_costs[0]["down_periods"] == 4
        assert bd.chain_costs[0]["num_runs"] == 1
        assert bd.retooling_cost == pytest.approx(400)
        assert bd.production_loss == pytest.approx(900 * 4)
        assert bd.num_grouped_tasks == 0      # they touch but do not overlap

    def test_separate_windows_are_separate_outages(self):
        inst = _chain_instance(2)
        sched = {0: [2], 1: [8]}
        bd = deterministic_cost(inst, sched)
        assert bd.chain_costs[0]["num_runs"] == 2
        assert bd.retooling_cost == pytest.approx(800)
        assert bd.production_loss == pytest.approx(1800)

    def test_full_basis_charges_every_event(self):
        inst = _chain_instance(3)
        sched = {0: [4], 1: [4], 2: [4]}
        full = deterministic_cost(inst, sched, chain_grouping=False)
        assert full.production_loss == pytest.approx(3 * 900)
        assert full.retooling_cost == pytest.approx(3 * 400)

    def test_attribution_sums_to_the_chain_totals(self):
        inst = generate_small(seed=0)
        sched = fixed_interval_schedule(inst, "half_max").machine_schedules
        for basis in (True, False):
            bd = deterministic_cost(inst, sched, chain_grouping=basis)
            shares = attribute_chain_costs(inst, sched, chain_grouping=basis)
            assert sum(p for p, _ in shares.values()) == pytest.approx(
                sum(cc["prod_loss"] for cc in bd.chain_costs.values()))
            assert sum(r for _, r in shares.values()) == pytest.approx(
                sum(cc["retooling"] for cc in bd.chain_costs.values()))

    def test_per_machine_costs_sum_to_the_total(self):
        inst = generate_small(seed=0)
        sched = fixed_interval_schedule(inst, "analytical").machine_schedules
        bd = deterministic_cost(inst, sched)
        assert sum(bd.per_machine.values()) == pytest.approx(bd.total)


class TestImperfectRepairCosting:
    """repair_factor changes the price of a schedule, and only upward."""

    def test_perfect_repair_is_the_special_case(self):
        inst = generate_tiny(seed=0)
        sched = fixed_interval_schedule(inst, "half_max").machine_schedules
        before = deterministic_cost(inst, sched).failure_cost
        for m in inst.machines:
            m.repair_factor = 1.0
        assert deterministic_cost(inst, sched).failure_cost == pytest.approx(before)

    def test_imperfect_repair_leaves_more_risk_behind(self):
        inst = generate_tiny(seed=0)
        sched = fixed_interval_schedule(inst, "half_max").machine_schedules
        perfect = deterministic_cost(inst, sched)
        for m in inst.machines:
            m.repair_factor = 0.5
        imperfect = deterministic_cost(inst, sched)
        assert imperfect.failure_cost > perfect.failure_cost
        assert imperfect.pm_cost == perfect.pm_cost
        assert imperfect.production_loss == perfect.production_loss


class TestSavings:

    def test_a_schedule_saves_nothing_against_itself(self):
        inst = generate_tiny(seed=0)
        sched = fixed_interval_schedule(inst, "half_max").machine_schedules
        assert savings_vs(inst, sched, sched) == pytest.approx(0.0)

    def test_savings_is_symmetric_in_sign(self):
        inst = generate_small(seed=0)
        a = fixed_interval_schedule(inst, "max_interval").machine_schedules
        b = fixed_interval_schedule(inst, "half_max").machine_schedules
        assert savings_vs(inst, a, b) * savings_vs(inst, b, a) <= 0
