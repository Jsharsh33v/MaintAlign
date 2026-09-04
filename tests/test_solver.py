"""Solver tests for the Tier 1 rework.

Task slots      Default to the physical packing bound (a valid bound), so
                OPTIMAL means optimal; the heuristic cap survives only as an
                explicitly labelled restricted model.
Chain outages   Priced per period: three machines down together pay one
                outage, not 1.5.
Imperfect PM    ``repair_factor`` reaches the objective, and the objective
                still equals ``deterministic_cost`` of the returned schedule.
"""

import pytest

from core.costing import deterministic_cost
from core.instance import MachineSpec, ProblemInstance, ProductionChain
from core.solver import (
    _compute_max_tasks_heuristic,
    _compute_max_tasks_physical,
    solve,
)
from utils.generator import generate_small, generate_tiny


def _tolerance(instance, num_tasks):
    """Failure tables are truncated to the cent, twice per gap."""
    return 0.01 * (2 * (num_tasks + instance.num_machines) + 2)


def _machine(mid, d=1, g=1, W=8, horizon=12):
    return MachineSpec(id=mid, name=f"M{mid}", maintenance_duration=d, pm_cost=100,
                       cm_cost=5000, production_value=50, weibull_beta=2.5,
                       weibull_eta=6.0, max_interval=W, min_gap=g)


class TestTaskSlotBound:

    @pytest.mark.parametrize("d,g,H", [(1, 1, 12), (2, 1, 20), (1, 0, 7), (3, 2, 30), (2, 3, 13)])
    def test_physical_bound_is_exactly_what_fits(self, d, g, H):
        m = _machine(0, d=d, g=g, W=H, horizon=H)
        n = _compute_max_tasks_physical(m, H)
        # n windows fit: start j*(d+g), last one ends at (n-1)*(d+g) + d <= H
        assert (n - 1) * (d + g) + d <= H
        # n + 1 do not
        assert n * (d + g) + d > H

    def test_heuristic_bound_never_exceeds_the_physical_bound(self):
        for seed in range(5):
            inst = generate_small(seed=seed)
            for i, m in enumerate(inst.machines):
                assert (_compute_max_tasks_heuristic(m, inst.horizon, i)
                        <= _compute_max_tasks_physical(m, inst.horizon))

    def test_default_model_is_full_and_restricted_is_labelled(self):
        inst = generate_tiny(seed=0)
        full = solve(inst, time_limit_seconds=10)
        restricted = solve(inst, time_limit_seconds=10, task_slot_bound="heuristic")
        assert full.model_variant == "full"
        assert restricted.model_variant == "restricted-slots"
        assert "RESTRICTED" in restricted.summary()
        if full.status == "OPTIMAL" and restricted.status == "OPTIMAL":
            # the full model can express everything the restricted one can
            assert full.objective_value <= restricted.objective_value + 1e-6

    def test_unknown_bound_is_rejected(self):
        with pytest.raises(ValueError):
            solve(generate_tiny(seed=0), time_limit_seconds=1, task_slot_bound="magic")

    def test_infeasible_result_still_carries_the_variant(self):
        # min_gap 5 > max_interval 2 over a 12-period horizon: consecutive
        # PMs must be both >= 5 and <= 2 apart, so no schedule exists.
        m = _machine(0, d=1, g=5, W=2, horizon=12)
        inst = ProblemInstance("bad", 1, 1, 12, [m])
        res = solve(inst, time_limit_seconds=5)
        assert res.status == "INFEASIBLE"
        assert res.model_variant == "full"


class TestSymmetryBreaking:

    @staticmethod
    def _different_chain_instance():
        """Machines 0 and 1 look identical but are not interchangeable.

        Their chain partners have different maintenance patterns and chain
        economics, so the unrestricted optimum needs M0's first PM after M1's.
        """
        machines = [
            MachineSpec(0, "M0", 2, 100, 500, 0, 1.5, 4, 7, 1),
            MachineSpec(1, "M1", 2, 100, 500, 0, 1.5, 4, 7, 1),
            MachineSpec(2, "M2", 1, 20, 2000, 0, 2.0, 4, 6, 1),
            MachineSpec(3, "M3", 1, 100, 2000, 0, 3.0, 2, 1, 1),
        ]
        chains = [
            ProductionChain(0, "A", [0, 2], chain_value=10, retooling_cost=100),
            ProductionChain(1, "B", [1, 3], chain_value=100, retooling_cost=0),
        ]
        return ProblemInstance(
            "different-chains", 4, 2, 8, machines,
            chains=chains, blocked_periods=[2],
        )

    def test_different_chains_are_not_treated_as_symmetric(self):
        inst = self._different_chain_instance()
        with_symmetry = solve(
            inst, time_limit_seconds=5, num_workers=1,
            use_symmetry_breaking=True,
        )
        without_symmetry = solve(
            inst, time_limit_seconds=5, num_workers=1,
            use_symmetry_breaking=False,
        )

        assert with_symmetry.status == without_symmetry.status == "OPTIMAL"
        assert with_symmetry.objective_value == pytest.approx(
            without_symmetry.objective_value
        )


class TestFailureCostFloor:

    def test_concave_hazard_does_not_cut_off_a_cheaper_schedule(self):
        """For beta < 1, the Jensen balanced-gap expression is not a floor."""
        machine = MachineSpec(
            0, "M0", 1, 1000, 1100, 0, 0.5, 1.0, 6, 0,
        )
        inst = ProblemInstance("concave-hazard", 1, 1, 10, [machine])

        # One PM at t=3 satisfies both max-interval constraints: the head gap
        # is 3 and the tail gap is 6. An optimum may not cost more than this
        # explicitly feasible schedule.
        feasible_cost = deterministic_cost(inst, {0: [3]}).total
        result = solve(
            inst, time_limit_seconds=5, num_workers=1,
            use_symmetry_breaking=False,
        )

        assert result.status == "OPTIMAL"
        assert result.objective_value <= feasible_cost + _tolerance(inst, 1)


class TestSharedOutage:

    def _tri(self, K=3):
        machines = [_machine(i) for i in range(3)]
        chain = ProductionChain(0, "line", [0, 1, 2], chain_value=900, retooling_cost=400)
        return ProblemInstance("tri", 3, K, 12, machines, chains=[chain])

    def test_three_machines_down_together_pay_one_outage(self):
        inst = self._tri()
        res = solve(inst, time_limit_seconds=20)
        assert res.status == "OPTIMAL"
        det = deterministic_cost(inst, res.machine_schedules)
        cc = det.chain_costs[0]
        # objective == cost model, and the chain's cost is V × down periods
        # + R × restarts — not 1.5 × per-event as the 50% heuristic charged
        assert abs(res.objective_value - det.total) <= _tolerance(inst, len(res.tasks))
        assert cc["prod_loss"] == pytest.approx(900 * cc["down_periods"])
        assert cc["retooling"] == pytest.approx(400 * cc["num_runs"])
        # with three technicians the optimum clusters the chain's maintenance
        assert res.machine_schedules[0] == res.machine_schedules[1] == res.machine_schedules[2]
        assert cc["down_periods"] == len(res.machine_schedules[0])
        full = deterministic_cost(inst, res.machine_schedules, chain_grouping=False)
        assert full.production_loss == pytest.approx(3 * cc["prod_loss"])

    def test_per_task_shares_sum_to_the_chain_totals(self):
        inst = self._tri()
        res = solve(inst, time_limit_seconds=20)
        assert sum(t.cost_prod_loss for t in res.tasks) == pytest.approx(res.total_production_loss)
        assert sum(t.cost_retooling for t in res.tasks) == pytest.approx(res.total_retooling_cost)

    def test_one_technician_forces_serial_outages(self):
        inst = self._tri(K=1)
        res = solve(inst, time_limit_seconds=20)
        assert res.status in ("OPTIMAL", "FEASIBLE")
        det = deterministic_cost(inst, res.machine_schedules)
        assert det.chain_costs[0]["down_periods"] == len(res.tasks)
        assert det.total <= res.objective_value + _tolerance(inst, len(res.tasks))


class TestImperfectRepair:

    def test_objective_matches_cost_model_with_imperfect_repair(self):
        inst = generate_tiny(seed=1)
        for m in inst.machines:
            m.repair_factor = 0.7
        res = solve(inst, time_limit_seconds=15)
        assert res.status in ("OPTIMAL", "FEASIBLE")
        det = deterministic_cost(inst, res.machine_schedules)
        tol = _tolerance(inst, len(res.tasks))
        if res.status == "OPTIMAL":
            assert abs(res.objective_value - det.total) <= tol
        else:
            assert det.total <= res.objective_value + tol

    def test_repair_factor_is_not_ignored(self):
        """For every schedule an imperfect PM leaves more risk behind, so the
        optimum under r < 1 cannot be cheaper than under perfect repair."""
        perfect = generate_tiny(seed=1)
        imperfect = generate_tiny(seed=1)
        for m in imperfect.machines:
            m.repair_factor = 0.7
        a = solve(perfect, time_limit_seconds=15)
        b = solve(imperfect, time_limit_seconds=15)
        assert a.status == "OPTIMAL" and b.status == "OPTIMAL"
        assert b.objective_value >= a.objective_value - 1e-6
        # and the perfect-repair schedule really does cost more when r < 1
        assert (deterministic_cost(imperfect, a.machine_schedules).total
                > deterministic_cost(perfect, a.machine_schedules).total)
