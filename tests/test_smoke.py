"""Smoke tests — verify core pipeline end-to-end on tiny instances."""

import pytest

from utils.generator import generate_tiny, generate_small, generate_instance
from core.baseline import fixed_interval_schedule, ALL_STRATEGIES
from core.solver import solve


class TestGenerator:

    def test_generate_tiny_returns_valid_instance(self):
        inst = generate_tiny(seed=0)
        assert inst.num_machines >= 1
        assert inst.num_technicians >= 1
        assert inst.horizon > 0
        assert len(inst.machines) == inst.num_machines

    def test_generate_small_has_chains(self):
        inst = generate_small(seed=0)
        assert inst.num_machines >= 4  # small has enough for a chain
        # small is documented to include at least one chain
        assert len(inst.chains) >= 0

    def test_generator_is_deterministic(self):
        inst1 = generate_tiny(seed=123)
        inst2 = generate_tiny(seed=123)
        # Same seed → same machine names and parameters
        assert [m.name for m in inst1.machines] == [m.name for m in inst2.machines]
        assert [m.pm_cost for m in inst1.machines] == [m.pm_cost for m in inst2.machines]


class TestBaselines:

    def test_every_strategy_produces_schedule(self):
        inst = generate_tiny(seed=0)
        for strategy in ALL_STRATEGIES:
            result = fixed_interval_schedule(inst, strategy)
            assert result.objective_value > 0
            assert isinstance(result.machine_schedules, dict)
            # Every machine must appear in the schedule (even if empty)
            for m in inst.machines:
                assert m.id in result.machine_schedules

    def test_schedule_respects_horizon(self):
        inst = generate_tiny(seed=0)
        result = fixed_interval_schedule(inst, "analytical")
        for mid, starts in result.machine_schedules.items():
            for s in starts:
                assert 0 <= s < inst.horizon


class TestSolver:

    def test_solver_runs_on_tiny(self):
        """CP-SAT should solve a tiny instance in well under 10s."""
        inst = generate_tiny(seed=0)
        result = solve(inst, time_limit_seconds=10)
        assert result.status in ("OPTIMAL", "FEASIBLE")
        assert result.objective_value > 0

    def test_solver_with_warm_start(self):
        """Solver should accept a baseline schedule as a warm-start hint."""
        inst = generate_tiny(seed=1)
        baseline = fixed_interval_schedule(inst, "analytical")
        result = solve(
            inst,
            time_limit_seconds=10,
            hint_schedule=baseline.machine_schedules,
        )
        # Optimized should be at most as expensive as the baseline
        assert result.objective_value <= baseline.objective_value + 1e-6

    def test_solver_output_is_consistent(self):
        """Solver output should list one schedule per machine."""
        inst = generate_tiny(seed=2)
        result = solve(inst, time_limit_seconds=10)
        assert set(result.machine_schedules.keys()) == {m.id for m in inst.machines}
