"""Monte Carlo simulator tests.

Failure process   The simulator samples the same failure process the
                  objective assumes: minimal repair, so the mean number of
                  failures over a bare gap of length g converges to (g/η)^β.
One cost basis    c_cm is all-in. A single-machine instance prices
                  identically under the objective and under Monte Carlo,
                  within MC error.
Imperfect PM      Kijima type I repair is honoured by the simulator too.
Random streams    Each machine has its own stream (common random numbers).
"""

import math
import statistics

import pytest

from analysis.simulator import simulate_schedule
from core.baseline import fixed_interval_schedule
from core.costing import deterministic_cost
from core.instance import MachineSpec, ProblemInstance
from utils.generator import generate_small

N_SIMS = 2500
Z = 4.0  # accept within 4 standard errors: a false alarm every ~16,000 runs


def _single_machine(beta=2.2, eta=7.0, horizon=20, repair_factor=1.0):
    m = MachineSpec(id=0, name="m", maintenance_duration=1, pm_cost=100,
                    cm_cost=1000, production_value=50, weibull_beta=beta,
                    weibull_eta=eta, max_interval=horizon, min_gap=1,
                    repair_factor=repair_factor)
    return ProblemInstance("one", 1, 1, horizon, [m])


def _mean_se(values):
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    return mean, se


class TestFailureProcess:

    def test_bare_gap_mean_failures_converge_to_the_power_law(self):
        inst = _single_machine()
        m = inst.machines[0]
        counts = [simulate_schedule(inst, {0: []}, seed=s).num_failures
                  for s in range(N_SIMS)]
        mean, se = _mean_se(counts)
        theory = (inst.horizon / m.weibull_eta) ** m.weibull_beta
        assert abs(mean - theory) <= Z * se, (mean, se, theory)

    def test_corrective_repair_does_not_reset_the_age(self):
        """A renewal process (age reset after every CM) would give far fewer
        failures than the NHPP over a long gap with a steep hazard."""
        inst = _single_machine(beta=3.0, eta=5.0, horizon=20)
        m = inst.machines[0]
        counts = [simulate_schedule(inst, {0: []}, seed=s).num_failures
                  for s in range(N_SIMS)]
        mean, se = _mean_se(counts)
        theory = (inst.horizon / m.weibull_eta) ** m.weibull_beta   # = 64
        assert abs(mean - theory) <= Z * se
        # renewal with Weibull(3, 5) lifetimes: mean life ≈ 4.46 ⇒ ≈ 4.5 failures
        assert mean > 40

    def test_pm_resets_the_age_and_gaps_add_up(self):
        inst = _single_machine(horizon=20)
        m = inst.machines[0]
        sched = {0: [8]}
        counts = [simulate_schedule(inst, sched, seed=s).num_failures
                  for s in range(N_SIMS)]
        mean, se = _mean_se(counts)
        theory = m.expected_failures(8) + m.expected_failures(20 - 9)
        assert abs(mean - theory) <= Z * se

    def test_imperfect_repair_follows_the_virtual_age(self):
        inst = _single_machine(horizon=20, repair_factor=0.5)
        m = inst.machines[0]
        sched = {0: [8]}
        counts = [simulate_schedule(inst, sched, seed=s).num_failures
                  for s in range(N_SIMS)]
        mean, se = _mean_se(counts)
        va = m.virtual_age_after_pm(8)   # 4
        theory = m.expected_failures(8) + m.expected_failures_imperfect(20 - 9, va)
        assert va == 4
        assert abs(mean - theory) <= Z * se
        assert theory > m.expected_failures(8) + m.expected_failures(11)


class TestOneCostBasis:

    def test_single_machine_prices_identically_under_objective_and_monte_carlo(self):
        inst = _single_machine()
        sched = {0: [6, 13]}
        expected = deterministic_cost(inst, sched).total
        costs = [simulate_schedule(inst, sched, seed=s).total_cost
                 for s in range(N_SIMS)]
        mean, se = _mean_se(costs)
        assert abs(mean - expected) <= Z * se, (mean, se, expected)

    def test_single_machine_with_imperfect_repair_prices_identically(self):
        inst = _single_machine(repair_factor=0.6)
        sched = {0: [6, 13]}
        expected = deterministic_cost(inst, sched).total
        costs = [simulate_schedule(inst, sched, seed=s).total_cost
                 for s in range(N_SIMS)]
        mean, se = _mean_se(costs)
        assert abs(mean - expected) <= Z * se

    def test_deterministic_part_is_the_costing_functions(self):
        inst = generate_small(seed=0)
        sched = fixed_interval_schedule(inst, "half_max").machine_schedules
        det = deterministic_cost(inst, sched)
        sim = simulate_schedule(inst, sched, seed=1)
        assert sim.total_pm_cost == pytest.approx(det.pm_cost)
        assert sim.total_production_loss == pytest.approx(det.production_loss)
        assert sim.total_retooling_cost == pytest.approx(det.retooling_cost)
        assert sim.total_cost == pytest.approx(
            det.deterministic_part + sim.num_failures * 0 + sim.total_cm_cost)
        assert sum(e.cost for e in sim.events) == pytest.approx(sim.total_cost)

    def test_a_failure_costs_cm_and_nothing_else(self):
        inst = generate_small(seed=0)
        sched = {m.id: [] for m in inst.machines}
        sim = simulate_schedule(inst, sched, seed=3)
        failures = [e for e in sim.events if e.event_type == "failure"]
        assert failures
        for e in failures:
            assert e.cost == inst.machines[e.machine_id].cm_cost
            assert e.chain_loss == 0.0
        assert sim.total_production_loss == 0.0
        assert sim.total_cm_cost == pytest.approx(sum(e.cost for e in failures))

    def test_chain_schedule_mean_matches_objective(self):
        inst = generate_small(seed=0)
        sched = fixed_interval_schedule(inst, "half_max").machine_schedules
        expected = deterministic_cost(inst, sched).total
        costs = [simulate_schedule(inst, sched, seed=s).total_cost
                 for s in range(800)]
        mean, se = _mean_se(costs)
        assert abs(mean - expected) <= Z * se


class TestCommonRandomNumbers:

    @staticmethod
    def _failure_times(sim, exclude):
        return sorted((e.machine_id, round(e.time, 9)) for e in sim.events
                      if e.event_type == "failure" and e.machine_id != exclude)

    def test_same_seed_reproduces_the_run(self):
        inst = generate_small(seed=0)
        sched = fixed_interval_schedule(inst, "half_max").machine_schedules
        a = simulate_schedule(inst, sched, seed=11)
        b = simulate_schedule(inst, sched, seed=11)
        assert self._failure_times(a, None) == self._failure_times(b, None)
        assert a.total_cost == b.total_cost

    def test_changing_one_machine_leaves_the_others_draws_untouched(self):
        """The defect: with one global stream, changing machine 0's schedule
        desynchronised every later machine (0 of 300 runs identical)."""
        inst = generate_small(seed=0)
        sched = fixed_interval_schedule(inst, "half_max").machine_schedules
        altered = {k: list(v) for k, v in sched.items()}
        altered[0] = [3]
        for seed in range(30):
            a = simulate_schedule(inst, sched, seed=seed)
            b = simulate_schedule(inst, altered, seed=seed)
            assert self._failure_times(a, exclude=0) == self._failure_times(b, exclude=0)
