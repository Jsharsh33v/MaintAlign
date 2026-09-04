"""Data-model tests: the analytical PM interval and the Kijima virtual age.

Acceptance check for the interval: ``optimal_interval_analytical`` must
match a numerical minimisation of the same cost rate to within 1%.
"""

import pytest

from core.instance import MachineSpec


def _machine(beta, eta, pm, cm, repair_factor=1.0):
    return MachineSpec(id=0, name="m", maintenance_duration=1, pm_cost=pm,
                       cm_cost=cm, production_value=10, weibull_beta=beta,
                       weibull_eta=eta, max_interval=100, min_gap=1,
                       repair_factor=repair_factor)


def _cost_rate(m, T):
    """Long-run cost per period of a PM every T periods with minimal repair
    in between — the cost rate the objective's failure term implies."""
    return (m.pm_cost + m.cm_cost * (T / m.weibull_eta) ** m.weibull_beta) / T


def _numerical_argmin(m):
    lo, hi = 1e-3 * m.weibull_eta, 20.0 * m.weibull_eta
    n = 20000
    grid = [lo + (hi - lo) * i / n for i in range(n + 1)]
    best = min(grid, key=lambda T: _cost_rate(m, T))
    # golden-section refinement around the grid minimiser
    a, b = max(lo, best - (hi - lo) / n), min(hi, best + (hi - lo) / n)
    phi = (5 ** 0.5 - 1) / 2
    for _ in range(60):
        c, d = b - phi * (b - a), a + phi * (b - a)
        if _cost_rate(m, c) < _cost_rate(m, d):
            b = d
        else:
            a = c
    return (a + b) / 2


class TestAnalyticalInterval:

    @pytest.mark.parametrize("beta", [1.2, 1.5, 2.0, 2.5, 3.0, 4.0])
    @pytest.mark.parametrize("eta", [5.0, 12.5, 40.0])
    @pytest.mark.parametrize("ratio", [3, 5, 10, 20])
    def test_matches_numerical_minimisation_within_one_percent(self, beta, eta, ratio):
        m = _machine(beta, eta, pm=200, cm=200 * ratio)
        t_star = m.optimal_interval_analytical()
        t_num = _numerical_argmin(m)
        assert abs(t_star - t_num) / t_num < 0.01, (t_star, t_num)

    def test_closed_form_is_barlow_hunter(self):
        m = _machine(2.0, 10.0, pm=200, cm=1000)
        # η (c_pm / ((β − 1) c_cm))^(1/β) = 10 · (200 / 1000)^(1/2)
        assert m.optimal_interval_analytical() == pytest.approx(10 * 0.2 ** 0.5)

    def test_old_formula_was_too_short(self):
        """The defect: η (c_pm / (β (c_cm − c_pm)))^(1/β) is not the minimiser."""
        m = _machine(2.0, 10.0, pm=200, cm=1000)
        old = 10.0 * (200 / (2.0 * (1000 - 200))) ** 0.5
        assert old < m.optimal_interval_analytical()
        assert _cost_rate(m, old) > _cost_rate(m, m.optimal_interval_analytical())

    @pytest.mark.parametrize("beta", [0.5, 1.0])
    def test_no_interior_optimum_without_wear_out(self, beta):
        assert _machine(beta, 10.0, 200, 1000).optimal_interval_analytical() == float("inf")


class TestVirtualAge:

    def test_perfect_repair_returns_to_zero(self):
        m = _machine(2.0, 10.0, 200, 1000, repair_factor=1.0)
        assert m.virtual_age_after_pm(17) == 0

    def test_imperfect_repair_leaves_a_residual_in_whole_periods(self):
        m = _machine(2.0, 10.0, 200, 1000, repair_factor=0.7)
        assert m.virtual_age_after_pm(10) == 3          # 30% of the age remains
        assert isinstance(m.virtual_age_after_pm(10), int)
        assert m.virtual_age_after_pm(0) == 0

    def test_imperfect_reduces_to_perfect_at_zero_virtual_age(self):
        m = _machine(2.2, 9.0, 200, 1000)
        for gap in range(0, 15):
            assert m.expected_failures_imperfect(gap, 0) == pytest.approx(
                m.expected_failures(gap))

    def test_a_gap_from_a_positive_virtual_age_costs_at_least_as_much(self):
        """Λ convex through the origin ⇒ Λ(a+g) − Λ(a) ≥ Λ(g)."""
        m = _machine(2.2, 9.0, 200, 1000)
        for a in (1, 3, 7):
            for gap in (1, 4, 10):
                assert (m.expected_failures_imperfect(gap, a)
                        >= m.expected_failures(gap) - 1e-12)
