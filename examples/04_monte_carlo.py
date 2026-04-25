"""
Example 04 — Monte Carlo Risk Analysis
========================================
A schedule that looks cheapest on paper isn't always cheapest in practice,
because real machine failures are random. Monte Carlo simulation replays
each schedule against thousands of random Weibull-drawn breakdowns and
reports expected cost, volatility, and tail risk.

This example runs 500 simulations for each strategy on a small instance
and prints a comparison table.

Run from the project root:
    python examples/04_monte_carlo.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.evaluator import compare_schedules
from core.baseline import ALL_STRATEGIES, fixed_interval_schedule
from core.solver import solve
from utils.generator import generate_small


def main() -> None:
    # Build an instance
    instance = generate_small(seed=0)
    print(instance.summary())

    # Generate schedules for every strategy
    schedules = {}
    best_baseline_name = None
    best_cost = float("inf")

    for strategy in ALL_STRATEGIES:
        b = fixed_interval_schedule(instance, strategy)
        schedules[strategy] = b.machine_schedules
        if b.objective_value < best_cost:
            best_cost = b.objective_value
            best_baseline_name = strategy

    # Add the optimized schedule too (warm-started from best baseline)
    opt = solve(
        instance,
        time_limit_seconds=30,
        hint_schedule=schedules[best_baseline_name],
    )
    schedules["optimized"] = opt.machine_schedules

    # Run Monte Carlo — uses the same seeds for every strategy so the
    # comparison is apples-to-apples
    print("\nRunning Monte Carlo simulation (500 sims per strategy)...")
    results = compare_schedules(instance, schedules, n_sims=500, base_seed=42)

    # Print comparison table
    print(f"\n{'Strategy':<18}{'Mean':>12}{'Std':>10}{'VaR95':>12}"
          f"{'Failures':>10}")
    print("─" * 62)
    for name, er in results.items():
        print(f"{name:<18}"
              f"${er.mean_cost:>10,.0f} "
              f"${er.std_cost:>8,.0f} "
              f"${er.var95:>10,.0f} "
              f"{er.mean_failures:>9.1f}")

    print("\nVaR95 = average cost of the worst 5% of scenarios (tail risk).")
    print("Lower mean AND lower VaR95 is better.")


if __name__ == "__main__":
    main()
