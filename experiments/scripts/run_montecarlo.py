"""
Experiment 3: Monte Carlo Risk Analysis
==========================================
Tests how each schedule survives random machine failures.

Uses the Monte Carlo simulator (Weibull failure model) to compare
optimized vs baseline schedules on:
  - Mean realized cost (with random breakdowns)
  - Standard deviation (cost volatility)
  - VaR95 (tail risk: average of worst 5% scenarios)
  - Average number of failures
  - Average downtime

Sizes: small, medium_easy, medium_hard
Simulations: 500 per schedule per instance
Seeds: 2 per size
"""

import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from analysis.evaluator import compare_schedules
from core.baseline import ALL_STRATEGIES, fixed_interval_schedule
from core.solver import solve
from utils.generator import generate_instance

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "montecarlo_results.csv")

CONFIGS = [
    ("small",     6,  2, 20, 1),
    ("med_easy", 10,  4, 30, 2),
    ("med_hard", 10,  2, 30, 2),
]

SEEDS = [0, 1]
NUM_SIMS = 500
TIME_LIMIT = 180


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    total_runs = len(CONFIGS) * len(SEEDS)
    run_num = 0

    print(f"{'='*65}")
    print(f" Experiment 3: Monte Carlo Risk Analysis ({total_runs} instances)")
    print(f" {NUM_SIMS} simulations per schedule per instance")
    print(f"{'='*65}")

    for label, M, K, T, C in CONFIGS:
        for seed in SEEDS:
            run_num += 1
            name = f"{label}_s{seed}"
            inst = generate_instance(name, M, K, T, num_chains=C, seed=seed)

            # Get all schedules
            baselines = {}
            for strat in ALL_STRATEGIES:
                b = fixed_interval_schedule(inst, strat)
                baselines[strat] = b

            best_b_name = min(baselines, key=lambda k: baselines[k].objective_value)
            hint = baselines[best_b_name].machine_schedules

            opt = solve(inst, time_limit_seconds=TIME_LIMIT, hint_schedule=hint)

            # Build schedule dict for Monte Carlo
            schedules = {strat: b.machine_schedules for strat, b in baselines.items()}
            schedules["optimized"] = opt.machine_schedules

            print(f"\n  [{run_num}/{total_runs}] {name} — running {NUM_SIMS} sims...")

            # Run Monte Carlo comparison (same random seeds for fairness)
            mc_results = compare_schedules(inst, schedules, n_sims=NUM_SIMS)

            for strat_name, er in mc_results.items():
                rows.append({
                    "run_id": run_num,
                    "label": label,
                    "num_machines": M,
                    "num_technicians": K,
                    "horizon": T,
                    "num_chains": C,
                    "seed": seed,
                    "strategy": strat_name,
                    "mean_cost": round(er.mean_cost, 2),
                    "std_cost": round(er.std_cost, 2),
                    "median_cost": round(er.median_cost, 2),
                    "p5_cost": round(er.p5_cost, 2),
                    "p95_cost": round(er.p95_cost, 2),
                    "var95": round(er.var95, 2),
                    "mean_failures": round(er.mean_failures, 2),
                    "mean_downtime": round(er.mean_downtime, 2),
                    "mean_pm_cost": round(er.mean_pm_cost, 2),
                    "mean_cm_cost": round(er.mean_cm_cost, 2),
                    "mean_prod_loss": round(er.mean_prod_loss, 2),
                    "num_sims": NUM_SIMS,
                })

                print(f"    {strat_name:<16} mean=${er.mean_cost:>9,.0f} "
                      f"±${er.std_cost:>7,.0f}  "
                      f"VaR95=${er.var95:>9,.0f}  "
                      f"failures={er.mean_failures:.1f}")

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n Results saved to {OUTPUT_FILE}")
    print(f" {len(rows)} rows written")


if __name__ == "__main__":
    main()
