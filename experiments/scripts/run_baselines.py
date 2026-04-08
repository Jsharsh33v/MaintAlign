"""
Experiment 2: Optimization Quality vs Baselines
==================================================
Compares CP-SAT optimized cost against all 4 baseline strategies.

Varies: problem size (small, medium_easy, medium_hard, large)
Compares: CP-SAT vs max_interval, half_max, analytical, condition_based
Repeats: 3 seeds per size
Records: total cost, cost breakdown (PM, failure, prod loss, retooling), savings %
"""

import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.generator import generate_instance
from core.solver import solve
from core.baseline import fixed_interval_schedule, ALL_STRATEGIES

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "baseline_comparison.csv")

CONFIGS = [
    ("small",     6,  2, 20, 1),
    ("med_easy", 10,  4, 30, 2),
    ("med_hard", 10,  2, 30, 2),
    ("large",    20,  5, 50, 4),
]

SEEDS = [0, 1, 2]
TIME_LIMIT = 180


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    total_runs = len(CONFIGS) * len(SEEDS)
    run_num = 0

    print(f"{'='*65}")
    print(f" Experiment 2: Optimization Quality vs Baselines ({total_runs} runs)")
    print(f"{'='*65}")

    for label, M, K, T, C in CONFIGS:
        for seed in SEEDS:
            run_num += 1
            name = f"{label}_s{seed}"
            inst = generate_instance(name, M, K, T, num_chains=C, seed=seed)

            # Run all baselines
            baselines = {}
            for strat in ALL_STRATEGIES:
                baselines[strat] = fixed_interval_schedule(inst, strat)

            # Best baseline for hint
            best_b_name = min(baselines, key=lambda k: baselines[k].objective_value)
            hint = baselines[best_b_name].machine_schedules

            # Optimized
            opt = solve(inst, time_limit_seconds=TIME_LIMIT, hint_schedule=hint)

            # Record baseline rows
            for strat, b in baselines.items():
                rows.append({
                    "run_id": run_num,
                    "label": label,
                    "num_machines": M,
                    "num_technicians": K,
                    "horizon": T,
                    "num_chains": C,
                    "seed": seed,
                    "strategy": strat,
                    "total_cost": round(b.objective_value, 2),
                    "pm_cost": round(b.total_pm_cost, 2),
                    "failure_cost": round(b.total_failure_cost, 2),
                    "prod_loss": round(b.total_production_loss, 2),
                    "retooling_cost": round(b.total_retooling_cost, 2),
                    "num_tasks": len(b.tasks),
                })

            # Record optimized row
            best_b_cost = baselines[best_b_name].objective_value
            savings = (1 - opt.objective_value / best_b_cost) * 100 if best_b_cost > 0 else 0

            rows.append({
                "run_id": run_num,
                "label": label,
                "num_machines": M,
                "num_technicians": K,
                "horizon": T,
                "num_chains": C,
                "seed": seed,
                "strategy": "optimized",
                "total_cost": round(opt.objective_value, 2),
                "pm_cost": round(opt.total_pm_cost, 2),
                "failure_cost": round(opt.total_failure_cost, 2),
                "prod_loss": round(opt.total_production_loss, 2),
                "retooling_cost": round(opt.total_retooling_cost, 2),
                "num_tasks": len(opt.tasks),
            })

            print(f"  [{run_num:>2}/{total_runs}] {name:<16} "
                  f"Opt=${opt.objective_value:>9,.0f}  "
                  f"BestBase=${best_b_cost:>9,.0f}  "
                  f"Save={savings:>+6.1f}%  "
                  f"({opt.status})")

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