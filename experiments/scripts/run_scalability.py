"""
Experiment 1: Solver Scalability
==================================
Measures CP-SAT solve time as problem size increases.

Varies: number of machines (3, 6, 10, 15, 20)
Holds constant: time limit (180s), solver workers (12)
Repeats: 3 seeds per size
Records: solve_time, objective_value, solver_status, num_tasks
"""

import csv
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.baseline import fixed_interval_schedule
from core.solver import solve
from utils.generator import generate_instance

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "scalability_results.csv")

# Problem sizes to test
CONFIGS = [
    # (label, num_machines, num_technicians, horizon, num_chains)
    ("tiny",    3,  1, 12, 0),
    ("small",   6,  2, 20, 1),
    ("med_easy", 10, 4, 30, 2),
    ("med_hard", 15, 4, 40, 3),
    ("large",   20, 5, 50, 4),
]

SEEDS = [0, 1, 2]
TIME_LIMIT = 180  # seconds per solve


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    total_runs = len(CONFIGS) * len(SEEDS)
    run_num = 0

    print(f"{'='*65}")
    print(f" Experiment 1: Solver Scalability ({total_runs} runs)")
    print(f"{'='*65}")

    for label, M, K, T, C in CONFIGS:
        for seed in SEEDS:
            run_num += 1
            name = f"{label}_s{seed}"
            inst = generate_instance(name, M, K, T, num_chains=C, seed=seed)

            # Get best baseline for warm-start hint
            best_b = None
            best_b_cost = float('inf')
            for strat in ["analytical", "half_max", "max_interval", "condition_based"]:
                b = fixed_interval_schedule(inst, strat)
                if b.objective_value < best_b_cost:
                    best_b_cost = b.objective_value
                    best_b = b

            # Solve with CP-SAT
            result = solve(
                inst,
                time_limit_seconds=TIME_LIMIT,
                hint_schedule=best_b.machine_schedules if best_b else None,
            )

            row = {
                "run_id": run_num,
                "label": label,
                "num_machines": M,
                "num_technicians": K,
                "horizon": T,
                "num_chains": C,
                "seed": seed,
                "rc": round(inst.resource_constrainedness, 3),
                "solve_time_sec": round(result.solve_time_seconds, 4),
                "objective_value": round(result.objective_value, 2),
                "status": result.status,
                "num_tasks": len(result.tasks),
                "best_baseline_cost": round(best_b_cost, 2),
            }
            rows.append(row)

            print(f"  [{run_num:>2}/{total_runs}] {name:<16} "
                  f"M={M:<3} K={K:<2} T={T:<3} "
                  f"time={result.solve_time_seconds:>6.2f}s  "
                  f"status={result.status:<8}  "
                  f"cost=${result.objective_value:>10,.0f}")

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
