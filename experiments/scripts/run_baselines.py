"""
Experiment 2: Optimisation Quality vs Baselines
================================================

Compares the CP-SAT schedule against all four fixed-interval baselines on
the same instance, and reports the cost reduction.

WHAT CHANGED AND WHY
--------------------
The previous version of this script divided the CP-SAT objective (which
granted overlapping chain maintenance a 50% discount) by a baseline total
computed at full chain price. The savings column was therefore a ratio of
two different accountings, overstating the advantage by 5-16 percentage
points depending on the instance; on the ``optimized`` rows the four cost
columns did not even sum to ``total_cost``.

Every cost here now comes from ``core.costing.deterministic_cost``, so the
optimiser and the baselines are priced by the same function. Each row also
carries ``total_cost_full_basis`` -- the same schedule with no grouping
credit at all -- so ``savings_pct`` and ``savings_pct_full_basis`` bracket
how much of the result depends on the chain-grouping model rather than on
scheduling. Report both.

Rows are appended as they complete and finished work is skipped on restart,
so this can be interrupted and resumed.

USAGE
-----
    python experiments/scripts/run_baselines.py
    python experiments/scripts/run_baselines.py --seeds 0 1 2 3 4 --time-limit 300
    python experiments/scripts/run_baselines.py --only small med_easy --fresh
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from _runner import (  # noqa: E402
    RESULTS_DIR,
    Checkpoint,
    cost_columns,
    provenance,
    solver_columns,
)

from core.baseline import ALL_STRATEGIES, fixed_interval_schedule  # noqa: E402
from core.solver import solve  # noqa: E402
from utils.generator import generate_instance  # noqa: E402

CONFIGS = {
    "small":    (6,  2, 20, 1),
    "med_easy": (10, 4, 30, 2),
    "med_hard": (10, 2, 30, 2),
    "large":    (20, 5, 50, 4),
}

FIELDNAMES = [
    "label", "seed", "strategy",
    "num_machines", "num_technicians", "horizon", "num_chains",
    "total_cost", "pm_cost", "failure_cost", "prod_loss", "retooling_cost",
    "num_tasks", "num_grouped_tasks", "total_cost_full_basis",
    "savings_pct", "savings_pct_full_basis",
    "status", "model_variant", "proved_optimal", "best_bound", "gap_pct",
    "solve_time_sec",
    "time_limit_sec", "solver_version", "platform",
]

NA = {"status": "", "model_variant": "", "proved_optimal": "",
      "best_bound": "", "gap_pct": "", "solve_time_sec": ""}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="More seeds means tighter confidence intervals. "
                        "Three is not enough to put error bars on.")
    p.add_argument("--time-limit", type=int, default=180)
    p.add_argument("--only", nargs="+", choices=list(CONFIGS),
                   default=list(CONFIGS))
    p.add_argument("--fresh", action="store_true",
                   help="Ignore existing results (the old file is kept as .bak)")
    p.add_argument("--out", default=os.path.join(
        RESULTS_DIR, "baseline_comparison.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    prov = provenance(args.time_limit)
    cp = Checkpoint(args.out, FIELDNAMES, ["label", "seed", "strategy"],
                    fresh=args.fresh)

    todo = [(lbl, s) for lbl in args.only for s in args.seeds]
    print("=" * 72)
    print(f" Experiment 2: Optimisation quality vs baselines "
          f"({len(todo)} instances)")
    print(f" CP-SAT limit {args.time_limit}s | seeds {args.seeds}")
    print("=" * 72)

    for n, (label, seed) in enumerate(todo, 1):
        M, K, T, C = CONFIGS[label]
        if cp.done(label=label, seed=seed, strategy="optimized"):
            print(f"  [{n:>2}/{len(todo)}] {label}_s{seed:<3} already done")
            continue

        inst = generate_instance(f"{label}_s{seed}", M, K, T,
                                 num_chains=C, seed=seed)
        shape = {"label": label, "seed": seed, "num_machines": M,
                 "num_technicians": K, "horizon": T, "num_chains": C}

        baselines = {s: fixed_interval_schedule(inst, s) for s in ALL_STRATEGIES}
        best_name = min(baselines, key=lambda s: baselines[s].objective_value)
        best = baselines[best_name]
        best_grouped = best.objective_value
        best_full = cost_columns(
            inst, best.machine_schedules)["total_cost_full_basis"]

        opt = solve(inst, time_limit_seconds=args.time_limit,
                    hint_schedule=best.machine_schedules)

        for strategy, b in baselines.items():
            cp.append({**shape, **prov, **NA, "strategy": strategy,
                       **cost_columns(inst, b.machine_schedules),
                       "savings_pct": "", "savings_pct_full_basis": ""})

        opt_costs = cost_columns(inst, opt.machine_schedules)
        savings = (100 * (1 - opt_costs["total_cost"] / best_grouped)
                   if best_grouped > 0 else 0.0)
        savings_full = (
            100 * (1 - opt_costs["total_cost_full_basis"] / best_full)
            if best_full > 0 else 0.0)

        cp.append({**shape, **prov, "strategy": "optimized", **opt_costs,
                   **solver_columns(opt),
                   "savings_pct": round(savings, 3),
                   "savings_pct_full_basis": round(savings_full, 3)})

        flag = "" if opt.status == "OPTIMAL" else f"  [{opt.status}, gap {opt.gap_pct:.1f}%]"
        print(f"  [{n:>2}/{len(todo)}] {label}_s{seed:<3} "
              f"opt=${opt_costs['total_cost']:>9,.0f}  "
              f"best({best_name})=${best_grouped:>9,.0f}  "
              f"save={savings:>+6.1f}%  (full basis {savings_full:>+6.1f}%)"
              f"{flag}")

    print(f"\n {len(cp)} rows in {os.path.normpath(args.out)}")


if __name__ == "__main__":
    main()
