"""
Experiment 1: Solver Scalability
=================================

How large an instance can CP-SAT close, and how far from optimal is it when
it cannot?

WHAT CHANGED AND WHY
--------------------
The previous version recorded solve time and status but never the bound, so
there was no way to tell "stopped 2% from optimal" from "stopped having
made no progress". Since two thirds of the runs hit the time limit, the
resulting plot of solve-time-against-size showed a flat line at 180 s that
looked like convergence and was actually censoring.

Every row now carries ``best_bound``, ``gap_pct`` and ``proved_optimal``
alongside the time limit, solver version and platform. The honest scalability
claim is about the fraction of instances proved optimal and the gap where
they are not -- not about wall time, which is pinned to the limit by
construction.

USAGE
-----
    python experiments/scripts/run_scalability.py
    python experiments/scripts/run_scalability.py --seeds 0 1 2 3 4 --time-limit 300
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
    "tiny":     (3,  1, 12, 0),
    "small":    (6,  2, 20, 1),
    "med_easy": (10, 4, 30, 2),
    "med_hard": (15, 4, 40, 3),
    "large":    (20, 5, 50, 4),
}

FIELDNAMES = [
    "label", "seed", "num_machines", "num_technicians", "horizon",
    "num_chains", "rc",
    "total_cost", "pm_cost", "failure_cost", "prod_loss", "retooling_cost",
    "num_tasks", "num_grouped_tasks", "total_cost_full_basis",
    "best_baseline", "best_baseline_cost",
    "status", "model_variant", "proved_optimal", "best_bound", "gap_pct",
    "solve_time_sec",
    "hit_time_limit", "time_limit_sec", "solver_version", "platform",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--time-limit", type=int, default=180)
    p.add_argument("--only", nargs="+", choices=list(CONFIGS),
                   default=list(CONFIGS))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--out", default=os.path.join(
        RESULTS_DIR, "scalability_results.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    prov = provenance(args.time_limit)
    cp = Checkpoint(args.out, FIELDNAMES, ["label", "seed"], fresh=args.fresh)

    todo = [(lbl, s) for lbl in args.only for s in args.seeds]
    print("=" * 76)
    print(f" Experiment 1: Solver scalability ({len(todo)} runs, "
          f"limit {args.time_limit}s)")
    print("=" * 76)
    print(f" {'instance':<14}{'M':>4}{'RC':>7}{'time':>9}{'gap%':>8}"
          f"{'status':>11}")
    print(" " + "-" * 60)

    for label, seed in todo:
        if cp.done(label=label, seed=seed):
            continue
        M, K, T, C = CONFIGS[label]
        inst = generate_instance(f"{label}_s{seed}", M, K, T,
                                 num_chains=C, seed=seed)

        baselines = {s: fixed_interval_schedule(inst, s) for s in ALL_STRATEGIES}
        best_name = min(baselines, key=lambda s: baselines[s].objective_value)
        best = baselines[best_name]

        res = solve(inst, time_limit_seconds=args.time_limit,
                    hint_schedule=best.machine_schedules)

        row = {
            "label": label, "seed": seed, "num_machines": M,
            "num_technicians": K, "horizon": T, "num_chains": C,
            "rc": round(inst.resource_constrainedness, 4),
            "best_baseline": best_name,
            "best_baseline_cost": round(best.objective_value, 2),
            # A solve that stops within 2% of the limit stopped because of the
            # limit, not because it finished. Marking that is the difference
            # between a scalability plot and a picture of the time limit.
            "hit_time_limit": int(
                res.solve_time_seconds >= args.time_limit * 0.98),
            **cost_columns(inst, res.machine_schedules),
            **solver_columns(res),
            **prov,
        }
        cp.append(row)
        print(f" {label + '_s' + str(seed):<14}{M:>4}{row['rc']:>7.3f}"
              f"{row['solve_time_sec']:>9.1f}{row['gap_pct']:>8.2f}"
              f"{res.status:>11}")

    print(f"\n {len(cp)} rows in {os.path.normpath(args.out)}")
    print(" Report the fraction proved optimal and the gap where it is not."
          "\n Solve time alone is pinned to the limit and says nothing.")


if __name__ == "__main__":
    main()
