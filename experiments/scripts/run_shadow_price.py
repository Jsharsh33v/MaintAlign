"""
Experiment: the shadow price of maintenance capacity
=====================================================

RESEARCH QUESTION
-----------------
What is one more maintenance technician worth to the firm, and how many
should it employ?

METHOD
------
Hold the factory fixed -- same machines, same chains, same horizon, same
reliability -- and vary only the size of the maintenance workforce K. Solve
each to (near) optimality and price every schedule with the one cost
function in ``core.costing``.

The marginal value of the K-th technician is the reduction in total expected
cost it buys:

    MV(K) = C*(K - 1) - C*(K)

C*(K) is non-increasing in K (a larger workforce can always replicate a
smaller one's schedule), so MV(K) >= 0, and it is decreasing: technicians
run into diminishing returns as the binding constraint stops being labour.

MV(K) plotted against K is the firm's DERIVED DEMAND FOR MAINTENANCE
LABOUR. Intersect it with what a technician actually costs -- from
``utils.calibration``, built on the BLS wage -- and the crossing point is
the cost-minimising workforce:

    K* = max { K : MV(K) >= w_loaded * h * horizon }

Hire while the marginal product exceeds the marginal factor cost. That is
the standard condition, recovered from an operational scheduling model.

WHY THIS COMPARISON IS SAFE
---------------------------
Every point on the curve is an optimiser solution priced by the same
function, so it does not depend on the baselines being tuned fairly. It is
immune to the accounting mismatch that inflated the savings-vs-baseline
figure, because nothing here is divided by a baseline.

CAVEAT
------
MV is a difference of two solved objectives. When the solver hits its time
limit the difference includes solver noise, and MV can even come out
negative -- which is a proof-quality warning, not an economic finding. The
optimality gap is recorded for every point and non-monotonic costs are
flagged. Raise ``--time-limit`` until the flags disappear before quoting a
number.

USAGE
-----
    python experiments/scripts/run_shadow_price.py
    python experiments/scripts/run_shadow_price.py --machines 10 --horizon 30 \
        --k-max 8 --time-limit 120
"""

import argparse
import csv
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from core.baseline import ALL_STRATEGIES, fixed_interval_schedule  # noqa: E402
from core.costing import deterministic_cost  # noqa: E402
from core.solver import solve  # noqa: E402
from utils.calibration import (  # noqa: E402
    HOURS_PER_PERIOD,
    LOADED_LABOR_MULTIPLIER,
    TECHNICIAN_WAGE,
    labor_cost_per_technician_period,
)
from utils.generator import generate_instance  # noqa: E402

logger = logging.getLogger("shadow_price")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machines", type=int, default=8)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-min", type=int, default=1)
    p.add_argument("--k-max", type=int, default=6)
    p.add_argument("--time-limit", type=int, default=60,
                   help="CP-SAT seconds per K. Raise until gaps close.")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--tag", default="shadow_price")
    return p.parse_args()


def run_one_k(args, k: int) -> dict:
    """Solve the same factory with K technicians and price the result."""
    inst = generate_instance(
        f"shadow_M{args.machines}_K{k}", args.machines, k, args.horizon,
        num_chains=args.chains, seed=args.seed,
    )

    baselines = {s: fixed_interval_schedule(inst, s) for s in ALL_STRATEGIES}
    best_name, best = min(baselines.items(), key=lambda kv: kv[1].objective_value)

    res = solve(inst, time_limit_seconds=args.time_limit,
                hint_schedule=best.machine_schedules)

    # An INFEASIBLE K is not an expensive factory, it is a factory that
    # cannot run at all: with this workforce there is no schedule meeting the
    # max-interval policy. Pricing the empty schedule there would produce a
    # meaningless marginal value for the first feasible technician, so these
    # points are excluded from MV and reported separately as the minimum
    # viable workforce.
    feasible = res.status in ("OPTIMAL", "FEASIBLE")
    cost = deterministic_cost(inst, res.machine_schedules)

    return {
        "num_technicians": k,
        "rc": round(inst.resource_constrainedness, 4),
        "feasible": int(feasible),
        "total_cost": round(cost.total, 2) if feasible else "",
        "pm_cost": round(cost.pm_cost, 2),
        "prod_loss": round(cost.production_loss, 2),
        "retooling_cost": round(cost.retooling_cost, 2),
        "failure_cost": round(cost.failure_cost, 2),
        "num_tasks": cost.num_tasks,
        "num_grouped_tasks": cost.num_grouped_tasks,
        "status": res.status,
        "best_bound": round(res.best_bound, 2),
        "gap_pct": round(res.gap_pct, 3),
        "solve_time_sec": round(res.solve_time_seconds, 3),
        "best_baseline": best_name,
        "best_baseline_cost": round(best.objective_value, 2),
    }


def add_marginal_values(rows: list[dict], factor_cost: float) -> list[dict]:
    """MV(K) = C*(K-1) - C*(K), plus the hire/don't-hire verdict."""
    for i, row in enumerate(rows):
        prev = rows[i - 1] if i else None
        if (prev is None or not row["feasible"] or not prev["feasible"]):
            row["marginal_value"] = ""
            row["worth_hiring"] = ""
            row["monotonicity_ok"] = ""
            continue
        mv = prev["total_cost"] - row["total_cost"]
        row["marginal_value"] = round(mv, 2)
        row["worth_hiring"] = int(mv >= factor_cost)
        row["monotonicity_ok"] = int(mv >= 0)
    return rows


def optimal_workforce(rows: list[dict], factor_cost: float) -> int:
    """Largest K whose marginal value still covers what it costs."""
    feasible = [r for r in rows if r["feasible"]]
    if not feasible:
        return 0
    best = feasible[0]["num_technicians"]          # minimum viable workforce
    for row in feasible[1:]:
        if row["marginal_value"] != "" and row["marginal_value"] >= factor_cost:
            best = row["num_technicians"]
        else:
            break
    return best


def write_csv(rows: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot(rows, factor_cost, k_star, path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feas = [r for r in rows if r["feasible"]]
    ks = [r["num_technicians"] for r in feas]
    costs = [r["total_cost"] for r in feas]
    mv_rows = [r for r in feas if r["marginal_value"] != ""]
    mv_ks = [r["num_technicians"] for r in mv_rows]
    mvs = [r["marginal_value"] for r in mv_rows]
    if not mvs:
        mvs, mv_ks = [0.0], [ks[0] if ks else 0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.13})

    # ── total cost ────────────────────────────────────────────────
    ax1.plot(ks, costs, marker="o", color="#1c605a", lw=2, zorder=3,
             label="optimised total expected cost")
    ax1.plot(ks, [r["best_baseline_cost"] for r in feas], marker="s",
             ms=4, color="#a8342a", lw=1.4, ls="--", alpha=.8,
             label="best fixed-interval baseline")
    unproven = [(k, c) for k, c, r in zip(ks, costs, feas, strict=True)
                if r["status"] != "OPTIMAL"]
    if unproven:
        ax1.scatter([k for k, _ in unproven], [c for _, c in unproven],
                    s=110, facecolors="none", edgecolors="#9a6b12",
                    lw=1.8, zorder=4, label="time limit hit (not proved optimal)")
    infeasible_ks = [r["num_technicians"] for r in rows if not r["feasible"]]
    if infeasible_ks:
        ax1.annotate(
            f"K = {', '.join(str(k) for k in infeasible_ks)}: no feasible "
            f"schedule exists\n(cannot meet the max-interval policy at any cost)",
            xy=(0.015, 0.06), xycoords="axes fraction", fontsize=8.5,
            color="#a8342a", va="bottom")
    ax1.set_ylabel("Total expected cost ($)")
    ax1.set_title(
        f"The value of maintenance capacity\n"
        f"{args.machines} machines, {args.chains} production chains, "
        f"{args.horizon} periods, seed {args.seed}",
        fontsize=12, fontweight="bold", loc="left")
    ax1.grid(alpha=.25)
    ax1.legend(fontsize=8.5, framealpha=.95)

    # ── marginal value = derived demand for labour ────────────────
    colors = ["#1c605a" if m >= factor_cost else "#b8c4c0" for m in mvs]
    ax2.bar(mv_ks, mvs, color=colors, width=.62, zorder=3,
            label="marginal value of the K-th technician")
    # Label every bar: a marginal value of $0 is the finding here, and an
    # unlabelled zero-height bar reads as a broken chart rather than a result.
    span = max(max(mvs), factor_cost)
    for k, m in zip(mv_ks, mvs, strict=True):
        ax2.annotate(f"${m:,.0f}", xy=(k, m), xytext=(0, 5),
                     textcoords="offset points", ha="center",
                     fontsize=8.5, color="#41504b")
    ax2.set_ylim(0, span * 1.12)
    ax2.axhline(factor_cost, color="#a8342a", lw=1.8, ls="--", zorder=4,
                label=f"cost of a technician over the horizon "
                      f"(${factor_cost:,.0f})")
    ax2.axvline(k_star + .5, color="#9a6b12", lw=1.4, ls=":", zorder=2)
    ax2.annotate(f"K* = {k_star}", xy=(k_star + .5, max(mvs) * .82),
                 xytext=(6, 0), textcoords="offset points",
                 fontsize=10, fontweight="bold", color="#9a6b12")
    ax2.set_xlabel("Maintenance technicians (K)")
    ax2.set_ylabel("Marginal value of the K-th technician ($)")
    ax2.set_xticks(ks)
    ax2.grid(alpha=.25, axis="y")
    ax2.legend(fontsize=8.5, framealpha=.95)

    fig.text(0.012, 0.012,
             f"Wage {TECHNICIAN_WAGE.value:.2f} $/h (BLS, May 2025) x "
             f"{LOADED_LABOR_MULTIPLIER.value:g} loaded x "
             f"{HOURS_PER_PERIOD.value:g} h/period x {args.horizon} periods. "
             f"All schedules priced by core.costing.deterministic_cost. "
             f"CP-SAT limit {args.time_limit}s.",
             fontsize=7, color="#6d7d77")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s | %(message)s")
    args = parse_args()

    factor_cost = labor_cost_per_technician_period() * args.horizon

    print("=" * 72)
    print(" Shadow price of maintenance capacity")
    print(f" {args.machines} machines, {args.chains} chains, "
          f"{args.horizon} periods, seed {args.seed}")
    print(f" One technician over the horizon costs ${factor_cost:,.2f}")
    print("=" * 72)
    print(f" {'K':>3} {'RC':>6} {'total cost':>13} {'MV(K)':>12} "
          f"{'gap%':>7} {'status':>9}")
    print(" " + "-" * 62)

    rows = []
    for k in range(args.k_min, args.k_max + 1):
        row = run_one_k(args, k)
        rows.append(row)
        add_marginal_values(rows, factor_cost)
        mv = row.get("marginal_value", "")
        mv_s = f"${mv:>11,.0f}" if mv != "" else " " * 12
        cost_s = (f"${row['total_cost']:>12,.0f}" if row["feasible"]
                  else f"{'no schedule':>13}")
        print(f" {k:>3} {row['rc']:>6.3f} {cost_s} {mv_s} "
              f"{row['gap_pct']:>7.2f} {row['status']:>9}")

    k_star = optimal_workforce(rows, factor_cost)
    bad = [r["num_technicians"] for r in rows[1:] if r["monotonicity_ok"] == 0]

    csv_path = os.path.join(RESULTS_DIR, f"{args.tag}.csv")
    write_csv(rows, csv_path)
    print(f"\n Results  -> {os.path.normpath(csv_path)}")

    if not args.no_figure:
        fig_path = os.path.join(FIGURES_DIR, f"{args.tag}.png")
        plot(rows, factor_cost, k_star, fig_path, args)
        print(f" Figure   -> {os.path.normpath(fig_path)}")

    infeasible = [r["num_technicians"] for r in rows if not r["feasible"]]
    feasible = [r["num_technicians"] for r in rows if r["feasible"]]

    print()
    if infeasible:
        print(f" Minimum viable workforce: K = {min(feasible)} "
              f"(no feasible schedule exists at K={infeasible}). Below this "
              f"the plant cannot meet its own maintenance policy at any price.")
    print(f" Cost-minimising workforce: K* = {k_star}")
    print(f" Hire while marginal value >= ${factor_cost:,.0f} per technician.")
    if bad:
        print(f"\n WARNING: cost increased when K rose at K={bad}. Total cost "
              f"must be non-increasing in K, so this is solver noise, not "
              f"economics. Raise --time-limit and re-run before quoting MV.")
    unproven = [r["num_technicians"] for r in rows
                if r["feasible"] and r["status"] != "OPTIMAL"]
    if unproven:
        print(f" NOTE: optimality not proved at K={unproven}; "
              f"marginal values at those points include solver slack.")


if __name__ == "__main__":
    main()
