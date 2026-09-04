"""
Shared plumbing for the experiment scripts.
============================================

Three things every experiment needs and none of them had:

CHECKPOINTING
    A full suite is tens of CP-SAT solves at a 180 s limit -- over an hour.
    Previously every script accumulated rows in memory and wrote the CSV at
    the very end, so a laptop sleeping or a Ctrl-C threw the whole run away.
    Rows are now appended as they complete and finished work is skipped on
    restart, so a run can be resumed.

PROVENANCE
    A solve time means nothing without the time limit it ran under, and an
    objective means nothing without the bound. Every row carries the limit,
    the bound, the gap, the solver version and the machine it ran on, so a
    number in a figure can be traced back to the conditions that produced it.

ONE COST BASIS
    Costs come from core.costing so the optimiser and the baselines are
    priced identically. Each row also carries the strict full-chain-cost
    figure, so a reader can see how much of the measured advantage depends
    on the chain-grouping model rather than on scheduling.
"""

import csv
import os
import platform
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from core.costing import deterministic_cost  # noqa: E402

RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results"))
FIGURES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "figures"))


def solver_version() -> str:
    try:
        from ortools import __version__ as v
        return f"ortools-{v}"
    except Exception:
        return "ortools-unknown"


def platform_string() -> str:
    return (f"{platform.system()}-{platform.machine()}-"
            f"py{sys.version_info.major}.{sys.version_info.minor}")


def provenance(time_limit: int) -> dict:
    """Columns every row carries so a figure can be traced to its conditions."""
    return {
        "time_limit_sec": time_limit,
        "solver_version": solver_version(),
        "platform": platform_string(),
    }


def cost_columns(instance, schedule, prefix: str = "") -> dict:
    """Price a schedule on both bases with the single costing function."""
    grouped = deterministic_cost(instance, schedule, chain_grouping=True)
    full = deterministic_cost(instance, schedule, chain_grouping=False)
    return {
        f"{prefix}total_cost": round(grouped.total, 2),
        f"{prefix}pm_cost": round(grouped.pm_cost, 2),
        f"{prefix}failure_cost": round(grouped.failure_cost, 2),
        f"{prefix}prod_loss": round(grouped.production_loss, 2),
        f"{prefix}retooling_cost": round(grouped.retooling_cost, 2),
        f"{prefix}num_tasks": grouped.num_tasks,
        f"{prefix}num_grouped_tasks": grouped.num_grouped_tasks,
        f"{prefix}total_cost_full_basis": round(full.total, 2),
    }


def solver_columns(result) -> dict:
    """Proof quality. A FEASIBLE incumbent is not an optimum, and OPTIMAL
    is only a proof when ``model_variant`` is ``full``."""
    return {
        "status": result.status,
        "model_variant": result.model_variant,
        "proved_optimal": int(result.status == "OPTIMAL"
                              and result.model_variant == "full"),
        "best_bound": round(result.best_bound, 2),
        "gap_pct": round(result.gap_pct, 3),
        "solve_time_sec": round(result.solve_time_seconds, 3),
    }


class Checkpoint:
    """Append-as-you-go CSV with resume.

    ``key_fields`` identify a unit of work. On construction the existing file
    is read and its keys remembered, so ``done()`` tells a runner to skip
    work already on disk.
    """

    def __init__(self, path: str, fieldnames: list[str],
                 key_fields: list[str], fresh: bool = False):
        self.path = path
        self.fieldnames = fieldnames
        self.key_fields = key_fields
        self._keys: set[tuple] = set()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if fresh and os.path.exists(path):
            os.replace(path, path + ".bak")
            print(f" previous results moved to {os.path.basename(path)}.bak")

        if os.path.exists(path):
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames == fieldnames:
                    for row in reader:
                        self._keys.add(self._key(row))
                else:
                    # Schema changed: the old file cannot be extended.
                    os.replace(path, path + ".bak")
                    print(f" schema changed; old results moved to "
                          f"{os.path.basename(path)}.bak")

        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        if self._keys:
            print(f" resuming: {len(self._keys)} rows already complete")

    def _key(self, row: dict) -> tuple:
        return tuple(str(row[k]) for k in self.key_fields)

    def done(self, **kw) -> bool:
        return tuple(str(kw[k]) for k in self.key_fields) in self._keys

    def append(self, row: dict):
        missing = set(self.fieldnames) - set(row)
        if missing:
            raise KeyError(f"row is missing columns: {sorted(missing)}")
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(
                {k: row[k] for k in self.fieldnames})
        self._keys.add(self._key(row))

    def __len__(self) -> int:
        return len(self._keys)
