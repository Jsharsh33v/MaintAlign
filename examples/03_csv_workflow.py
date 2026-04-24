"""
Example 03 — CSV Workflow
===========================
Load a factory instance from CSV files and solve it. This is the workflow
a real user would follow: enter their machines in a spreadsheet, export
to CSV, then run MaintAlign.

Sample data files live in examples/sample_data/.

Run from the project root:
    python examples/03_csv_workflow.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.csv_loader import load_instance
from core.solver import solve
from core.baseline import fixed_interval_schedule, ALL_STRATEGIES
from core.validators import validate_instance, MaintAlignError


HERE = os.path.dirname(os.path.abspath(__file__))
MACHINES_CSV = os.path.join(HERE, "sample_data", "example_machines.csv")
CHAINS_CSV = os.path.join(HERE, "sample_data", "example_chains.csv")


def main() -> None:
    # Step 1: Load from CSV. The loader handles #CONFIG lines, headers,
    # and comment lines. Invalid rows are skipped with warnings.
    try:
        instance = load_instance(MACHINES_CSV, CHAINS_CSV)
    except FileNotFoundError as e:
        print(f"ERROR: CSV file not found — {e}")
        print(f"Expected files at:\n  {MACHINES_CSV}\n  {CHAINS_CSV}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: could not parse CSV — {e}")
        sys.exit(1)

    # Step 2: Validate the loaded instance
    try:
        validate_instance(instance)
    except MaintAlignError as e:
        print(f"ERROR: invalid instance — {e}")
        sys.exit(1)

    print(instance.summary())

    # Step 3: Compare all four baselines
    print("\n── Baselines ──")
    best_baseline = None
    best_cost = float("inf")
    for strategy in ALL_STRATEGIES:
        b = fixed_interval_schedule(instance, strategy)
        print(f"  {strategy:<16} ${b.objective_value:>10,.2f}  "
              f"({len(b.tasks)} tasks)")
        if b.objective_value < best_cost:
            best_cost = b.objective_value
            best_baseline = b

    # Step 4: Solve with warm-start from best baseline
    print("\n── Optimized (CP-SAT) ──")
    result = solve(
        instance,
        time_limit_seconds=30,
        hint_schedule=best_baseline.machine_schedules,
    )
    print(f"  status:        {result.status}")
    print(f"  total cost:    ${result.objective_value:,.2f}")
    print(f"  solve time:    {result.solve_time_seconds:.2f}s")
    print(f"  savings:       "
          f"{(1 - result.objective_value / best_cost) * 100:.1f}% over "
          f"best baseline")


if __name__ == "__main__":
    main()
