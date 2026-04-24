"""
Example 01 — Quick Start
==========================
The simplest way to use MaintAlign. Generates a tiny 3-machine instance,
solves it, and prints the schedule.

Run from the project root:
    python examples/01_quick_start.py
"""

import sys
import os

# Make sure we can import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.generator import generate_tiny
from core.solver import solve


def main() -> None:
    # Step 1: Generate a tiny problem instance (3 machines, 12 periods)
    instance = generate_tiny(seed=42)
    print(instance.summary())

    # Step 2: Solve with CP-SAT (30-second time limit)
    result = solve(instance, time_limit_seconds=30)

    # Step 3: Report results
    print(f"\nSolver status: {result.status}")
    print(f"Total expected cost: ${result.objective_value:,.2f}")
    print(f"Solve time: {result.solve_time_seconds:.2f}s")
    print(f"Maintenance tasks scheduled: {len(result.tasks)}")

    print("\nPer-machine schedule (start periods):")
    for machine_id, starts in sorted(result.machine_schedules.items()):
        machine_name = instance.machines[machine_id].name
        print(f"  {machine_name:<16} {starts}")


if __name__ == "__main__":
    main()
