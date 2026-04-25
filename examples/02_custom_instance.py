"""
Example 02 — Custom Instance
==============================
Build a problem instance by hand instead of using a preset generator.
Useful when modeling a real factory with specific machines.

This example models a small 4-machine job shop with:
    - 2 standalone machines (a lathe and a drill press)
    - 1 production chain linking a welder and a press

Run from the project root:
    python examples/02_custom_instance.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.baseline import fixed_interval_schedule
from core.instance import MachineSpec, ProblemInstance, ProductionChain
from core.solver import solve
from core.validators import InvalidInstanceError, validate_instance


def build_instance() -> ProblemInstance:
    """Construct a 4-machine instance by hand."""

    machines = [
        # Standalone: CNC Lathe — expensive, predictable wear-out
        MachineSpec(
            id=0,
            name="CNC_Lathe",
            maintenance_duration=2,
            pm_cost=400,
            cm_cost=3200,        # 8x PM cost if it breaks
            production_value=300,
            weibull_beta=2.5,    # fairly predictable wear-out
            weibull_eta=14,      # typical life: 14 periods
            max_interval=14,
            min_gap=4,
        ),
        # Standalone: Drill Press — cheaper, shorter life
        MachineSpec(
            id=1,
            name="Drill_Press",
            maintenance_duration=1,
            pm_cost=150,
            cm_cost=900,
            production_value=120,
            weibull_beta=2.0,
            weibull_eta=10,
            max_interval=10,
            min_gap=3,
        ),
        # Chain member: Welder
        MachineSpec(
            id=2,
            name="Welder",
            maintenance_duration=2,
            pm_cost=350,
            cm_cost=2800,
            production_value=250,
            weibull_beta=2.2,
            weibull_eta=12,
            max_interval=12,
            min_gap=3,
        ),
        # Chain member: Hydraulic Press
        MachineSpec(
            id=3,
            name="Press_Hyd",
            maintenance_duration=2,
            pm_cost=450,
            cm_cost=3600,
            production_value=280,
            weibull_beta=2.3,
            weibull_eta=13,
            max_interval=13,
            min_gap=3,
        ),
    ]

    chains = [
        ProductionChain(
            id=0,
            name="Body_Line",
            machine_ids=[2, 3],        # Welder + Press together
            chain_value=700,            # synergy: worth more than 250 + 280 alone
            retooling_cost=250,
        ),
    ]

    instance = ProblemInstance(
        name="custom_shop",
        num_machines=len(machines),
        num_technicians=2,
        horizon=30,
        machines=machines,
        chains=chains,
    )

    return instance


def main() -> None:
    instance = build_instance()

    # Validate before solving — catches typos, inconsistencies, etc.
    try:
        validate_instance(instance)
    except InvalidInstanceError as e:
        print(f"ERROR: instance is invalid — {e}")
        sys.exit(1)

    print(instance.summary())

    # Compare optimized vs best baseline
    baseline = fixed_interval_schedule(instance, "analytical")
    optimized = solve(
        instance,
        time_limit_seconds=30,
        hint_schedule=baseline.machine_schedules,  # warm-start
    )

    savings = (1 - optimized.objective_value / baseline.objective_value) * 100
    print(f"\nBaseline (analytical):  ${baseline.objective_value:>10,.2f}")
    print(f"Optimized (CP-SAT):     ${optimized.objective_value:>10,.2f}")
    print(f"Savings:                 {savings:>10.1f}%")


if __name__ == "__main__":
    main()
