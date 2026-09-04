"""
MaintAlign - Baseline Strategies (v2: Chain-Aware)
====================================================
Four naive scheduling strategies for cost comparison:

  1. max_interval  — maintain as late as allowed (minimize PM events)
  2. half_max      — maintain at half the max interval (conservative)
  3. analytical    — use Weibull closed-form optimal interval
  4. condition_based — schedule PM when expected failure cost exceeds PM cost
                       (simple condition-based maintenance heuristic)

When technicians conflict, tasks are shifted with greedy first-fit.
Every baseline is priced by ``core.costing.deterministic_cost`` — the same
function that prices the optimiser's result — so the two are comparable.
"""

import logging

from core.costing import apply_task_attribution, deterministic_cost
from core.instance import ProblemInstance
from core.solver import MaintenanceTask, SolverResult

logger = logging.getLogger(__name__)

ALL_STRATEGIES = ["max_interval", "half_max", "analytical", "condition_based"]


def _compute_cbm_interval(machine, instance, m_idx) -> int:
    """
    Condition-based maintenance interval:
    Find the age t where E[failure_cost(t)] first exceeds the per-task PM cost.
    This is a simple threshold heuristic for CBM.
    """
    chain = instance.get_chain_for_machine(m_idx)
    if chain:
        task_cost = (machine.pm_cost
                     + chain.chain_value * machine.maintenance_duration
                     + chain.retooling_cost)
    else:
        task_cost = (machine.pm_cost
                     + machine.production_value * machine.maintenance_duration)

    # Find the crossover point
    for t in range(1, instance.horizon + 1):
        if machine.expected_failure_cost(t) >= task_cost:
            return max(machine.maintenance_duration + machine.min_gap, t)

    # If failure cost never exceeds task cost, use max_interval
    return machine.max_interval


def fixed_interval_schedule(
    instance: ProblemInstance,
    strategy: str = "max_interval",
) -> SolverResult:
    """
    Generate a fixed-interval maintenance schedule.

    Args:
        instance: problem instance
        strategy: "max_interval" | "half_max" | "analytical" | "condition_based"
    """
    if strategy not in ALL_STRATEGIES:
        logger.warning("Unknown strategy '%s', falling back to max_interval",
                       strategy)
        strategy = "max_interval"

    H = instance.horizon
    K = instance.num_technicians

    # Step 1: compute desired PM times per machine
    desired: dict[int, list[int]] = {}
    for m_idx, machine in enumerate(instance.machines):
        if strategy == "max_interval":
            interval = machine.max_interval
        elif strategy == "half_max":
            interval = max(1, machine.max_interval // 2)
        elif strategy == "analytical":
            t_star = machine.optimal_interval_analytical()
            interval = max(1, min(int(t_star), machine.max_interval))
        elif strategy == "condition_based":
            interval = _compute_cbm_interval(machine, instance, m_idx)
            interval = min(interval, machine.max_interval)
        else:
            interval = machine.max_interval

        times = []
        t = interval
        while t + machine.maintenance_duration <= H:
            times.append(t)
            t += interval + machine.maintenance_duration
        if not times and H > machine.max_interval:
            times.append(min(machine.max_interval,
                             H - machine.maintenance_duration))
        desired[m_idx] = times

    # Step 2: greedy first-fit for technician conflicts
    tech_usage = [0] * H
    actual: dict[int, list[int]] = {m: [] for m in range(instance.num_machines)}
    tasks = []

    # Priority: chain machines first (higher cost of failure), then by CM cost
    def priority(mid):
        chain = instance.get_chain_for_machine(mid)
        chain_val = chain.chain_value if chain else 0
        return -(chain_val + instance.machines[mid].cm_cost)

    machine_order = sorted(range(instance.num_machines), key=priority)

    for m_idx in machine_order:
        machine = instance.machines[m_idx]
        chain = instance.get_chain_for_machine(m_idx)

        for ds in desired[m_idx]:
            for offset in range(machine.max_interval):
                cand = ds + offset
                if cand + machine.maintenance_duration > H:
                    break
                can = all(
                    tech_usage[t] < K
                    for t in range(cand, cand + machine.maintenance_duration)
                )
                if can:
                    for t in range(cand, cand + machine.maintenance_duration):
                        tech_usage[t] += 1

                    d = machine.maintenance_duration
                    pm_c = machine.pm_cost
                    if chain:
                        prod_c = ret_c = 0.0   # attributed once the schedule is known
                    else:
                        prod_c = machine.production_value * d
                        ret_c = 0

                    actual[m_idx].append(cand)
                    tasks.append(MaintenanceTask(
                        machine_id=m_idx,
                        task_index=len(actual[m_idx]) - 1,
                        start_time=cand, end_time=cand + d,
                        cost_pm=pm_c, cost_prod_loss=prod_c,
                        cost_retooling=ret_c,
                        chain_id=chain.id if chain else None,
                    ))
                    break

    # Step 3: price the schedule with the SAME function the solver's
    # result is priced with. Previously this block charged every chain
    # event at full price while the solver's objective granted overlapping
    # chain work a discount, so "savings vs best baseline" divided a
    # discounted number by an undiscounted one.
    breakdown = deterministic_cost(instance, actual)

    # Restate per-task chain costs on the same basis as the totals: a
    # shared outage is split among the tasks that share it.
    apply_task_attribution(instance, tasks, actual)

    logger.info("Baseline %s: $%.2f (%d tasks)", strategy,
                breakdown.total, len(tasks))

    return SolverResult(
        status=f"BASELINE-{strategy}",
        objective_value=breakdown.total,
        solve_time_seconds=0.0,
        tasks=tasks,
        total_pm_cost=breakdown.pm_cost,
        total_production_loss=breakdown.production_loss,
        total_retooling_cost=breakdown.retooling_cost,
        total_failure_cost=breakdown.failure_cost,
        machine_schedules=actual,
        chain_costs=breakdown.chain_costs,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(name)s | %(message)s")
    from core.solver import solve
    from utils.generator import generate_medium_easy, generate_small

    for gen, label in [(generate_small, "SMALL"), (generate_medium_easy, "MED")]:
        inst = gen()
        print(f"\n{'━'*60}")
        print(f" {label}: {inst.num_machines}M / {inst.num_technicians}K / "
              f"{inst.horizon}T / {len(inst.chains)} chains")
        print(f"{'━'*60}")

        for strat in ALL_STRATEGIES:
            b = fixed_interval_schedule(inst, strat)
            print(f"  {strat:<14} ${b.objective_value:>10,.2f}  "
                  f"(PM={b.total_pm_cost:,.0f} Prod={b.total_production_loss:,.0f} "
                  f"Fail={b.total_failure_cost:,.0f})")

        opt = solve(inst, time_limit_seconds=30)
        print(f"  {'OPTIMIZED':<14} ${opt.objective_value:>10,.2f}  "
              f"(PM={opt.total_pm_cost:,.0f} Prod={opt.total_production_loss:,.0f} "
              f"Fail={opt.total_failure_cost:,.0f})")

        best_b = min(
            fixed_interval_schedule(inst, s).objective_value
            for s in ALL_STRATEGIES
        )
        if best_b > 0 and opt.objective_value < float('inf'):
            pct = (1 - opt.objective_value / best_b) * 100
            print(f"  Savings vs best baseline: {pct:+.1f}%")
