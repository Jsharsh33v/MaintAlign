"""
MaintAlign - Factory Decomposer
==================================
Split large factories into solvable subproblems.

Strategy:
  1. Each production chain → one subproblem (chain machines must stay together)
  2. Standalone machines → batched into groups of ~15
  3. Technicians allocated proportionally
  4. Solve each subproblem independently
  5. Merge and resolve technician conflicts
"""

import logging

from core.instance import MachineSpec, ProblemInstance, ProductionChain
from core.solver import MaintenanceTask, SolverResult, solve

logger = logging.getLogger(__name__)

MAX_SUBPROBLEM_SIZE = 15  # Max machines per subproblem


def _partition_machines(instance: ProblemInstance) -> list[list[int]]:
    """
    Partition machines into groups for decomposition.

    Rules:
      - Chain machines stay together in one group
      - Standalone machines are batched into groups of ~MAX_SUBPROBLEM_SIZE
    """
    groups = []

    # Each chain forms a group
    for chain in instance.chains:
        groups.append(chain.machine_ids[:])

    # Batch standalone machines
    standalone = instance.standalone_machines
    for i in range(0, len(standalone), MAX_SUBPROBLEM_SIZE):
        batch = standalone[i:i + MAX_SUBPROBLEM_SIZE]
        groups.append(batch)

    return groups


def _build_subproblem(
    instance: ProblemInstance,
    machine_ids: list[int],
    sub_id: int,
) -> ProblemInstance:
    """
    Create a sub-instance from a subset of machines.

    Technicians are allocated proportionally to the number of machines.
    """
    M = instance.num_machines
    K_total = instance.num_technicians
    frac = len(machine_ids) / M

    # At least 1 technician per subproblem
    K_sub = max(1, round(K_total * frac))

    # Map old machine IDs to new sequential IDs
    old_to_new = {old: new for new, old in enumerate(machine_ids)}

    # Rebuild machines with new IDs
    sub_machines = []
    for new_id, old_id in enumerate(machine_ids):
        m = instance.machines[old_id]
        sub_machines.append(MachineSpec(
            id=new_id,
            name=m.name,
            maintenance_duration=m.maintenance_duration,
            pm_cost=m.pm_cost,
            cm_cost=m.cm_cost,
            production_value=m.production_value,
            weibull_beta=m.weibull_beta,
            weibull_eta=m.weibull_eta,
            max_interval=m.max_interval,
            min_gap=m.min_gap,
            repair_factor=m.repair_factor,
        ))

    # Rebuild chains (only include chains whose machines are all in this group)
    sub_chains = []
    for chain in instance.chains:
        if all(mid in old_to_new for mid in chain.machine_ids):
            sub_chains.append(ProductionChain(
                id=len(sub_chains),
                name=chain.name,
                machine_ids=[old_to_new[mid] for mid in chain.machine_ids],
                chain_value=chain.chain_value,
                retooling_cost=chain.retooling_cost,
            ))

    sub_instance = ProblemInstance(
        name=f"{instance.name}_sub{sub_id}",
        num_machines=len(sub_machines),
        num_technicians=K_sub,
        horizon=instance.horizon,
        machines=sub_machines,
        chains=sub_chains,
        blocked_periods=instance.blocked_periods[:],
    )

    return sub_instance


def _merge_schedules(
    sub_results: list[tuple[list[int], SolverResult]],
    instance: ProblemInstance,
) -> SolverResult:
    """
    Merge subproblem solutions into a single schedule.
    If there are technician conflicts, shift conflicting tasks.
    """
    K = instance.num_technicians

    # Merge all schedules, mapping back to original machine IDs
    merged_schedule: dict[int, list[int]] = {}
    merged_tasks: list[MaintenanceTask] = []
    total_pm_cost = 0.0
    total_failure_cost = 0.0
    total_prod_loss = 0.0
    total_retooling = 0.0

    for machine_ids, result in sub_results:
        for new_id, starts in result.machine_schedules.items():
            old_id = machine_ids[new_id]
            merged_schedule[old_id] = starts[:]
            # Rebuild tasks with original machine IDs
            for idx, s in enumerate(starts):
                machine = instance.machines[old_id]
                chain = instance.get_chain_for_machine(old_id)
                d = machine.maintenance_duration
                merged_tasks.append(MaintenanceTask(
                    machine_id=old_id,
                    task_index=idx,
                    start_time=s,
                    end_time=s + d,
                    cost_pm=float(machine.pm_cost),
                    cost_prod_loss=0.0,
                    cost_retooling=0.0,
                    chain_id=chain.id if chain else None,
                ))
        total_pm_cost += result.total_pm_cost
        total_failure_cost += result.total_failure_cost
        total_prod_loss += result.total_production_loss
        total_retooling += result.total_retooling_cost

    # Check technician capacity at each time step
    for _ in range(10):  # Max 10 fix iterations
        violations = _find_tech_violations(
            merged_schedule, instance, K
        )
        if not violations:
            break
        # Fix: shift the cheapest task in the violated time slot
        for t, tasks_at_t in violations:
            _shift_cheapest_task(merged_schedule, instance, t, tasks_at_t)

    # Build final result
    total_obj = total_pm_cost + total_failure_cost + total_prod_loss + total_retooling
    return SolverResult(
        status="DECOMPOSED",
        objective_value=total_obj,
        solve_time_seconds=sum(r.solve_time_seconds for _, r in sub_results),
        tasks=merged_tasks,
        total_pm_cost=total_pm_cost,
        total_production_loss=total_prod_loss,
        total_retooling_cost=total_retooling,
        total_failure_cost=total_failure_cost,
        machine_schedules=merged_schedule,
    )


def _find_tech_violations(
    schedule: dict[int, list[int]],
    instance: ProblemInstance,
    K: int,
) -> list[tuple[int, list[tuple[int, int]]]]:
    """Find time slots where technician count exceeds K."""
    H = instance.horizon
    violations = []

    for t in range(H):
        tasks_at_t = []
        for m_idx, starts in schedule.items():
            d = instance.machines[m_idx].maintenance_duration
            for s in starts:
                if s <= t < s + d:
                    tasks_at_t.append((m_idx, s))
        if len(tasks_at_t) > K:
            violations.append((t, tasks_at_t))

    return violations


def _shift_cheapest_task(
    schedule: dict[int, list[int]],
    instance: ProblemInstance,
    conflict_time: int,
    tasks_at_t: list[tuple[int, int]],
):
    """Shift the cheapest task by 1 period to resolve a conflict."""
    # Find cheapest task to shift
    cheapest = None
    cheapest_cost = float('inf')
    for m_idx, start in tasks_at_t:
        cost = instance.task_cost(m_idx)['total']
        if cost < cheapest_cost:
            cheapest_cost = cost
            cheapest = (m_idx, start)

    if cheapest:
        m_idx, old_start = cheapest
        d = instance.machines[m_idx].maintenance_duration
        new_start = old_start + 1
        if new_start + d <= instance.horizon:
            starts = schedule[m_idx]
            if old_start in starts:
                starts.remove(old_start)
                starts.append(new_start)
                starts.sort()
                logger.debug("Shifted M%d PM from t=%d to t=%d (conflict fix)",
                             m_idx, old_start, new_start)


def solve_decomposed(
    instance: ProblemInstance,
    time_limit_seconds: int = 60,
    num_workers: int = 8,
    log_search: bool = False,
) -> SolverResult:
    """
    Solve a large instance using decomposition.

    1. Partition machines into groups
    2. Solve each subproblem independently
    3. Merge and fix conflicts
    """
    groups = _partition_machines(instance)

    logger.info("Decomposed %d machines into %d subproblems: %s",
                instance.num_machines, len(groups),
                [len(g) for g in groups])

    # Allocate time proportionally
    time_per_sub = max(10, time_limit_seconds // len(groups))

    sub_results = []
    for i, machine_ids in enumerate(groups):
        sub_instance = _build_subproblem(instance, machine_ids, i)
        logger.info("Solving subproblem %d: %d machines, %d techs, %d chains",
                     i, sub_instance.num_machines,
                     sub_instance.num_technicians, len(sub_instance.chains))

        result = solve(
            sub_instance,
            time_limit_seconds=time_per_sub,
            num_workers=num_workers,
            log_search=log_search,
        )

        logger.info("  Sub %d: %s in %.1fs",
                     i, result.status, result.solve_time_seconds)
        sub_results.append((machine_ids, result))

    # Merge
    merged = _merge_schedules(sub_results, instance)
    logger.info("Merged solution: %d tasks, total cost: %.0f",
                len(merged.tasks), merged.objective_value)

    return merged
