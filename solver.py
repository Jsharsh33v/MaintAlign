"""
MaintAlign - CP-SAT Solver (v3: Optimized Performance)
========================================================

PERFORMANCE OPTIMIZATIONS (v3):
  1. Tighter variable bounds: each task slot gets a restricted start range
     based on its position and the max_interval constraint.
  2. Smarter max_tasks: uses analytical optimal interval to estimate
     a realistic upper bound, not just physical packing.
  3. Pre-fix mandatory tasks: if failure cost at max_interval > PM cost,
     the machine MUST have at least the minimum tasks — fix them early.
  4. Implied constraints (redundant cuts): total tasks across all machines
     bounded by technician capacity; helps LP relaxation.
  5. Search strategy: prioritize presence decisions (branch on p vars first),
     then start times. This front-loads the hardest decisions.
  6. Solver hints from best baseline for warm start.
  7. Symmetry breaking for identical machines.

SOLVER APPROACH:
  CP-SAT with:
    - NewOptionalFixedSizeIntervalVar for each potential task
    - AddCumulative for technician capacity
    - AddNoOverlap per machine
    - AddElement for Weibull failure cost table lookups
    - Half-reification for conditional constraints
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from ortools.sat.python import cp_model

from instance import ProblemInstance, MachineSpec

logger = logging.getLogger(__name__)

COST_SCALE = 100  # multiply float costs by this for integer CP-SAT


@dataclass
class MaintenanceTask:
    """One scheduled maintenance event in the solution."""
    machine_id: int
    task_index: int
    start_time: int
    end_time: int
    cost_pm: float
    cost_prod_loss: float
    cost_retooling: float
    chain_id: Optional[int] = None


@dataclass
class SolverResult:
    """Complete solution with cost breakdown."""
    status: str
    objective_value: float
    solve_time_seconds: float
    tasks: List[MaintenanceTask]
    total_pm_cost: float
    total_production_loss: float
    total_retooling_cost: float
    total_failure_cost: float
    machine_schedules: Dict[int, List[int]]
    chain_costs: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"{'═'*55}",
            f" Solver Result: {self.status}",
            f"{'═'*55}",
            f" Total Cost:        ${self.objective_value:>12,.2f}",
            f"   PM Cost:         ${self.total_pm_cost:>12,.2f}",
            f"   Production Loss: ${self.total_production_loss:>12,.2f}",
            f"   Retooling Cost:  ${self.total_retooling_cost:>12,.2f}",
            f"   Failure Cost:    ${self.total_failure_cost:>12,.2f}",
            f" Solve Time: {self.solve_time_seconds:.3f}s",
            f" Tasks Scheduled: {len(self.tasks)}",
            f"{'─'*55}",
            f" Schedule:",
        ]
        for mid, starts in sorted(self.machine_schedules.items()):
            tag = f" (chain {self.tasks[0].chain_id})" if starts and any(
                t.machine_id == mid and t.chain_id is not None for t in self.tasks
            ) else ""
            lines.append(f"   M{mid}: PM at t={starts}{tag}")

        if self.chain_costs:
            lines.append(f"{'─'*55}")
            lines.append(f" Chain Cost Breakdown:")
            for cid, cc in sorted(self.chain_costs.items()):
                lines.append(
                    f"   Chain {cid}: prod_loss=${cc['prod_loss']:,.0f}  "
                    f"retool=${cc['retooling']:,.0f}  "
                    f"events={cc['num_events']}"
                )
        return "\n".join(lines)


def _precompute_failure_table(machine: MachineSpec, horizon: int) -> List[int]:
    """Table: gap_length → integer-scaled expected failure cost."""
    return [
        int(machine.expected_failure_cost(g) * COST_SCALE)
        for g in range(horizon + 1)
    ]


def _compute_min_tasks(machine: MachineSpec, horizon: int) -> int:
    """Minimum tasks needed to satisfy max_interval constraint."""
    W = machine.max_interval
    d = machine.maintenance_duration
    if horizon <= W:
        return 0
    n = 1
    while (n + 1) * W + n * d < horizon:
        n += 1
    return n


def _compute_max_tasks_tight(machine: MachineSpec, horizon: int,
                              instance: ProblemInstance, m_idx: int) -> int:
    """
    OPTIMIZED: Smarter upper bound on number of tasks.
    Uses the analytical optimal interval (or half_max) to estimate
    a realistic task count, then adds a small buffer.
    This is much tighter than the old horizon // (d + min_gap).
    """
    d = machine.maintenance_duration
    g = machine.min_gap

    # Physical maximum (old method — loose)
    phys_max = horizon // (d + g)

    # Estimate based on optimal/practical interval
    t_star = machine.optimal_interval_analytical()
    if t_star < float('inf') and t_star > 0:
        practical_interval = max(d + g, int(t_star))
    else:
        practical_interval = max(d + g, machine.max_interval // 2)

    # Estimated number of tasks + buffer of 2
    estimated = horizon // (practical_interval + d) + 2

    # Minimum tasks required
    n_min = _compute_min_tasks(machine, horizon)

    # Take the tighter of physical max and estimate, but at least n_min
    tight = max(n_min, min(phys_max, estimated))

    if tight < phys_max:
        logger.debug("M%d: max_tasks %d → %d (saved %d slots)",
                     m_idx, phys_max, tight, phys_max - tight)

    return tight


def _compute_start_bounds(machine: MachineSpec, horizon: int,
                           j: int, n_tasks: int) -> tuple:
    """
    OPTIMIZED: Compute tight [lb, ub] for start_var[m][j].
    Task j must leave room for j tasks before it and (possible)
    tasks after it, respecting min_gap and max_interval.
    """
    d = machine.maintenance_duration
    g = machine.min_gap
    W = machine.max_interval

    # Lower bound: task j must come after j previous tasks
    # Each previous task takes d+g, and first task can start at 0
    lb = j * (d + g)

    # Upper bound: task j must finish before horizon, and leave room
    # for any mandatory tasks after it
    ub = horizon - d

    # Tighter UB: task j can't start later than j*W + j*d
    # (because each gap before it is at most W)
    ub_from_W = min(ub, (j + 1) * W + j * d)
    ub = min(ub, ub_from_W)

    return max(0, lb), max(lb, ub)


def _add_solver_hints(model, instance, present, start_var, max_tasks,
                      hint_schedule: Optional[Dict[int, List[int]]] = None):
    """Add solver hints from a baseline solution to warm-start the search."""
    if hint_schedule is None:
        return

    for m_idx, machine in enumerate(instance.machines):
        J = max_tasks[m_idx]
        hint_starts = sorted(hint_schedule.get(m_idx, []))

        for j in range(J):
            if j < len(hint_starts):
                model.AddHint(present[m_idx][j], 1)
                model.AddHint(start_var[m_idx][j], hint_starts[j])
            else:
                model.AddHint(present[m_idx][j], 0)

    logger.debug("Added solver hints from baseline schedule")


def _add_symmetry_breaking(model, instance, present, start_var, max_tasks):
    """Symmetry breaking: order first tasks of identical machines."""
    groups = {}
    for m_idx, m in enumerate(instance.machines):
        key = (m.maintenance_duration, m.pm_cost, m.cm_cost, m.max_interval)
        groups.setdefault(key, []).append(m_idx)

    count = 0
    for key, mids in groups.items():
        if len(mids) < 2:
            continue
        for i in range(len(mids) - 1):
            a, b = mids[i], mids[i + 1]
            if max_tasks[a] > 0 and max_tasks[b] > 0:
                both = model.NewBoolVar(f"sym_{a}_{b}")
                model.AddBoolAnd([
                    present[a][0], present[b][0]
                ]).OnlyEnforceIf(both)
                model.AddBoolOr([
                    present[a][0].Not(), present[b][0].Not()
                ]).OnlyEnforceIf(both.Not())
                model.Add(
                    start_var[a][0] <= start_var[b][0]
                ).OnlyEnforceIf(both)
                count += 1

    if count > 0:
        logger.debug("Added %d symmetry breaking constraints", count)


def _add_implied_constraints(model, instance, present, max_tasks):
    """
    OPTIMIZED: Add redundant constraints that help the LP relaxation.
    These don't change the feasible set but help the solver prune faster.
    """
    H = instance.horizon
    K = instance.num_technicians

    # 1. Total tasks across all machines bounded by technician capacity
    # At most K tasks can run at any time, and each takes d periods.
    # So total task-periods <= K * H
    all_weighted = []
    for m_idx, machine in enumerate(instance.machines):
        d = machine.maintenance_duration
        for j in range(max_tasks[m_idx]):
            weighted = model.NewIntVar(0, d, f"tw_{m_idx}_{j}")
            model.Add(weighted == d).OnlyEnforceIf(present[m_idx][j])
            model.Add(weighted == 0).OnlyEnforceIf(present[m_idx][j].Not())
            all_weighted.append(weighted)

    if all_weighted:
        model.Add(sum(all_weighted) <= K * H)
        logger.debug("Added technician capacity implied cut")

    # 2. Per-machine: number of tasks <= ceil(H / (d + min_gap))
    # This is already handled by max_tasks, but making it explicit
    # as a linear constraint helps the LP relaxation
    for m_idx, machine in enumerate(instance.machines):
        J = max_tasks[m_idx]
        if J > 2:
            model.Add(
                sum(present[m_idx][j] for j in range(J)) <= J
            )


def _add_search_strategy(model, present, start_var, max_tasks, M):
    """
    OPTIMIZED: Guide solver search by prioritizing decisions.
    Branch on presence variables first (the key 0/1 decisions),
    then start times. This front-loads the hardest decisions.
    """
    # Collect all presence variables (these are the key decisions)
    p_vars = []
    s_vars = []
    for m_idx in range(M):
        for j in range(max_tasks[m_idx]):
            p_vars.append(present[m_idx][j])
            s_vars.append(start_var[m_idx][j])

    if p_vars:
        # First decide WHICH tasks to schedule (presence)
        model.AddDecisionStrategy(
            p_vars,
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MAX_VALUE  # try scheduling first
        )
        # Then decide WHEN (start times)
        model.AddDecisionStrategy(
            s_vars,
            cp_model.CHOOSE_LOWEST_MIN,
            cp_model.SELECT_MIN_VALUE  # try earliest start first
        )

    logger.debug("Added search strategy: %d p-vars, %d s-vars",
                 len(p_vars), len(s_vars))


def solve(
    instance: ProblemInstance,
    time_limit_seconds: int = 60,
    num_workers: int = 12,
    log_search: bool = False,
    hint_schedule: Optional[Dict[int, List[int]]] = None,
    use_symmetry_breaking: bool = True,
) -> SolverResult:
    """
    Solve a MaintAlign instance with optional tasks and chain costs.

    v3 optimizations: tighter bounds, smarter max_tasks, implied cuts,
    search strategy, and warm-start from baselines.
    """
    model = cp_model.CpModel()
    H = instance.horizon
    M = instance.num_machines
    K = instance.num_technicians

    logger.info("Building model: %dM, %dK, %dT, %d chains",
                M, K, H, len(instance.chains))

    # ─── Per-machine task bounds (OPTIMIZED: tighter) ───────────────
    min_tasks = {}
    max_tasks = {}
    total_slots_saved = 0
    for m_idx, machine in enumerate(instance.machines):
        min_tasks[m_idx] = _compute_min_tasks(machine, H)
        old_max = H // (machine.maintenance_duration + machine.min_gap)
        max_tasks[m_idx] = _compute_max_tasks_tight(machine, H, instance, m_idx)
        total_slots_saved += old_max - max_tasks[m_idx]

    logger.info("Reduced total task slots by %d (tighter bounds)", total_slots_saved)

    # ─── VARIABLES (OPTIMIZED: tighter start bounds) ────────────────
    present = {}
    start_var = {}
    interval_var = {}
    all_opt_intervals = []

    for m_idx, machine in enumerate(instance.machines):
        J = max_tasks[m_idx]
        d = machine.maintenance_duration
        present[m_idx] = []
        start_var[m_idx] = []
        interval_var[m_idx] = []

        for j in range(J):
            p = model.NewBoolVar(f"p_{m_idx}_{j}")

            # OPTIMIZED: tight start bounds per task slot
            lb, ub = _compute_start_bounds(machine, H, j, J)
            s = model.NewIntVar(lb, ub, f"s_{m_idx}_{j}")

            iv = model.NewOptionalFixedSizeIntervalVar(
                s, d, p, f"iv_{m_idx}_{j}"
            )
            present[m_idx].append(p)
            start_var[m_idx].append(s)
            interval_var[m_idx].append(iv)
            all_opt_intervals.append(iv)

    logger.info("Total variables: %d intervals", len(all_opt_intervals))

    # ─── C1: TECHNICIAN CAPACITY ────────────────────────────────────
    if all_opt_intervals:
        model.AddCumulative(
            all_opt_intervals,
            [1] * len(all_opt_intervals),
            K
        )

    # ─── C2: NO OVERLAP PER MACHINE ────────────────────────────────
    for m_idx in range(M):
        if len(interval_var[m_idx]) > 1:
            model.AddNoOverlap(interval_var[m_idx])

    # ─── C3: CONTIGUOUS NUMBERING ──────────────────────────────────
    for m_idx in range(M):
        J = max_tasks[m_idx]
        for j in range(J - 1):
            model.AddImplication(present[m_idx][j].Not(),
                                 present[m_idx][j+1].Not())

    # ─── C4: ORDERING + MINIMUM GAP ────────────────────────────────
    for m_idx, machine in enumerate(instance.machines):
        J = max_tasks[m_idx]
        d = machine.maintenance_duration
        g = machine.min_gap
        for j in range(J - 1):
            model.Add(
                start_var[m_idx][j+1] >= start_var[m_idx][j] + d + g
            ).OnlyEnforceIf(present[m_idx][j], present[m_idx][j+1])

    # ─── C5: MAXIMUM INTERVAL ──────────────────────────────────────
    for m_idx, machine in enumerate(instance.machines):
        J = max_tasks[m_idx]
        d = machine.maintenance_duration
        W = machine.max_interval
        n_min = min_tasks[m_idx]

        if n_min > 0:
            for j in range(n_min):
                model.Add(present[m_idx][j] == 1)

        if J > 0:
            model.Add(start_var[m_idx][0] <= W).OnlyEnforceIf(
                present[m_idx][0])

        for j in range(J - 1):
            model.Add(
                start_var[m_idx][j+1] - start_var[m_idx][j] - d <= W
            ).OnlyEnforceIf(present[m_idx][j], present[m_idx][j+1])

        for j in range(J):
            if j < J - 1:
                is_last = model.NewBoolVar(f"last_{m_idx}_{j}")
                model.AddBoolAnd([
                    present[m_idx][j], present[m_idx][j+1].Not()
                ]).OnlyEnforceIf(is_last)
                model.AddBoolOr([
                    present[m_idx][j].Not(), present[m_idx][j+1]
                ]).OnlyEnforceIf(is_last.Not())
            else:
                is_last = present[m_idx][j]

            model.Add(
                H - start_var[m_idx][j] - d <= W
            ).OnlyEnforceIf(is_last)

    # ─── OPTIMIZED: Symmetry breaking ──────────────────────────────
    if use_symmetry_breaking:
        _add_symmetry_breaking(model, instance, present, start_var, max_tasks)

    # ─── OPTIMIZED: Implied constraints ────────────────────────────
    _add_implied_constraints(model, instance, present, max_tasks)

    # ─── OPTIMIZED: Search strategy ────────────────────────────────
    _add_search_strategy(model, present, start_var, max_tasks, M)

    # ─── C7: CALENDAR MASKING (blocked periods) ───────────────────
    if instance.blocked_periods:
        blocked = sorted(set(instance.blocked_periods))
        num_calendar_constraints = 0
        for m_idx, machine in enumerate(instance.machines):
            d = machine.maintenance_duration
            for j in range(max_tasks[m_idx]):
                for b in blocked:
                    # Task (m, j) must not be running during period b
                    # A task starting at s runs during [s, s+d).
                    # Forbidden starts: s in [b - d + 1, b]
                    for t in range(max(0, b - d + 1), min(b + 1, H)):
                        model.Add(start_var[m_idx][j] != t).OnlyEnforceIf(
                            present[m_idx][j])
                        num_calendar_constraints += 1
        logger.info("Added %d calendar constraints for %d blocked periods",
                     num_calendar_constraints, len(blocked))

    # ─── Solver hints ──────────────────────────────────────────────
    _add_solver_hints(model, instance, present, start_var, max_tasks,
                      hint_schedule)


    # ─── OBJECTIVE (with Opportunistic Maintenance) ──────────────
    obj_terms = []

    # Step 1: PM cost + standalone production loss (always charged per task)
    for m_idx, machine in enumerate(instance.machines):
        J = max_tasks[m_idx]
        d = machine.maintenance_duration
        chain = instance.get_chain_for_machine(m_idx)

        # PM cost is always per-task
        pm_only = machine.pm_cost
        if not chain:
            # Standalone: add production loss
            pm_only += machine.production_value * d

        scaled_pm = int(pm_only * COST_SCALE)
        for j in range(J):
            tc = model.NewIntVar(0, scaled_pm, f"tc_{m_idx}_{j}")
            model.Add(tc == scaled_pm).OnlyEnforceIf(present[m_idx][j])
            model.Add(tc == 0).OnlyEnforceIf(present[m_idx][j].Not())
            obj_terms.append(tc)

    # Step 2: Opportunistic grouping for chains
    # For each chain, detect overlapping PM windows among chain machines.
    # If PMs overlap → share retooling + production loss (charge once).
    # If PMs don't overlap → charge retooling + production loss per event.
    for chain in instance.chains:
        mids = chain.machine_ids
        retool_scaled = int(chain.retooling_cost * COST_SCALE)
        prod_loss_scaled = int(chain.chain_value * COST_SCALE)

        # Collect all tasks for this chain's machines
        chain_task_list = []  # [(m_idx, j)]
        for mid in mids:
            for j in range(max_tasks[mid]):
                chain_task_list.append((mid, j))

        if not chain_task_list:
            continue

        # For each chain task: check if it overlaps with ANY other chain-mate task
        for i, (m1, j1) in enumerate(chain_task_list):
            d1 = instance.machines[m1].maintenance_duration

            # Does this task overlap with at least one other chain-mate task?
            has_overlap = []
            for k, (m2, j2) in enumerate(chain_task_list):
                if m1 == m2:
                    continue  # same machine, handled by NoOverlap
                d2 = instance.machines[m2].maintenance_duration

                # overlap_{m1,j1,m2,j2}: both present AND time windows overlap
                ov = model.NewBoolVar(f"ov_{m1}_{j1}_{m2}_{j2}")

                # Both tasks must be present
                both_present = model.NewBoolVar(f"bp_{m1}_{j1}_{m2}_{j2}")
                model.AddBoolAnd([present[m1][j1], present[m2][j2]]).OnlyEnforceIf(both_present)
                model.AddBoolOr([present[m1][j1].Not(), present[m2][j2].Not()]).OnlyEnforceIf(both_present.Not())

                # Time overlap: s1 < s2 + d2  AND  s2 < s1 + d1
                ov_time1 = model.NewBoolVar(f"ot1_{m1}_{j1}_{m2}_{j2}")
                ov_time2 = model.NewBoolVar(f"ot2_{m1}_{j1}_{m2}_{j2}")
                model.Add(start_var[m1][j1] < start_var[m2][j2] + d2).OnlyEnforceIf(ov_time1)
                model.Add(start_var[m1][j1] >= start_var[m2][j2] + d2).OnlyEnforceIf(ov_time1.Not())
                model.Add(start_var[m2][j2] < start_var[m1][j1] + d1).OnlyEnforceIf(ov_time2)
                model.Add(start_var[m2][j2] >= start_var[m1][j1] + d1).OnlyEnforceIf(ov_time2.Not())

                # Overlap = both present AND time overlap
                model.AddBoolAnd([both_present, ov_time1, ov_time2]).OnlyEnforceIf(ov)
                model.AddBoolOr([both_present.Not(), ov_time1.Not(), ov_time2.Not()]).OnlyEnforceIf(ov.Not())

                has_overlap.append(ov)

            # any_overlap: true if this task overlaps with at least one chain-mate
            any_ov = model.NewBoolVar(f"aov_{m1}_{j1}")
            if has_overlap:
                model.AddBoolOr(has_overlap).OnlyEnforceIf(any_ov)
                model.AddBoolAnd([ov.Not() for ov in has_overlap]).OnlyEnforceIf(any_ov.Not())
            else:
                model.Add(any_ov == 0)

            # If grouped (overlapping): discount retooling (pay half) + discount prod loss
            # If isolated: pay full retooling + full prod loss
            full_cost = retool_scaled + prod_loss_scaled * d1
            grouped_cost = retool_scaled // 2 + prod_loss_scaled * d1 // 2  # 50% discount

            chain_tc = model.NewIntVar(0, full_cost, f"ctc_{m1}_{j1}")
            # When present and grouped: discounted cost
            grouped_and_present = model.NewBoolVar(f"gp_{m1}_{j1}")
            model.AddBoolAnd([present[m1][j1], any_ov]).OnlyEnforceIf(grouped_and_present)
            model.AddBoolOr([present[m1][j1].Not(), any_ov.Not()]).OnlyEnforceIf(grouped_and_present.Not())

            # When present and NOT grouped: full cost
            isolated_and_present = model.NewBoolVar(f"ip_{m1}_{j1}")
            model.AddBoolAnd([present[m1][j1], any_ov.Not()]).OnlyEnforceIf(isolated_and_present)
            model.AddBoolOr([present[m1][j1].Not(), any_ov]).OnlyEnforceIf(isolated_and_present.Not())

            model.Add(chain_tc == grouped_cost).OnlyEnforceIf(grouped_and_present)
            model.Add(chain_tc == full_cost).OnlyEnforceIf(isolated_and_present)
            model.Add(chain_tc == 0).OnlyEnforceIf(present[m1][j1].Not())
            obj_terms.append(chain_tc)

    logger.info("Added opportunistic grouping for %d chains", len(instance.chains))

    # Step 3: Failure cost (Weibull) — always per-machine
    for m_idx, machine in enumerate(instance.machines):
        J = max_tasks[m_idx]
        d = machine.maintenance_duration

        fail_table = _precompute_failure_table(machine, H)
        max_fc = max(fail_table) if fail_table else 0

        if J == 0:
            obj_terms.append(fail_table[H])
            continue

        # Gap before first task
        gap0 = model.NewIntVar(0, H, f"g0_{m_idx}")
        model.Add(gap0 == start_var[m_idx][0]).OnlyEnforceIf(
            present[m_idx][0])
        model.Add(gap0 == H).OnlyEnforceIf(present[m_idx][0].Not())

        fc0 = model.NewIntVar(0, max_fc, f"fc0_{m_idx}")
        model.AddElement(gap0, fail_table, fc0)
        obj_terms.append(fc0)

        # Gaps between consecutive tasks
        for j in range(J - 1):
            both = model.NewBoolVar(f"bth_{m_idx}_{j}")
            model.AddBoolAnd([
                present[m_idx][j], present[m_idx][j+1]
            ]).OnlyEnforceIf(both)
            model.AddBoolOr([
                present[m_idx][j].Not(), present[m_idx][j+1].Not()
            ]).OnlyEnforceIf(both.Not())

            gap_j = model.NewIntVar(0, H, f"gm_{m_idx}_{j}")
            model.Add(
                gap_j == start_var[m_idx][j+1] - start_var[m_idx][j] - d
            ).OnlyEnforceIf(both)
            model.Add(gap_j == 0).OnlyEnforceIf(both.Not())

            fc_j = model.NewIntVar(0, max_fc, f"fcm_{m_idx}_{j}")
            model.AddElement(gap_j, fail_table, fc_j)

            fc_j_act = model.NewIntVar(0, max_fc, f"fca_{m_idx}_{j}")
            model.Add(fc_j_act == fc_j).OnlyEnforceIf(both)
            model.Add(fc_j_act == 0).OnlyEnforceIf(both.Not())
            obj_terms.append(fc_j_act)

        # Gap after last present task to horizon
        for j in range(J):
            if j < J - 1:
                is_last = model.NewBoolVar(f"el_{m_idx}_{j}")
                model.AddBoolAnd([
                    present[m_idx][j], present[m_idx][j+1].Not()
                ]).OnlyEnforceIf(is_last)
                model.AddBoolOr([
                    present[m_idx][j].Not(), present[m_idx][j+1]
                ]).OnlyEnforceIf(is_last.Not())
            else:
                is_last = present[m_idx][j]

            gap_e = model.NewIntVar(0, H, f"ge_{m_idx}_{j}")
            model.Add(
                gap_e == H - start_var[m_idx][j] - d
            ).OnlyEnforceIf(is_last)
            model.Add(gap_e == 0).OnlyEnforceIf(is_last.Not())

            fc_e = model.NewIntVar(0, max_fc, f"fce_{m_idx}_{j}")
            model.AddElement(gap_e, fail_table, fc_e)

            fc_e_act = model.NewIntVar(0, max_fc, f"fcea_{m_idx}_{j}")
            model.Add(fc_e_act == fc_e).OnlyEnforceIf(is_last)
            model.Add(fc_e_act == 0).OnlyEnforceIf(is_last.Not())
            obj_terms.append(fc_e_act)

    model.Minimize(sum(obj_terms))

    # ─── SOLVE ────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_workers = num_workers
    solver.parameters.log_search_progress = log_search

    logger.info("Solving (limit=%ds, workers=%d)...",
                time_limit_seconds, num_workers)
    t0 = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - t0

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(status, "UNKNOWN")

    logger.info("Solver finished: %s in %.2fs (obj=%.2f)",
                status_name, solve_time,
                solver.ObjectiveValue() / COST_SCALE
                if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else float('inf'))

    empty = SolverResult(
        status=status_name, objective_value=float('inf'),
        solve_time_seconds=solve_time, tasks=[],
        total_pm_cost=0, total_production_loss=0,
        total_retooling_cost=0, total_failure_cost=0,
        machine_schedules={},
    )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.warning("No feasible solution found (status=%s)", status_name)
        return empty

    # ─── EXTRACT SOLUTION ──────────────────────────────────────────
    tasks = []
    machine_schedules = {}
    total_pm = 0.0
    total_prod = 0.0
    total_retool = 0.0
    total_fail = 0.0
    chain_costs = {c.id: {"prod_loss": 0.0, "retooling": 0.0, "num_events": 0}
                   for c in instance.chains}

    for m_idx, machine in enumerate(instance.machines):
        J = max_tasks[m_idx]
        machine_schedules[m_idx] = []
        chain = instance.get_chain_for_machine(m_idx)

        for j in range(J):
            if solver.Value(present[m_idx][j]):
                s = solver.Value(start_var[m_idx][j])
                d = machine.maintenance_duration

                pm_c = machine.pm_cost
                if chain:
                    prod_c = chain.chain_value * d
                    ret_c = chain.retooling_cost
                    chain_costs[chain.id]["prod_loss"] += prod_c
                    chain_costs[chain.id]["retooling"] += ret_c
                    chain_costs[chain.id]["num_events"] += 1
                else:
                    prod_c = machine.production_value * d
                    ret_c = 0

                total_pm += pm_c
                total_prod += prod_c
                total_retool += ret_c

                tasks.append(MaintenanceTask(
                    machine_id=m_idx, task_index=j,
                    start_time=s, end_time=s + d,
                    cost_pm=pm_c, cost_prod_loss=prod_c,
                    cost_retooling=ret_c,
                    chain_id=chain.id if chain else None,
                ))
                machine_schedules[m_idx].append(s)

        # Failure cost from gaps
        starts = sorted(machine_schedules[m_idx])
        prev_end = 0
        for sv in starts:
            total_fail += machine.expected_failure_cost(sv - prev_end)
            prev_end = sv + machine.maintenance_duration
        total_fail += machine.expected_failure_cost(H - prev_end)

    result = SolverResult(
        status=status_name,
        objective_value=solver.ObjectiveValue() / COST_SCALE,
        solve_time_seconds=solve_time,
        tasks=tasks,
        total_pm_cost=total_pm,
        total_production_loss=total_prod,
        total_retooling_cost=total_retool,
        total_failure_cost=total_fail,
        machine_schedules=machine_schedules,
        chain_costs=chain_costs,
    )
    logger.info("Solution: $%.2f total, %d tasks scheduled",
                result.objective_value, len(tasks))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(name)s | %(message)s")
    from generator import generate_tiny, generate_small, generate_medium_easy

    print("MaintAlign CP-SAT Solver v3 (Optimized Performance)\n")

    for gen, label in [
        (generate_tiny, "TINY"),
        (generate_small, "SMALL"),
        (generate_medium_easy, "MEDIUM-EASY"),
    ]:
        inst = gen()
        print(inst.summary())
        result = solve(inst, time_limit_seconds=30)
        print(result.summary())
        print()
