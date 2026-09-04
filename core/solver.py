"""
MaintAlign - CP-SAT Solver (v5: exact chain outages, valid task bounds)
=======================================================================

WHAT THE MODEL IS
  Optional fixed-size interval per potential task, a cumulative constraint
  for technician capacity, no-overlap per machine, and Weibull failure cost
  looked up from a table over the gap between maintenance events. See
  ``core.instance`` for the full formulation.

WHAT CHANGED IN v5 (and why)
  Chain outages   Chain production loss and retooling are priced with
         per-period chain-down indicators: ``down[c][t]`` is forced to 1
         whenever any machine of chain c is under maintenance in period t,
         and ``restart[c][t] >= down[c][t] - down[c][t-1]``. The objective
         pays V_c per down period and R_c per restart. Because both are
         minimised, they take exactly the OR / rising-edge values, so the
         solver's objective equals ``core.costing.deterministic_cost`` of
         the schedule it returns. The previous 50% "grouping discount" per
         overlapping task was right for two machines by coincidence and
         charged 1.5 outages for three.
  Task slots      Slots per machine default to the physical packing bound
         n*d + (n-1)*g <= H, which is a valid upper bound, so OPTIMAL means
         optimal. The old heuristic cap (from the analytical interval) is
         kept as an explicitly labelled restricted model:
         ``solve(..., task_slot_bound="heuristic")`` sets
         ``SolverResult.model_variant = "restricted-slots"``.
  Imperfect PM    ``repair_factor`` is wired in. Failure cost over a gap
         starting at virtual age a is T[a + g] - T[a]; the virtual age after
         a PM is a second table lookup, round((1 - r) * age). With r = 1 the
         virtual age is the constant 0 and the model reduces to the old one.
  Breakdown cost  c_cm is all-in; the objective charges E[N] * c_cm and
         nothing else for failures (unchanged here, now matched by the
         simulator).

PERFORMANCE FEATURES KEPT FROM v4
  Tight start bounds per task slot, pre-fixed mandatory tasks, presence-
  first search strategy, warm-start hints from the best baseline, symmetry
  breaking for identical machines, linearization level 2.
"""

import logging
import time
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from core.costing import apply_task_attribution, deterministic_cost
from core.instance import MachineSpec, ProblemInstance

logger = logging.getLogger(__name__)

COST_SCALE = 100  # multiply float costs by this for integer CP-SAT

TASK_SLOT_BOUNDS = ("physical", "heuristic")
MODEL_VARIANTS = {"physical": "full", "heuristic": "restricted-slots"}


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
    chain_id: int | None = None


@dataclass
class SolverResult:
    """Complete solution with cost breakdown."""
    status: str
    objective_value: float
    solve_time_seconds: float
    tasks: list[MaintenanceTask]
    total_pm_cost: float
    total_production_loss: float
    total_retooling_cost: float
    total_failure_cost: float
    machine_schedules: dict[int, list[int]]
    chain_costs: dict[int, dict[str, float]] = field(default_factory=dict)
    # Proof quality of this solution. A FEASIBLE result is an incumbent, not
    # an optimum: without the bound there is no way to tell "within 2% of
    # optimal" from "gave up". NaN for schedules the CP-SAT solver did not
    # produce (baselines, decomposition).
    best_bound: float = float("nan")
    gap_pct: float = float("nan")
    # Which model the status refers to. "full": task slots use the physical
    # packing bound, so OPTIMAL is optimal. "restricted-slots": slots were
    # capped by a heuristic, so OPTIMAL is optimal within that restriction
    # only (the true optimum may be unrepresentable). "n/a" for schedules
    # that did not come from the CP-SAT solver.
    model_variant: str = "n/a"

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
        ]
        if self.best_bound == self.best_bound:  # not NaN
            lines.append(f" Best Bound:        ${self.best_bound:>12,.2f}"
                         f"   (gap {self.gap_pct:.2f}%)")
        if self.model_variant == "restricted-slots":
            lines.append(" Model: RESTRICTED (task slots capped by heuristic) — "
                         "status is relative to this model")
        elif self.model_variant == "full":
            lines.append(" Model: full (task slots = physical packing bound)")
        lines += [
            f" Solve Time: {self.solve_time_seconds:.3f}s",
            f" Tasks Scheduled: {len(self.tasks)}",
            f"{'─'*55}",
            " Schedule:",
        ]
        chain_of = {t.machine_id: t.chain_id for t in self.tasks
                    if t.chain_id is not None}
        for mid, starts in sorted(self.machine_schedules.items()):
            tag = f" (chain {chain_of[mid]})" if mid in chain_of else ""
            lines.append(f"   M{mid}: PM at t={starts}{tag}")

        if self.chain_costs:
            lines.append(f"{'─'*55}")
            lines.append(" Chain Cost Breakdown:")
            for cid, cc in sorted(self.chain_costs.items()):
                runs = cc.get("num_runs")
                runs_s = f"  outages={runs}" if runs is not None else ""
                lines.append(
                    f"   Chain {cid}: prod_loss=${cc['prod_loss']:,.0f}  "
                    f"retool=${cc['retooling']:,.0f}  "
                    f"events={cc['num_events']}{runs_s}"
                )
        return "\n".join(lines)


def _precompute_failure_table(machine: MachineSpec, horizon: int) -> list[int]:
    """Table: age → integer-scaled expected failure cost from age 0.

    T[x] = int(c_cm * (x/η)^β * COST_SCALE). The cost over a gap that starts
    at virtual age a and ends at age a + g is T[a + g] - T[a]; with perfect
    repair a = 0 and this is T[g]. Monotone non-decreasing in x.
    """
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


def _compute_max_tasks_physical(machine: MachineSpec, horizon: int) -> int:
    """Valid upper bound: the most PM windows that physically fit.

    n windows of length d separated by at least g need n*d + (n-1)*g <= H,
    i.e. n <= (H + g) // (d + g). Any schedule with more tasks violates the
    min-gap or the horizon, so an optimum can never need more slots than
    this — which is what makes OPTIMAL mean optimal.
    """
    d = machine.maintenance_duration
    g = machine.min_gap
    phys = (horizon + g) // (d + g)
    return max(_compute_min_tasks(machine, horizon), phys)


def _compute_max_tasks_heuristic(machine: MachineSpec, horizon: int,
                                 m_idx: int) -> int:
    """RESTRICTED MODEL: cap task slots by an economic estimate.

    Uses the analytical optimal interval (or half of max_interval) to
    estimate how many tasks a sensible schedule has, plus one. This is NOT
    a valid bound: the optimum may need more tasks than the estimate, in
    which case CP-SAT reports OPTIMAL for a model that cannot express it.
    Kept as an explicitly labelled variant for speed comparisons
    (``SolverResult.model_variant == "restricted-slots"``); never quote its
    OPTIMAL as a proof.
    """
    d = machine.maintenance_duration
    g = machine.min_gap
    phys_max = _compute_max_tasks_physical(machine, horizon)

    t_star = machine.optimal_interval_analytical()
    if t_star < float('inf') and t_star > 0:
        practical_interval = max(d + g, int(t_star))
    else:
        practical_interval = max(d + g, machine.max_interval // 2)

    estimated = horizon // (practical_interval + d) + 1
    n_min = _compute_min_tasks(machine, horizon)
    tight = max(n_min, min(phys_max, estimated))

    if tight < phys_max:
        logger.debug("M%d: max_tasks %d → %d (restricted model)",
                     m_idx, phys_max, tight)
    return tight


def _compute_start_bounds(machine: MachineSpec, horizon: int,
                          j: int, n_tasks: int) -> tuple:
    """
    Tight [lb, ub] for start_var[m][j]. Task j must come after j earlier
    tasks (each at least d + g long) and cannot start later than the
    max-interval policy allows for the (j+1)-th event.
    """
    d = machine.maintenance_duration
    g = machine.min_gap
    W = machine.max_interval

    lb = j * (d + g)
    ub = horizon - d
    ub_from_W = min(ub, (j + 1) * W + j * d)
    ub = min(ub, ub_from_W)

    return max(0, lb), max(lb, ub)


def _add_solver_hints(model, instance, present, start_var, at_lits, max_tasks,
                      hint_schedule: dict[int, list[int]] | None = None):
    """Add solver hints from a baseline solution to warm-start the search."""
    if hint_schedule is None:
        return

    for m_idx, _machine in enumerate(instance.machines):
        J = max_tasks[m_idx]
        hint_starts = sorted(hint_schedule.get(m_idx, []))

        for j in range(J):
            if j < len(hint_starts):
                model.AddHint(present[m_idx][j], 1)
                model.AddHint(start_var[m_idx][j], hint_starts[j])
                if m_idx in at_lits:
                    for v, lit in at_lits[m_idx][j].items():
                        model.AddHint(lit, int(v == hint_starts[j]))
            else:
                model.AddHint(present[m_idx][j], 0)
                if m_idx in at_lits:
                    for lit in at_lits[m_idx][j].values():
                        model.AddHint(lit, 0)

    logger.debug("Added solver hints from baseline schedule")


def _add_symmetry_breaking(model, instance, present, start_var, max_tasks):
    """Symmetry breaking: order first tasks of truly interchangeable machines.

    Identical machine parameters are not enough for interchangeability: chain
    membership changes the outage and retooling costs around a machine.  Keep
    chain machines in a group only when they belong to the same chain, and keep
    standalone machines in a separate scope keyed by their production value.
    """
    groups = {}
    for m_idx, m in enumerate(instance.machines):
        chain = instance.get_chain_for_machine(m_idx)
        economic_scope = (("chain", chain.id) if chain is not None
                          else ("standalone", m.production_value))
        key = (m.maintenance_duration, m.pm_cost, m.cm_cost, m.max_interval,
               m.min_gap, m.weibull_beta, m.weibull_eta, m.repair_factor,
               economic_scope)
        groups.setdefault(key, []).append(m_idx)

    count = 0
    for _key, mids in groups.items():
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


def _add_time_indexed_starts(model, instance, present, start_var, start_bounds,
                             max_tasks) -> dict[int, list[dict]]:
    """Literals ``at[m][j][v]`` ⇔ task (m, j) is present and starts at v.

    Only built for machines that belong to a chain; standalone machines
    do not need them. Linked to the integer start by
    Σ_v at[v] == present and start == Σ_v v·at[v] (when present).
    """
    at_lits: dict[int, list[dict]] = {}
    for chain in instance.chains:
        for mid in chain.machine_ids:
            at_lits[mid] = []
            for j in range(max_tasks[mid]):
                lb, ub = start_bounds[mid][j]
                lits = {v: model.NewBoolVar(f"at_{mid}_{j}_{v}")
                        for v in range(lb, ub + 1)}
                model.Add(sum(lits.values()) == present[mid][j])
                model.Add(start_var[mid][j]
                          == sum(v * lit for v, lit in lits.items())
                          ).OnlyEnforceIf(present[mid][j])
                at_lits[mid].append(lits)
    return at_lits


def _add_chain_outage_costs(model, instance, present, at_lits, max_tasks,
                            obj_terms, redundant_cuts=True):
    """Per-period chain-down indicators (replaces the 50% heuristic).

    down[c][t]    ≥ Σ_{tasks of machine m in c that cover t} at-literal, for
                    every machine m of the chain. The sum is ≤ 1 because of
                    the machine's NoOverlap, so this is a valid lower bound
                    that forces the chain down whenever any of its machines
                    is under maintenance.
    restart[c][t] ≥ down[c][t] − down[c][t−1]   (restart[c][0] ≥ down[c][0])

    Objective: V_c·Σ_t down[c][t] + R_c·Σ_t restart[c][t].

    Half-reified on purpose: nothing forces an indicator DOWN, but both are
    minimised, so at an optimum they are exactly the OR and the rising edge.
    A time-limited incumbent can only over-charge a chain, never under-charge
    it, which keeps ``deterministic_cost(schedule) <= objective``.
    """
    H = instance.horizon
    n_bool = 0
    for chain in instance.chains:
        V = int(chain.chain_value * COST_SCALE)
        R = int(chain.retooling_cost * COST_SCALE)
        if V == 0 and R == 0:
            continue

        down = [model.NewBoolVar(f"down_{chain.id}_{t}") for t in range(H)]
        restart = [model.NewBoolVar(f"restart_{chain.id}_{t}") for t in range(H)]
        n_bool += 2 * H

        any_covering: list[list] = [[] for _ in range(H)]
        for mid in chain.machine_ids:
            d = instance.machines[mid].maintenance_duration
            covering: list[list] = [[] for _ in range(H)]
            for j in range(max_tasks[mid]):
                for v, lit in at_lits[mid][j].items():
                    for t in range(v, min(H, v + d)):
                        covering[t].append(lit)
            for t in range(H):
                if covering[t]:
                    model.Add(sum(covering[t]) <= down[t])
                    any_covering[t].extend(covering[t])
            # Redundant but LP-tightening: this machine alone keeps the chain
            # down for d periods per PM, and any PM at all forces a restart.
            if redundant_cuts and max_tasks[mid] > 0:
                model.Add(sum(down) >= d * sum(present[mid]))
                model.Add(sum(restart) >= present[mid][0])

        # Make the indicators exact (down ⇒ some task covers t; restart ⇒
        # rising edge). Not needed for correctness of the optimum — the
        # objective already pulls them down — but it lets propagation push
        # back from a cost bound onto the schedule.
        for t in range(H):
            if any_covering[t]:
                model.AddBoolOr(any_covering[t]).OnlyEnforceIf(down[t])
            else:
                model.Add(down[t] == 0)
            model.AddImplication(restart[t], down[t])
            if t > 0:
                model.AddImplication(restart[t], down[t - 1].Not())

        model.Add(restart[0] >= down[0])
        for t in range(1, H):
            model.Add(restart[t] >= down[t] - down[t - 1])

        for t in range(H):
            if V:
                obj_terms.append(V * down[t])
            if R:
                obj_terms.append(R * restart[t])

    logger.info("Added per-period outage indicators for %d chains (%d booleans)",
                len(instance.chains), n_bool)


def _failure_cost_floor(machine: MachineSpec, horizon: int, n: int) -> int:
    """Least integer-scaled failure cost of ANY schedule with n PMs.

    The n + 1 gaps sum to H − n·d whatever the placement, and the expected
    failure cost is convex in the gap when beta >= 1, so by Jensen the balanced
    partition costs least (a valid floor under imperfect repair too, since a
    gap starting at virtual age a > 0 costs at least as much as one from 0).
    For beta < 1 the cumulative hazard is concave and both claims reverse, so
    this redundant cut is disabled rather than risk excluding valid schedules.
    One scaled unit per gap is subtracted to cover the table's truncation.
    Redundant with the model, but it hands the LP relaxation the real
    trade-off between "how many PMs" and "how much failure risk".
    """
    if machine.weibull_beta < 1:
        return 0

    G = horizon - n * machine.maintenance_duration
    if G <= 0:
        return 0
    parts = n + 1
    q, r = divmod(G, parts)
    exact = (r * machine.expected_failure_cost(q + 1)
             + (parts - r) * machine.expected_failure_cost(q)) * COST_SCALE
    return max(0, int(exact) - parts)


def _add_failure_costs(model, instance, present, start_var, is_last, max_tasks,
                       obj_terms, redundant_cuts=True):
    """Term (5): expected breakdown cost over every gap, with virtual age.

    For task slot j of machine m:
      gap_j   = start_j − end_{j−1} when present (gap_0 = start_0; if no PM
                at all, gap_0 = H so the whole horizon is one gap)
      age_j   = va_j + gap_j        (va_j = virtual age at the gap's start)
      cost    = T[age_j] − T[va_j]  (0 when absent: gap 0 ⇒ same age)
      va_{j+1} = VA[age_j]          (VA[x] = round((1−r)·x); 0 if r = 1)
    plus, for the last present task, the gap to the horizon starting at
    va_{j+1}. Constants are folded: with perfect repair no virtual-age
    variables are created and the model is the plain (g/η)^β lookup.
    """
    H = instance.horizon

    def lookup(index, table, name):
        if isinstance(index, int):
            return table[index]
        target = model.NewIntVar(0, max(table), name)
        model.AddElement(index, table, target)
        return target

    def plus(a, b, name):
        if isinstance(a, int) and a == 0:
            return b
        v = model.NewIntVar(0, H, name)
        model.Add(v == a + b)
        return v

    for m_idx, machine in enumerate(instance.machines):
        J = max_tasks[m_idx]
        d = machine.maintenance_duration
        table = _precompute_failure_table(machine, H)

        if J == 0:
            obj_terms.append(table[H])
            continue

        imperfect = machine.repair_factor < 1.0
        va_table = ([machine.virtual_age_after_pm(x) for x in range(H + 1)]
                    if imperfect else None)

        va = 0  # virtual age at the start of the gap before task j
        machine_terms = []
        for j in range(J):
            p = present[m_idx][j]

            gap = model.NewIntVar(0, H, f"gap_{m_idx}_{j}")
            if j == 0:
                model.Add(gap == start_var[m_idx][0]).OnlyEnforceIf(p)
                model.Add(gap == H).OnlyEnforceIf(p.Not())
            else:
                model.Add(gap == start_var[m_idx][j] - start_var[m_idx][j - 1] - d
                          ).OnlyEnforceIf(p)
                model.Add(gap == 0).OnlyEnforceIf(p.Not())

            age = plus(va, gap, f"age_{m_idx}_{j}")
            machine_terms.append(lookup(age, table, f"fc_{m_idx}_{j}")
                                 - lookup(va, table, f"fcv_{m_idx}_{j}"))

            va_next = lookup(age, va_table, f"va_{m_idx}_{j + 1}") if imperfect else 0

            gap_e = model.NewIntVar(0, H, f"gape_{m_idx}_{j}")
            model.Add(gap_e == H - start_var[m_idx][j] - d).OnlyEnforceIf(is_last[m_idx][j])
            model.Add(gap_e == 0).OnlyEnforceIf(is_last[m_idx][j].Not())
            age_e = plus(va_next, gap_e, f"agee_{m_idx}_{j}")
            machine_terms.append(lookup(age_e, table, f"fce_{m_idx}_{j}")
                                 - lookup(va_next, table, f"fcev_{m_idx}_{j}"))

            va = va_next

        obj_terms.extend(machine_terms)

        # Redundant floor: failure cost ≥ best case for this many PMs.
        floors = [_failure_cost_floor(machine, H, n) for n in range(J + 1)]
        if redundant_cuts and any(floors):
            n_present = model.NewIntVar(0, J, f"n_{m_idx}")
            model.Add(n_present == sum(present[m_idx]))
            floor_var = model.NewIntVar(0, max(floors), f"ffloor_{m_idx}")
            model.AddElement(n_present, floors, floor_var)
            model.Add(sum(machine_terms) >= floor_var)


def _add_search_strategy(model, present, start_var, max_tasks, M):
    """
    Guide solver search by prioritizing decisions.
    Branch on presence variables first (the key 0/1 decisions),
    then start times. This front-loads the hardest decisions.
    """
    p_vars = []
    s_vars = []
    for m_idx in range(M):
        for j in range(max_tasks[m_idx]):
            p_vars.append(present[m_idx][j])
            s_vars.append(start_var[m_idx][j])

    if p_vars:
        model.AddDecisionStrategy(
            p_vars,
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MAX_VALUE  # try scheduling first
        )
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
    hint_schedule: dict[int, list[int]] | None = None,
    use_symmetry_breaking: bool = True,
    task_slot_bound: str = "physical",
    redundant_cuts: bool = True,
) -> SolverResult:
    """
    Solve a MaintAlign instance with optional tasks and chain costs.

    Args:
        task_slot_bound: "physical" (default) gives every machine as many
            task slots as can physically fit, so OPTIMAL is optimal.
            "heuristic" caps slots by an economic estimate — a smaller,
            faster, RESTRICTED model whose OPTIMAL is only optimal within
            the restriction. The choice is recorded in
            ``SolverResult.model_variant``.
        redundant_cuts: add the valid, implied inequalities that tighten
            the LP relaxation (per-machine failure-cost floor, presence-to-
            outage links). They never change the optimum; the switch exists
            so that can be checked, and for speed comparisons.
    """
    if task_slot_bound not in TASK_SLOT_BOUNDS:
        raise ValueError(f"task_slot_bound must be one of {TASK_SLOT_BOUNDS}, "
                         f"got {task_slot_bound!r}")
    restricted = task_slot_bound == "heuristic"
    model_variant = MODEL_VARIANTS[task_slot_bound]

    model = cp_model.CpModel()
    H = instance.horizon
    M = instance.num_machines
    K = instance.num_technicians

    logger.info("Building model: %dM, %dK, %dT, %d chains (%s)",
                M, K, H, len(instance.chains), model_variant)

    # ─── Per-machine task bounds ────────────────────────────────────
    min_tasks = {}
    max_tasks = {}
    for m_idx, machine in enumerate(instance.machines):
        min_tasks[m_idx] = _compute_min_tasks(machine, H)
        if restricted:
            max_tasks[m_idx] = _compute_max_tasks_heuristic(machine, H, m_idx)
        else:
            max_tasks[m_idx] = _compute_max_tasks_physical(machine, H)

    logger.info("Task slots: %d total", sum(max_tasks.values()))

    # ─── VARIABLES ──────────────────────────────────────────────────
    present = {}
    start_var = {}
    interval_var = {}
    start_bounds = {}
    is_last = {}
    all_opt_intervals = []

    for m_idx, machine in enumerate(instance.machines):
        J = max_tasks[m_idx]
        d = machine.maintenance_duration
        present[m_idx] = []
        start_var[m_idx] = []
        interval_var[m_idx] = []
        start_bounds[m_idx] = []
        is_last[m_idx] = []

        for j in range(J):
            p = model.NewBoolVar(f"p_{m_idx}_{j}")
            lb, ub = _compute_start_bounds(machine, H, j, J)
            s = model.NewIntVar(lb, ub, f"s_{m_idx}_{j}")
            iv = model.NewOptionalFixedSizeIntervalVar(s, d, p, f"iv_{m_idx}_{j}")
            present[m_idx].append(p)
            start_var[m_idx].append(s)
            interval_var[m_idx].append(iv)
            start_bounds[m_idx].append((lb, ub))
            all_opt_intervals.append(iv)

        # is_last[j] ⇔ task j present and task j+1 absent (or j is the final slot)
        for j in range(J):
            if j < J - 1:
                lit = model.NewBoolVar(f"last_{m_idx}_{j}")
                model.AddBoolAnd([present[m_idx][j], present[m_idx][j + 1].Not()]
                                 ).OnlyEnforceIf(lit)
                model.AddBoolOr([present[m_idx][j].Not(), present[m_idx][j + 1]]
                                ).OnlyEnforceIf(lit.Not())
            else:
                lit = present[m_idx][j]
            is_last[m_idx].append(lit)

    logger.info("Total variables: %d intervals", len(all_opt_intervals))

    # ─── C1: TECHNICIAN CAPACITY ────────────────────────────────────
    if all_opt_intervals:
        model.AddCumulative(all_opt_intervals, [1] * len(all_opt_intervals), K)

    # ─── C2: NO OVERLAP PER MACHINE ────────────────────────────────
    for m_idx in range(M):
        if len(interval_var[m_idx]) > 1:
            model.AddNoOverlap(interval_var[m_idx])

    # ─── C3: CONTIGUOUS NUMBERING ──────────────────────────────────
    for m_idx in range(M):
        J = max_tasks[m_idx]
        for j in range(J - 1):
            model.AddImplication(present[m_idx][j].Not(), present[m_idx][j + 1].Not())

    # ─── C4: ORDERING + MINIMUM GAP ────────────────────────────────
    for m_idx, machine in enumerate(instance.machines):
        J = max_tasks[m_idx]
        d = machine.maintenance_duration
        g = machine.min_gap
        for j in range(J - 1):
            model.Add(
                start_var[m_idx][j + 1] >= start_var[m_idx][j] + d + g
            ).OnlyEnforceIf(present[m_idx][j], present[m_idx][j + 1])

    # ─── C5: MAXIMUM INTERVAL ──────────────────────────────────────
    for m_idx, machine in enumerate(instance.machines):
        J = max_tasks[m_idx]
        d = machine.maintenance_duration
        W = machine.max_interval
        n_min = min_tasks[m_idx]

        for j in range(n_min):
            model.Add(present[m_idx][j] == 1)

        if J > 0:
            model.Add(start_var[m_idx][0] <= W).OnlyEnforceIf(present[m_idx][0])

        for j in range(J - 1):
            model.Add(
                start_var[m_idx][j + 1] - start_var[m_idx][j] - d <= W
            ).OnlyEnforceIf(present[m_idx][j], present[m_idx][j + 1])

        for j in range(J):
            model.Add(H - start_var[m_idx][j] - d <= W).OnlyEnforceIf(is_last[m_idx][j])

    # ─── Symmetry breaking ─────────────────────────────────────────
    if use_symmetry_breaking:
        _add_symmetry_breaking(model, instance, present, start_var, max_tasks)

    # ─── Search strategy ───────────────────────────────────────────
    _add_search_strategy(model, present, start_var, max_tasks, M)

    # ─── C7: CALENDAR MASKING (blocked periods) ───────────────────
    if instance.blocked_periods:
        blocked = sorted(set(instance.blocked_periods))
        num_calendar_constraints = 0
        for m_idx, machine in enumerate(instance.machines):
            d = machine.maintenance_duration
            for j in range(max_tasks[m_idx]):
                for b in blocked:
                    # A task starting at s runs during [s, s+d); it must not
                    # run during b, so starts in [b - d + 1, b] are forbidden.
                    for t in range(max(0, b - d + 1), min(b + 1, H)):
                        model.Add(start_var[m_idx][j] != t).OnlyEnforceIf(
                            present[m_idx][j])
                        num_calendar_constraints += 1
        logger.info("Added %d calendar constraints for %d blocked periods",
                    num_calendar_constraints, len(blocked))

    # ─── Time-indexed start literals for chain machines ────────────
    at_lits = _add_time_indexed_starts(model, instance, present, start_var,
                                       start_bounds, max_tasks)

    # ─── Solver hints ──────────────────────────────────────────────
    _add_solver_hints(model, instance, present, start_var, at_lits, max_tasks,
                      hint_schedule)

    # ─── OBJECTIVE ─────────────────────────────────────────────────
    obj_terms = []

    # (1) PM cost + (2) standalone production loss: per task
    for m_idx, machine in enumerate(instance.machines):
        J = max_tasks[m_idx]
        d = machine.maintenance_duration
        per_task = machine.pm_cost
        if instance.get_chain_for_machine(m_idx) is None:
            per_task += machine.production_value * d
        scaled = int(per_task * COST_SCALE)
        for j in range(J):
            obj_terms.append(scaled * present[m_idx][j])

    # (3) chain production loss + (4) retooling: per-period outage indicators
    _add_chain_outage_costs(model, instance, present, at_lits, max_tasks,
                            obj_terms, redundant_cuts=redundant_cuts)

    # (5) expected failure cost: per gap, with virtual age
    _add_failure_costs(model, instance, present, start_var, is_last, max_tasks,
                       obj_terms, redundant_cuts=redundant_cuts)

    model.Minimize(sum(obj_terms))

    # ─── SOLVE ────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_workers = num_workers
    solver.parameters.log_search_progress = log_search
    solver.parameters.linearization_level = 2

    logger.info("Solving (limit=%ds, workers=%d)...", time_limit_seconds, num_workers)
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

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.warning("No feasible solution found (status=%s)", status_name)
        return SolverResult(
            status=status_name, objective_value=float('inf'),
            solve_time_seconds=solve_time, tasks=[],
            total_pm_cost=0, total_production_loss=0,
            total_retooling_cost=0, total_failure_cost=0,
            machine_schedules={}, model_variant=model_variant,
        )

    if restricted and status == cp_model.OPTIMAL:
        logger.warning("OPTIMAL is relative to the RESTRICTED model "
                       "(task slots capped by heuristic); the true optimum "
                       "may need more tasks than the model allows")

    # ─── EXTRACT SOLUTION ──────────────────────────────────────────
    tasks = []
    machine_schedules = {}

    for m_idx, machine in enumerate(instance.machines):
        J = max_tasks[m_idx]
        machine_schedules[m_idx] = []
        chain = instance.get_chain_for_machine(m_idx)
        d = machine.maintenance_duration

        for j in range(J):
            if solver.Value(present[m_idx][j]):
                s = solver.Value(start_var[m_idx][j])
                if chain:
                    prod_c, ret_c = 0.0, 0.0  # attributed below
                else:
                    prod_c, ret_c = machine.production_value * d, 0.0
                tasks.append(MaintenanceTask(
                    machine_id=m_idx, task_index=j,
                    start_time=s, end_time=s + d,
                    cost_pm=machine.pm_cost, cost_prod_loss=prod_c,
                    cost_retooling=ret_c,
                    chain_id=chain.id if chain else None,
                ))
                machine_schedules[m_idx].append(s)

    # ─── PRICE THE SCHEDULE (single source of truth) ───────────────
    # The reported breakdown uses the same accounting as the objective just
    # minimised, so the numbers on a SolverResult add up to its own
    # objective_value (up to failure-table truncation).
    breakdown = deterministic_cost(instance, machine_schedules)
    apply_task_attribution(instance, tasks, machine_schedules)

    obj_value = solver.ObjectiveValue() / COST_SCALE
    bound = solver.BestObjectiveBound() / COST_SCALE
    gap = (abs(obj_value - bound) / abs(obj_value) * 100.0) if obj_value else 0.0

    result = SolverResult(
        status=status_name,
        objective_value=obj_value,
        solve_time_seconds=solve_time,
        tasks=tasks,
        total_pm_cost=breakdown.pm_cost,
        total_production_loss=breakdown.production_loss,
        total_retooling_cost=breakdown.retooling_cost,
        total_failure_cost=breakdown.failure_cost,
        machine_schedules=machine_schedules,
        chain_costs=breakdown.chain_costs,
        best_bound=bound,
        gap_pct=gap,
        model_variant=model_variant,
    )
    logger.info("Solution: $%.2f total, %d tasks scheduled",
                result.objective_value, len(tasks))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(name)s | %(message)s")
    from utils.generator import generate_medium_easy, generate_small, generate_tiny

    print("MaintAlign CP-SAT Solver v5\n")

    for gen in (generate_tiny, generate_small, generate_medium_easy):
        inst = gen()
        print(inst.summary())
        result = solve(inst, time_limit_seconds=30)
        print(result.summary())
        print()
