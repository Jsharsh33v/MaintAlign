"""
MaintAlign - Deterministic Schedule Costing (single source of truth)
=====================================================================

ONE function prices a schedule, whoever produced it: the CP-SAT solver,
a baseline heuristic, the decomposer, the Monte Carlo simulator (for the
deterministic part of a realisation), or a schedule typed in by hand.

WHY THIS MODULE EXISTS
----------------------
Before this module, three different cost accountings coexisted:

  * ``core.solver`` minimised an objective that granted overlapping
    chain maintenance a 50% discount on retooling and chain production
    loss, but then reported a breakdown that charged full price.
  * ``core.baseline`` computed totals at full price only.
  * ``analysis.simulator`` applied the discount to every schedule.

"Savings" was therefore computed by dividing a discounted number by an
undiscounted one, which inflated the result by 5-16 percentage points
depending on the instance. Any comparison between two schedules is only
meaningful when both are priced by the same function — this one.

THE COST MODEL (mirrors ``core.solver``'s objective exactly)
------------------------------------------------------------
For every scheduled maintenance task on machine m of duration d:

    PM cost                       c_pm_m
    standalone production loss    v_m * d                  (m in no chain)

For every production chain c (per-period chain-down indicators):

    chain production loss         V_c * |{t : some machine of c is under PM at t}|
    chain retooling               R_c * (number of maximal down-runs of c)

A chain is down in a period if at least one of its machines is under
maintenance then, and it pays for that period ONCE however many of its
machines share the outage. Retooling is paid once per restart, i.e. once
per maximal run of consecutive down periods. This is exact for any number
of overlapping machines. The model it replaced granted a flat 50% credit
to any task overlapping a chain-mate: right by coincidence for two
machines (two halves make one outage), wrong for three (1.5 outages
charged for one).

For every gap g between maintenance events on machine m (including the
gap from t=0 to the first PM and from the last PM to the horizon), with
virtual age a at the start of the gap:

    expected failure cost   c_cm_m * ( ((a + g) / eta_m)^beta_m - (a / eta_m)^beta_m )

c_cm is ALL-IN — repair labour, expedited parts and the production lost
during the unplanned outage — so no separate downtime term exists for a
failure anywhere in the code base. With perfect PM the virtual age
is 0 after every PM and the term is the familiar c_cm (g/eta)^beta. With
``repair_factor`` r < 1 (Kijima type I) the virtual age after a PM is
round((1 - r) * age before it); see ``MachineSpec.virtual_age_after_pm``.

BASES
-----
``chain_grouping=True``  the model above: a shared chain outage is paid
once. This is the solver's objective.

``chain_grouping=False`` charges every chain event in full, V_c * d + R_c,
as though no two machines of a chain were ever down together. It is the
"no coordination credit" basis. Report it alongside the first so a reader
can see how much of a measured advantage comes from sharing outages rather
than from anything else. Both bases are legitimate; mixing them is not.

PER-TASK ATTRIBUTION
--------------------
Under the shared-outage model a chain's cost belongs to the chain's
schedule, not to any one task. ``attribute_chain_costs`` splits it back
onto tasks so that per-task fields (``MaintenanceTask.cost_prod_loss``,
``cost_retooling``) still sum to the totals: each down period's V_c is
shared equally by the tasks covering it, and each run's R_c is shared
equally by the tasks in that run.

NOTE ON PLACEMENT: this lives in ``core`` rather than ``analysis`` so the
dependency arrow stays ``core -> core`` (``analysis`` already imports from
``core``; the reverse would couple the two packages in both directions).
"""

from dataclasses import dataclass, field

from core.instance import ProblemInstance, ProductionChain

Schedule = dict[int, list[int]]
TaskKey = tuple[int, int]  # (machine_id, start_time)


@dataclass
class CostBreakdown:
    """What a schedule costs, decomposed. ``total`` is always the sum."""

    pm_cost: float = 0.0
    production_loss: float = 0.0
    retooling_cost: float = 0.0
    failure_cost: float = 0.0
    num_tasks: int = 0
    num_grouped_tasks: int = 0
    chain_grouping: bool = True
    per_machine: dict[int, float] = field(default_factory=dict)
    chain_costs: dict[int, dict[str, float]] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (self.pm_cost + self.production_loss
                + self.retooling_cost + self.failure_cost)

    @property
    def deterministic_part(self) -> float:
        """Everything except the expected failure cost: what a realisation
        of the schedule pays for certain before any breakdown happens."""
        return self.pm_cost + self.production_loss + self.retooling_cost

    def as_dict(self) -> dict:
        return {
            "total_cost": round(self.total, 2),
            "pm_cost": round(self.pm_cost, 2),
            "prod_loss": round(self.production_loss, 2),
            "retooling_cost": round(self.retooling_cost, 2),
            "failure_cost": round(self.failure_cost, 2),
            "num_tasks": self.num_tasks,
            "num_grouped_tasks": self.num_grouped_tasks,
            "basis": "grouped" if self.chain_grouping else "full",
        }

    def summary(self) -> str:
        basis = ("shared chain outages paid once" if self.chain_grouping
                 else "every chain event at full price")
        return (
            f"${self.total:,.2f} total  ({basis})\n"
            f"   PM               ${self.pm_cost:>12,.2f}\n"
            f"   Production loss  ${self.production_loss:>12,.2f}\n"
            f"   Retooling        ${self.retooling_cost:>12,.2f}\n"
            f"   Expected failure ${self.failure_cost:>12,.2f}\n"
            f"   Tasks: {self.num_tasks} ({self.num_grouped_tasks} overlap a chain-mate)"
        )


def overlaps_chain_mate(
    instance: ProblemInstance,
    machine_id: int,
    start: int,
    schedule: Schedule,
) -> bool:
    """True when this task's window intersects a chain-mate's window.

    Windows are half-open [start, start + duration), matching the solver.
    Used for reporting only (``num_grouped_tasks``); the price of an
    overlap is whatever the shared-outage model says it is.
    """
    chain = instance.get_chain_for_machine(machine_id)
    if not chain:
        return False

    end = start + instance.machines[machine_id].maintenance_duration
    for mate_id in chain.machine_ids:
        if mate_id == machine_id:
            continue
        mate_d = instance.machines[mate_id].maintenance_duration
        for mate_start in schedule.get(mate_id, []):
            if start < mate_start + mate_d and mate_start < end:
                return True
    return False


def chain_down_profile(
    instance: ProblemInstance,
    chain: ProductionChain,
    schedule: Schedule,
) -> list[list[TaskKey]]:
    """For each period t, the chain's tasks under way then: ``cover[t]``.

    The chain is down in period t exactly when ``cover[t]`` is non-empty.
    Windows are clipped to [0, horizon).
    """
    H = instance.horizon
    cover: list[list[TaskKey]] = [[] for _ in range(H)]
    for mid in chain.machine_ids:
        d = instance.machines[mid].maintenance_duration
        for s in schedule.get(mid, []):
            for t in range(max(0, s), min(H, s + d)):
                cover[t].append((mid, s))
    return cover


def chain_down_runs(cover: list[list[TaskKey]]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive down periods, as half-open [a, b)."""
    runs = []
    t, H = 0, len(cover)
    while t < H:
        if cover[t]:
            a = t
            while t < H and cover[t]:
                t += 1
            runs.append((a, t))
        else:
            t += 1
    return runs


def _price_chain(
    instance: ProblemInstance,
    chain: ProductionChain,
    schedule: Schedule,
    chain_grouping: bool,
) -> tuple[dict[str, float], dict[TaskKey, tuple[float, float]]]:
    """Price one chain and attribute the result to its tasks.

    Returns ``(stats, per_task)`` where ``stats`` carries the chain's
    production loss, retooling, event/run counts, and ``per_task`` maps
    ``(machine_id, start)`` to ``(production_loss, retooling)`` shares that
    sum exactly to the chain totals.
    """
    V, R = chain.chain_value, chain.retooling_cost
    per_task: dict[TaskKey, list[float]] = {}
    for mid in chain.machine_ids:
        for s in schedule.get(mid, []):
            per_task[(mid, s)] = [0.0, 0.0]

    cover = chain_down_profile(instance, chain, schedule)
    runs = chain_down_runs(cover)
    down_periods = sum(1 for c in cover if c)

    if chain_grouping:
        # One outage per period, shared equally by whoever is down in it.
        for c in cover:
            if c:
                share = V / len(c)
                for key in c:
                    per_task[key][0] += share
        # One retooling per restart, shared equally by the run's members.
        for a, b in runs:
            members = {key for t in range(a, b) for key in cover[t]}
            share = R / len(members)
            for key in members:
                per_task[key][1] += share
        prod_loss = float(V * down_periods)
        retooling = float(R * len(runs))
    else:
        # Full basis: every event pays as if it were alone.
        for (mid, _s), acc in per_task.items():
            acc[0] = float(V * instance.machines[mid].maintenance_duration)
            acc[1] = float(R)
        prod_loss = sum(acc[0] for acc in per_task.values())
        retooling = sum(acc[1] for acc in per_task.values())

    stats = {
        "prod_loss": prod_loss,
        "retooling": retooling,
        "num_events": len(per_task),
        "num_grouped": sum(
            1 for (mid, s) in per_task
            if overlaps_chain_mate(instance, mid, s, schedule)),
        "down_periods": down_periods,
        "num_runs": len(runs),
    }
    return stats, {k: (v[0], v[1]) for k, v in per_task.items()}


def attribute_chain_costs(
    instance: ProblemInstance,
    schedule: Schedule,
    *,
    chain_grouping: bool = True,
) -> dict[TaskKey, tuple[float, float]]:
    """Per-task (production_loss, retooling) shares for every chain task.

    Keyed by ``(machine_id, start_time)``. Summing the shares over a chain's
    tasks gives that chain's totals from ``deterministic_cost`` exactly.
    Standalone machines are not included (their production loss is a
    property of the task alone: v_m * d).
    """
    out: dict[TaskKey, tuple[float, float]] = {}
    for chain in instance.chains:
        _stats, per_task = _price_chain(instance, chain, schedule, chain_grouping)
        out.update(per_task)
    return out


def apply_task_attribution(instance: ProblemInstance, tasks, schedule: Schedule,
                           *, chain_grouping: bool = True) -> None:
    """Restate ``cost_prod_loss`` / ``cost_retooling`` on task objects.

    ``tasks`` are ``core.solver.MaintenanceTask``-like objects (anything with
    ``machine_id``, ``start_time`` and ``chain_id``). Standalone tasks are
    left alone. Afterwards the per-task fields sum to the chain totals of
    ``deterministic_cost(instance, schedule)`` on the same basis.
    """
    shares = attribute_chain_costs(instance, schedule, chain_grouping=chain_grouping)
    for t in tasks:
        if t.chain_id is None:
            continue
        prod, retool = shares.get((t.machine_id, t.start_time), (0.0, 0.0))
        t.cost_prod_loss = prod
        t.cost_retooling = retool


def expected_failure_cost_of_machine(machine, starts: list[int], horizon: int) -> float:
    """Expected breakdown cost of one machine over the horizon.

    Gaps run from t=0 to the first PM, between PMs, and from the last PM to
    the horizon. The virtual age is 0 at t=0, advances through each gap,
    and is reset by each PM according to ``machine.virtual_age_after_pm``
    (to 0 for perfect repair). Corrective repair is minimal and does not
    appear here at all: only PMs move the virtual age.
    """
    d = machine.maintenance_duration
    virtual_age = 0
    prev_end = 0
    cost = 0.0
    for start in sorted(starts):
        gap = start - prev_end
        cost += machine.expected_failures_imperfect(gap, virtual_age) * machine.cm_cost
        virtual_age = machine.virtual_age_after_pm(virtual_age + max(0, gap))
        prev_end = start + d
    gap = horizon - prev_end
    cost += machine.expected_failures_imperfect(gap, virtual_age) * machine.cm_cost
    return cost


def deterministic_cost(
    instance: ProblemInstance,
    schedule: Schedule,
    *,
    chain_grouping: bool = True,
) -> CostBreakdown:
    """Price any maintenance schedule under one consistent accounting.

    Args:
        instance: the problem instance the schedule was built for.
        schedule: ``{machine_id: [start_times]}``. Machines absent from the
            dict are treated as receiving no maintenance at all.
        chain_grouping: pay a shared chain outage once (the solver's
            objective). Set False to charge every chain event in full.

    Returns:
        A CostBreakdown whose ``total`` equals the CP-SAT objective for a
        schedule the solver produced, up to integer-scaling rounding of
        the failure-cost tables.
    """
    bd = CostBreakdown(chain_grouping=chain_grouping)
    H = instance.horizon

    # ── Per-machine: PM cost, standalone production loss, failures ──
    for m_idx, machine in enumerate(instance.machines):
        d = machine.maintenance_duration
        chain = instance.get_chain_for_machine(m_idx)
        starts = sorted(schedule.get(m_idx, []))
        machine_cost = 0.0

        for start in starts:
            bd.num_tasks += 1
            bd.pm_cost += machine.pm_cost
            machine_cost += machine.pm_cost
            if chain is None:
                loss = machine.production_value * d
                bd.production_loss += loss
                machine_cost += loss
            elif overlaps_chain_mate(instance, m_idx, start, schedule):
                bd.num_grouped_tasks += 1

        fc = expected_failure_cost_of_machine(machine, starts, H)
        bd.failure_cost += fc
        machine_cost += fc
        bd.per_machine[m_idx] = machine_cost

    # ── Per-chain: shared outages and restarts ─────────────────────
    for chain in instance.chains:
        stats, per_task = _price_chain(instance, chain, schedule, chain_grouping)
        bd.production_loss += stats["prod_loss"]
        bd.retooling_cost += stats["retooling"]
        bd.chain_costs[chain.id] = stats
        for (mid, _s), (prod, retool) in per_task.items():
            bd.per_machine[mid] = bd.per_machine.get(mid, 0.0) + prod + retool

    return bd


def savings_vs(
    instance: ProblemInstance,
    schedule: Schedule,
    reference: Schedule,
    *,
    chain_grouping: bool = True,
) -> float:
    """Percentage cost reduction of ``schedule`` against ``reference``.

    Both schedules are priced by the same function on the same basis, which
    is the entire point. Returns 0.0 when the reference costs nothing.
    """
    ref = deterministic_cost(instance, reference,
                             chain_grouping=chain_grouping).total
    if ref <= 0:
        return 0.0
    got = deterministic_cost(instance, schedule,
                             chain_grouping=chain_grouping).total
    return (1.0 - got / ref) * 100.0


def best_of(
    instance: ProblemInstance,
    schedules: dict[str, Schedule],
    *,
    chain_grouping: bool = True,
) -> tuple[str, Schedule, CostBreakdown]:
    """Cheapest of several named schedules, priced consistently."""
    scored = [
        (name, sched, deterministic_cost(instance, sched,
                                         chain_grouping=chain_grouping))
        for name, sched in schedules.items()
    ]
    return min(scored, key=lambda t: t[2].total)
