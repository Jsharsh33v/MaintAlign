"""
MaintAlign - Monte Carlo Simulator
=====================================
Replay a PM schedule against random breakdowns and price what happened.

THE FAILURE PROCESS (matches the objective)
  Failures on a machine form a non-homogeneous Poisson process with the
  Weibull intensity λ(a) = (β/η)(a/η)^(β−1), where a is the machine's
  VIRTUAL AGE. Corrective repair is MINIMAL: it returns the machine to
  service at the same virtual age, so successive failures within a gap are
  drawn from the conditional NHPP and the expected count over a gap of
  length g that starts at virtual age a is ((a + g)/η)^β − (a/η)^β —
  exactly the term the solver minimises. Only PM changes the virtual age:
  to 0 under perfect repair, to round((1 − repair_factor) × age) under
  Kijima type I imperfect repair (``MachineSpec.virtual_age_after_pm``).

  The previous simulator reset the age to 0 after every corrective repair
  (a renewal process), which produced 1.8–3.5× fewer failures than the
  objective assumed, so the Monte Carlo chapter validated a different model
  from the one being optimised.

COST STRUCTURE (one basis)
  Deterministic part (PM, production loss, retooling): priced by
  ``core.costing.deterministic_cost``, the same function that prices the
  solver's objective and the baselines. A shared chain outage is paid once
  per period and retooling once per restart; per-event costs in the event
  log are the ``core.costing`` attribution of those totals.

  Failure cost: c_cm per failure, and NOTHING ELSE. c_cm is all-in (repair
  labour, expedited parts and the production lost during the unplanned
  outage), which is what the objective assumes and what
  ``utils.calibration.corrective_maintenance_cost`` constructs. The
  previous simulator charged c_cm and then added chain_value × cm_duration
  on top, double-counting the downtime under that definition.

  Consequently the mean simulated total cost converges to
  ``deterministic_cost(instance, schedule).total`` — the Monte Carlo run
  validates the objective's expectation and adds the distribution around it.

  ``cm_duration_multiplier`` therefore affects neither cost nor the failure
  clock; it only feeds the ``total_downtime`` statistic.

RANDOM STREAMS
  Each machine draws from its own ``random.Random`` seeded from
  (seed, machine_id), so two schedules simulated with the same seed see the
  same failure draws on every machine whose schedule is unchanged. That is
  what makes ``analysis.evaluator``'s same-seed comparison a common-random-
  numbers comparison rather than two independent estimates.
"""

import logging
import math
import random
from dataclasses import dataclass, field

from core.costing import attribute_chain_costs, deterministic_cost
from core.instance import MachineSpec, ProblemInstance

logger = logging.getLogger(__name__)


@dataclass
class SimulationEvent:
    """One event during simulation."""
    time: float
    machine_id: int
    event_type: str       # 'pm' | 'failure'
    cost: float = 0.0
    downtime: int = 0
    chain_loss: float = 0.0


@dataclass
class SimulationResult:
    """Result of one simulation run."""
    total_pm_cost: float = 0.0
    total_cm_cost: float = 0.0
    total_production_loss: float = 0.0
    total_retooling_cost: float = 0.0
    total_downtime: int = 0
    total_cost: float = 0.0
    num_failures: int = 0
    events: list[SimulationEvent] = field(default_factory=list)


def machine_rng(seed: int | None, machine_id: int) -> random.Random:
    """One independent, reproducible stream per machine (common random numbers)."""
    if seed is None:
        return random.Random()
    return random.Random(f"maintalign/{seed}/{machine_id}")


def next_failure_age(beta: float, eta: float, age: float, rng: random.Random) -> float:
    """Virtual age at the next failure, given none since virtual age ``age``.

    Inverts the conditional NHPP survival
        P(no failure in (a, a + x]) = exp(−(Λ(a + x) − Λ(a))),  Λ(t) = (t/η)^β
    with U ~ Uniform(0, 1):  a_next = η · ((a/η)^β − ln U)^(1/β).
    """
    u = rng.random()
    while u <= 0.0:
        u = rng.random()
    cumulative = (age / eta) ** beta - math.log(u)
    return eta * cumulative ** (1.0 / beta)


def sample_failures_in_gap(machine: MachineSpec, gap_start: float, gap_length: float,
                           virtual_age: float, rng: random.Random) -> list[float]:
    """Absolute failure times in [gap_start, gap_start + gap_length).

    The virtual-age clock runs from ``virtual_age`` at ``gap_start``;
    corrective repair is minimal, so the clock is never reset inside the gap.
    """
    if gap_length <= 0:
        return []
    end_age = virtual_age + gap_length
    times = []
    age = float(virtual_age)
    while True:
        age = next_failure_age(machine.weibull_beta, machine.weibull_eta, age, rng)
        if age >= end_age:
            return times
        times.append(gap_start + (age - virtual_age))


def simulate_schedule(
    instance: ProblemInstance,
    schedule: dict[int, list[int]],
    seed: int | None = None,
    cm_duration_multiplier: float = 2.0,
) -> SimulationResult:
    """
    Simulate one realization of random failures against a PM schedule.

    Args:
        instance: problem instance
        schedule: {machine_id: [start_times]} — the PM schedule
        seed: base seed; each machine gets its own stream derived from it
        cm_duration_multiplier: corrective repair takes this × PM duration.
            Feeds ``total_downtime`` only — the cost of that downtime is
            inside ``cm_cost``, and the failure clock does not stop for it.

    Returns:
        SimulationResult with costs, failures, and event log
    """
    H = instance.horizon
    result = SimulationResult()

    # ── Deterministic part: exactly what the objective charges ──────
    det = deterministic_cost(instance, schedule)
    shares = attribute_chain_costs(instance, schedule)
    result.total_pm_cost = det.pm_cost
    result.total_production_loss = det.production_loss
    result.total_retooling_cost = det.retooling_cost

    for m_idx, machine in enumerate(instance.machines):
        pm_starts = sorted(schedule.get(m_idx, []))
        d = machine.maintenance_duration
        cm_duration = int(d * cm_duration_multiplier)
        chain = instance.get_chain_for_machine(m_idx)

        for s in pm_starts:
            if chain:
                prod_loss, retooling = shares.get((m_idx, s), (0.0, 0.0))
            else:
                prod_loss, retooling = machine.production_value * d, 0.0
            result.events.append(SimulationEvent(
                time=s, machine_id=m_idx, event_type='pm',
                cost=machine.pm_cost + prod_loss + retooling, downtime=d,
                chain_loss=prod_loss if chain else 0.0,
            ))

        # ── Random part: minimal-repair failures over every gap ────
        rng = machine_rng(seed, m_idx)
        virtual_age = 0
        prev_end = 0
        gaps = []
        for s in pm_starts:
            gaps.append((prev_end, s - prev_end, virtual_age))
            virtual_age = machine.virtual_age_after_pm(virtual_age + max(0, s - prev_end))
            prev_end = s + d
        gaps.append((prev_end, H - prev_end, virtual_age))

        for gap_start, gap_length, va in gaps:
            for fail_time in sample_failures_in_gap(machine, gap_start, gap_length, va, rng):
                result.total_cm_cost += machine.cm_cost
                result.total_downtime += cm_duration
                result.num_failures += 1
                result.events.append(SimulationEvent(
                    time=fail_time, machine_id=m_idx, event_type='failure',
                    cost=machine.cm_cost, downtime=cm_duration, chain_loss=0.0,
                ))

    result.events.sort(key=lambda e: (e.time, e.machine_id))
    result.total_cost = (result.total_pm_cost + result.total_cm_cost
                         + result.total_production_loss
                         + result.total_retooling_cost)
    return result
