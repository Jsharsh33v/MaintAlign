"""
MaintAlign - Monte Carlo Simulator
=====================================
Simulate random machine failures against a given PM schedule.

Uses Weibull failure model to generate random breakdown events between
PM windows and calculates actual costs including emergency repairs,
chain cascade downtime, and production losses.

COST STRUCTURE (matches solver):
  Standalone machines:
    PM event cost = pm_cost + production_value × duration

  Chain machines:
    PM event cost = pm_cost
      + if grouped (overlapping chain-mate PM): 50% retooling + 50% chain_value × d
      + if isolated (no overlap):              full retooling + full chain_value × d

  Failure cost (all machines):
    CM event cost = cm_cost + production_loss × cm_duration
"""

import logging
import math
import random
from dataclasses import dataclass, field

from core.instance import ProblemInstance

logger = logging.getLogger(__name__)


@dataclass
class SimulationEvent:
    """One event during simulation."""
    time: float
    machine_id: int
    event_type: str       # 'pm', 'failure', 'cm_repair'
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


def _sample_weibull_failure(beta: float, eta: float, age: float) -> float:
    """
    Sample time to next failure from a Weibull distribution.

    Uses the conditional Weibull: given that a machine has survived to 'age',
    sample when it will next fail.

    T | T > age ~ age + Weibull_residual

    For NHPP with rate λ(t) = (β/η)(t/η)^(β-1), we use thinning.
    Simplified: sample from Weibull(β, η) and reject if < age.
    """
    # Direct sampling using inverse CDF method
    # For a Weibull with shape β and scale η:
    # T = η × (-ln(U))^(1/β)
    # We need T > age, so we sample conditionally
    if beta <= 0 or eta <= 0:
        return float('inf')

    # Survival function at age: S(age) = exp(-(age/η)^β)
    surv_age = math.exp(-((age / eta) ** beta))
    if surv_age <= 0:
        # Already effectively failed
        return age + 0.001

    # Sample U ~ Uniform(0, S(age)) to get conditional failure time
    u = random.random() * surv_age
    if u <= 0:
        u = 1e-15

    # Inverse CDF: T = η × (-ln(U))^(1/β)
    t = eta * ((-math.log(u)) ** (1.0 / beta))
    return t


def _detect_chain_overlap(
    m_idx: int,
    pm_start: int,
    pm_end: int,
    instance: ProblemInstance,
    schedule: dict[int, list[int]],
) -> bool:
    """
    Check if a PM event on machine m_idx overlaps with any chain-mate PM.

    Overlap means the time windows [pm_start, pm_end) and [s2, s2+d2) intersect.
    This mirrors the solver's opportunistic grouping logic.
    """
    chain = instance.get_chain_for_machine(m_idx)
    if not chain:
        return False

    for mate_id in chain.machine_ids:
        if mate_id == m_idx:
            continue
        mate_d = instance.machines[mate_id].maintenance_duration
        for s2 in schedule.get(mate_id, []):
            # Overlap: pm_start < s2 + mate_d  AND  s2 < pm_end
            if pm_start < s2 + mate_d and s2 < pm_end:
                return True

    return False


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
        seed: random seed for reproducibility
        cm_duration_multiplier: CM takes this × PM duration

    Returns:
        SimulationResult with costs, failures, and event log
    """
    if seed is not None:
        random.seed(seed)

    H = instance.horizon
    result = SimulationResult()

    for m_idx, machine in enumerate(instance.machines):
        pm_starts = sorted(schedule.get(m_idx, []))
        d = machine.maintenance_duration
        cm_duration = int(d * cm_duration_multiplier)
        chain = instance.get_chain_for_machine(m_idx)

        # Build PM windows: [(start, end), ...]
        pm_windows = [(s, s + d) for s in pm_starts]

        # ─── PM costs (matches solver cost structure) ───────────
        for s in pm_starts:
            pm_end = s + d

            # Base PM cost (always charged)
            pm_cost = machine.pm_cost

            # Production loss + retooling depends on chain vs standalone
            prod_loss = 0.0
            retooling = 0.0

            if chain:
                # Chain machine: check for opportunistic grouping
                is_grouped = _detect_chain_overlap(
                    m_idx, s, pm_end, instance, schedule
                )
                if is_grouped:
                    # 50% discount on retooling + production loss
                    prod_loss = chain.chain_value * d * 0.5
                    retooling = chain.retooling_cost * 0.5
                else:
                    # Isolated: full cost
                    prod_loss = chain.chain_value * d
                    retooling = chain.retooling_cost
            else:
                # Standalone: production loss = production_value × duration
                prod_loss = machine.production_value * d

            result.total_pm_cost += pm_cost
            result.total_production_loss += prod_loss
            result.total_retooling_cost += retooling

            total_event_cost = pm_cost + prod_loss + retooling
            result.events.append(SimulationEvent(
                time=s, machine_id=m_idx, event_type='pm',
                cost=total_event_cost, downtime=d,
                chain_loss=prod_loss if chain else 0.0
            ))

        # ─── Simulate failures between PMs ──────────────────────
        # Machine age resets to 0 after each PM
        age = 0.0
        current_time = 0.0

        for pm_start, pm_end in pm_windows:
            # Simulate failures from current_time to pm_start
            while current_time < pm_start:
                # Sample next failure time
                fail_time = _sample_weibull_failure(
                    machine.weibull_beta, machine.weibull_eta, age
                )
                absolute_fail_time = current_time + (fail_time - age)

                if absolute_fail_time < pm_start:
                    # Failure occurs before PM!
                    cm_cost = machine.cm_cost
                    if chain:
                        chain_loss = chain.chain_value * cm_duration
                    else:
                        chain_loss = machine.production_value * cm_duration

                    result.total_cm_cost += cm_cost
                    result.total_production_loss += chain_loss
                    result.total_downtime += cm_duration
                    result.num_failures += 1
                    result.events.append(SimulationEvent(
                        time=absolute_fail_time, machine_id=m_idx,
                        event_type='failure',
                        cost=cm_cost, downtime=cm_duration,
                        chain_loss=chain_loss
                    ))

                    # After CM repair, machine age resets
                    current_time = absolute_fail_time + cm_duration
                    age = 0.0

                    if current_time >= pm_start:
                        break
                else:
                    # No failure before PM, advance to PM
                    break

            # PM happens → age resets
            current_time = pm_end
            age = 0.0

        # Simulate failures from last PM to end of horizon
        while current_time < H:
            fail_time = _sample_weibull_failure(
                machine.weibull_beta, machine.weibull_eta, age
            )
            absolute_fail_time = current_time + (fail_time - age)

            if absolute_fail_time < H:
                cm_cost = machine.cm_cost
                if chain:
                    chain_loss = chain.chain_value * cm_duration
                else:
                    chain_loss = machine.production_value * cm_duration

                result.total_cm_cost += cm_cost
                result.total_production_loss += chain_loss
                result.total_downtime += cm_duration
                result.num_failures += 1
                result.events.append(SimulationEvent(
                    time=absolute_fail_time, machine_id=m_idx,
                    event_type='failure', cost=cm_cost,
                    downtime=cm_duration, chain_loss=chain_loss
                ))
                current_time = absolute_fail_time + cm_duration
                age = 0.0
            else:
                break

    result.total_cost = (result.total_pm_cost + result.total_cm_cost
                         + result.total_production_loss
                         + result.total_retooling_cost)
    return result
