"""
MaintAlign - Schedule Evaluator
=================================
Run N Monte Carlo simulations and compute statistics.

Provides:
  - Mean, std, percentiles of total cost
  - VaR95 (Value at Risk: average of worst 5% outcomes)
  - Fair comparison of multiple schedules using same random seeds
"""

import logging
import statistics
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from core.instance import ProblemInstance
from analysis.simulator import simulate_schedule, SimulationResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Statistical summary of N simulation runs."""
    schedule_name: str
    n_sims: int
    mean_cost: float = 0.0
    std_cost: float = 0.0
    median_cost: float = 0.0
    p5_cost: float = 0.0       # 5th percentile (best case)
    p95_cost: float = 0.0      # 95th percentile (worst case)
    var95: float = 0.0         # Average of worst 5%
    mean_failures: float = 0.0
    mean_downtime: float = 0.0
    mean_pm_cost: float = 0.0
    mean_cm_cost: float = 0.0
    mean_prod_loss: float = 0.0
    mean_retooling_cost: float = 0.0
    all_costs: List[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"  {self.schedule_name:20s} | "
            f"Mean: ${self.mean_cost:,.0f} ± ${self.std_cost:,.0f} | "
            f"Median: ${self.median_cost:,.0f} | "
            f"VaR95: ${self.var95:,.0f} | "
            f"Failures: {self.mean_failures:.1f}"
        )


def evaluate_schedule(
    instance: ProblemInstance,
    schedule: Dict[int, List[int]],
    schedule_name: str = "schedule",
    n_sims: int = 1000,
    base_seed: int = 12345,
) -> EvaluationResult:
    """
    Run N simulations of a schedule and compute statistics.

    Args:
        instance: problem instance
        schedule: {machine_id: [start_times]}
        schedule_name: label for this schedule
        n_sims: number of simulation runs
        base_seed: base seed (each sim uses base_seed + i)

    Returns:
        EvaluationResult with statistics
    """
    costs = []
    total_failures = 0
    total_downtime = 0
    total_pm = 0.0
    total_cm = 0.0
    total_prod_loss = 0.0
    total_retooling = 0.0

    for i in range(n_sims):
        sim = simulate_schedule(instance, schedule, seed=base_seed + i)
        costs.append(sim.total_cost)
        total_failures += sim.num_failures
        total_downtime += sim.total_downtime
        total_pm += sim.total_pm_cost
        total_cm += sim.total_cm_cost
        total_prod_loss += sim.total_production_loss
        total_retooling += sim.total_retooling_cost

    costs.sort()
    n = len(costs)

    # VaR95: average of worst 5%
    worst_5pct = costs[int(n * 0.95):]
    var95 = statistics.mean(worst_5pct) if worst_5pct else costs[-1]

    result = EvaluationResult(
        schedule_name=schedule_name,
        n_sims=n_sims,
        mean_cost=statistics.mean(costs),
        std_cost=statistics.stdev(costs) if n > 1 else 0.0,
        median_cost=statistics.median(costs),
        p5_cost=costs[max(0, int(n * 0.05))],
        p95_cost=costs[min(n - 1, int(n * 0.95))],
        var95=var95,
        mean_failures=total_failures / n,
        mean_downtime=total_downtime / n,
        mean_pm_cost=total_pm / n,
        mean_cm_cost=total_cm / n,
        mean_prod_loss=total_prod_loss / n,
        mean_retooling_cost=total_retooling / n,
        all_costs=costs,
    )

    logger.info("Evaluated '%s': mean=$%.0f, std=$%.0f, VaR95=$%.0f, "
                "avg_failures=%.1f",
                schedule_name, result.mean_cost, result.std_cost,
                result.var95, result.mean_failures)
    return result


def compare_schedules(
    instance: ProblemInstance,
    schedules: Dict[str, Dict[int, List[int]]],
    n_sims: int = 1000,
    base_seed: int = 12345,
) -> Dict[str, EvaluationResult]:
    """
    Compare multiple schedules using the same random seeds for fairness.

    Args:
        instance: problem instance
        schedules: {name: {machine_id: [start_times]}}
        n_sims: number of simulations per schedule
        base_seed: base random seed

    Returns:
        {name: EvaluationResult}
    """
    results = {}
    for name, schedule in schedules.items():
        results[name] = evaluate_schedule(
            instance, schedule, name, n_sims, base_seed
        )

    # Print comparison
    logger.info("─── Monte Carlo Comparison (%d sims) ───", n_sims)
    for name, r in results.items():
        logger.info(r.summary())

    return results
