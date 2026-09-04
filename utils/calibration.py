"""
MaintAlign - Parameter Calibration
====================================

Every cost in MaintAlign used to be drawn from an interval chosen by hand:
``pm_cost ~ U(100, 500)``, ``cm_cost = pm_cost * U(3, 10)``. Those numbers
are indefensible in an economics comp, and the CM/PM ratio in particular is
the first thing a reader will question, because it is the parameter the
whole result turns on.

This module replaces "assume a ratio" with "construct the cost, then report
the ratio it implies". Each parameter carries a tier saying how much weight
it can bear:

  empirical    fitted from observed failure or operating data
  cited        taken from a published source, quoted with the source
  constructed  computed from other parameters by a stated identity
  assumed      chosen by us; MUST appear in the sensitivity analysis

The point is not that every number becomes real. It is that no number is
unaccounted for, and that the ones we chose are visible as choices. That is
what a calibration section is, and it is a recognised empirical method --
not a substitute for one.

FILL IN THE TODOs. A parameter marked ``tier="assumed"`` with a TODO source
is a promise to either find a citation or defend the sweep.
"""

from dataclasses import dataclass

from core.instance import MachineSpec, ProblemInstance, ProductionChain

# ═══════════════════════════════════════════════════════════════════
#  Parameter records
# ═══════════════════════════════════════════════════════════════════

TIERS = ("empirical", "cited", "constructed", "assumed")


@dataclass(frozen=True)
class Parameter:
    """One calibrated quantity, with where it came from and how far it moves."""

    name: str
    symbol: str
    value: float
    units: str
    tier: str
    source: str
    low: float
    high: float
    url: str = ""
    note: str = ""

    def __post_init__(self):
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}, expected one of {TIERS}")
        if not (self.low <= self.value <= self.high):
            raise ValueError(
                f"{self.name}: value {self.value} outside sensitivity "
                f"range [{self.low}, {self.high}]"
            )

    @property
    def is_sourced(self) -> bool:
        return self.tier in ("empirical", "cited") and not self.source.startswith("TODO")


# ═══════════════════════════════════════════════════════════════════
#  The calibration
# ═══════════════════════════════════════════════════════════════════

TECHNICIAN_WAGE = Parameter(
    name="Maintenance technician wage",
    symbol="w",
    value=30.82,
    units="USD per hour",
    tier="cited",
    source=("U.S. Bureau of Labor Statistics, Occupational Outlook Handbook, "
            "'Industrial Machinery Mechanics, Machinery Maintenance Workers, "
            "and Millwrights', median hourly wage, May 2025 "
            "($64,100/yr; millwrights $65,700, industrial machinery mechanics "
            "$64,520, machinery maintenance workers $60,850)"),
    url=("https://www.bls.gov/ooh/installation-maintenance-and-repair/"
         "industrial-machinery-mechanics-and-maintenance-workers-and-millwrights.htm"),
    low=24.00,
    high=40.00,
    note=("Range spans the 10th-90th percentile band across the three "
          "occupations. Fully-loaded cost to the employer is higher than the "
          "wage; see LOADED_LABOR_MULTIPLIER."),
)

LOADED_LABOR_MULTIPLIER = Parameter(
    name="Loaded labour multiplier",
    symbol="lambda",
    value=1.40,
    units="ratio of employer cost to wage",
    tier="assumed",
    source="TODO — cite BLS Employer Costs for Employee Compensation (ECEC).",
    low=1.25,
    high=1.60,
    note=("Benefits, payroll tax and overhead on top of the base wage. ECEC "
          "publishes benefits as a share of total compensation; convert and "
          "cite rather than leaving this assumed."),
)

HOURS_PER_PERIOD = Parameter(
    name="Working hours per scheduling period",
    symbol="h",
    value=8.0,
    units="hours",
    tier="assumed",
    source="Modelling choice: one period = one maintenance shift.",
    low=8.0,
    high=24.0,
    note=("Sets the exchange rate between the model's integer periods and "
          "wall-clock cost. State it explicitly in the thesis; every dollar "
          "figure scales with it."),
)

EMERGENCY_LABOR_PREMIUM = Parameter(
    name="Emergency labour premium",
    symbol="rho_L",
    value=1.50,
    units="multiplier on planned labour rate",
    tier="assumed",
    source="TODO — overtime/call-out rate; FLSA overtime is 1.5x as a floor.",
    low=1.25,
    high=2.50,
    note="Unplanned work is done at premium rates and outside shift patterns.",
)

EXPEDITE_PARTS_PREMIUM = Parameter(
    name="Expedited parts premium",
    symbol="rho_P",
    value=1.35,
    units="multiplier on planned parts cost",
    tier="assumed",
    source="TODO — cite a supply-chain or maintenance-economics source.",
    low=1.00,
    high=2.00,
    note="Freight and spot-purchase premium when a part is needed unplanned.",
)

CM_DURATION_MULTIPLIER = Parameter(
    name="Corrective repair duration multiplier",
    symbol="mu",
    value=2.0,
    units="multiplier on planned maintenance duration",
    tier="assumed",
    source=("Enters c_cm through corrective_maintenance_cost (longer premium "
            "labour, longer lost margin); analysis.simulator uses the same "
            "default for its downtime statistic only. TODO cite or estimate."),
    low=1.5,
    high=4.0,
    note=("An unplanned failure takes longer than the planned job: diagnosis, "
          "no staged parts, no prepared shutdown."),
)

DISCOUNT_RATE_ANNUAL = Parameter(
    name="Annual discount rate",
    symbol="r",
    value=0.08,
    units="per year",
    tier="assumed",
    source="TODO — firm's WACC or a cited industry cost of capital.",
    low=0.03,
    high=0.15,
    note=("Not yet used: the objective is undiscounted. Adding delta^t to the "
          "cost terms yields a signed comparative static (higher r defers "
          "maintenance) that an economics reader will look for."),
)

ALL_PARAMETERS = [
    TECHNICIAN_WAGE,
    LOADED_LABOR_MULTIPLIER,
    HOURS_PER_PERIOD,
    EMERGENCY_LABOR_PREMIUM,
    EXPEDITE_PARTS_PREMIUM,
    CM_DURATION_MULTIPLIER,
    DISCOUNT_RATE_ANNUAL,
]


# ═══════════════════════════════════════════════════════════════════
#  Cost construction
# ═══════════════════════════════════════════════════════════════════

def labor_cost_per_technician_period(
    wage: float = TECHNICIAN_WAGE.value,
    loaded: float = LOADED_LABOR_MULTIPLIER.value,
    hours: float = HOURS_PER_PERIOD.value,
) -> float:
    """Employer cost of one technician for one scheduling period.

    This is also the opportunity cost benchmark for the shadow-price
    experiment: an extra technician is worth hiring only if the marginal
    reduction in expected cost exceeds this, times the horizon.
    """
    return wage * loaded * hours


def planned_maintenance_cost(
    duration_periods: int,
    parts_cost: float,
    technicians_required: int = 1,
    **kw,
) -> float:
    """c_pm — the cost of doing the job on purpose.

        c_pm = (loaded wage) x hours x technicians x duration + parts
    """
    return (labor_cost_per_technician_period(**kw)
            * technicians_required * duration_periods
            + parts_cost)


def corrective_maintenance_cost(
    duration_periods: int,
    parts_cost: float,
    margin_per_period: float,
    technicians_required: int = 1,
    *,
    labor_premium: float = EMERGENCY_LABOR_PREMIUM.value,
    parts_premium: float = EXPEDITE_PARTS_PREMIUM.value,
    duration_multiplier: float = CM_DURATION_MULTIPLIER.value,
    secondary_damage: float = 0.0,
    **kw,
) -> float:
    """c_cm — the all-in cost of a breakdown, CONSTRUCTED rather than assumed.

        c_cm = premium labour over a longer repair
             + expedited parts
             + lost contribution margin during unplanned downtime
             + collateral damage

    DEFINITION. c_cm is ALL-IN. ``core.solver`` and
    ``core.costing`` charge failures as ``E[N] * c_cm`` and nothing else,
    and ``analysis.simulator`` charges ``c_cm`` per failure and nothing
    else: the production lost during the unplanned outage enters c_cm here,
    through ``margin_per_period * cm_duration``, and nowhere else. (The
    simulator used to add ``chain_value * cm_duration`` on top, which
    double-counted the downtime under this definition.)
    """
    cm_duration = duration_periods * duration_multiplier
    labor = (labor_cost_per_technician_period(**kw)
             * technicians_required * cm_duration * labor_premium)
    parts = parts_cost * parts_premium
    downtime = margin_per_period * cm_duration
    return labor + parts + downtime + secondary_damage


def implied_cost_ratio(c_pm: float, c_cm: float) -> float:
    """CM/PM ratio — now an OUTPUT of the construction, not an input.

    Report the distribution of this across a calibrated factory and compare
    it to the ranges used in the maintenance-optimisation literature. If it
    lands far outside them, the construction is wrong; if it lands inside,
    that is a validation on a moment you did not target.
    """
    return c_cm / c_pm if c_pm > 0 else float("inf")


# ═══════════════════════════════════════════════════════════════════
#  Building a calibrated instance
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MachineEconomics:
    """A machine described in economic terms, not model terms.

    ``margin_per_period`` is CONTRIBUTION MARGIN — price less variable cost
    of the output this machine's line produces in one period — not revenue.
    The opportunity cost of an idle machine is the margin it would have
    earned, and using revenue overstates it by the whole variable cost.
    """

    name: str
    duration_periods: int
    parts_cost: float
    margin_per_period: float
    weibull_beta: float
    weibull_eta: float
    max_interval: int
    min_gap: int = 1
    technicians_required: int = 1
    secondary_damage: float = 0.0
    beta_source: str = "TODO — fit from failure data, censored MLE"


def to_machine_spec(econ: MachineEconomics, machine_id: int) -> MachineSpec:
    """Turn an economic description into the solver's MachineSpec."""
    c_pm = planned_maintenance_cost(
        econ.duration_periods, econ.parts_cost, econ.technicians_required)
    c_cm = corrective_maintenance_cost(
        econ.duration_periods, econ.parts_cost, econ.margin_per_period,
        econ.technicians_required, secondary_damage=econ.secondary_damage)
    return MachineSpec(
        id=machine_id,
        name=econ.name,
        maintenance_duration=econ.duration_periods,
        pm_cost=int(round(c_pm)),
        cm_cost=int(round(c_cm)),
        production_value=int(round(econ.margin_per_period)),
        weibull_beta=econ.weibull_beta,
        weibull_eta=econ.weibull_eta,
        max_interval=econ.max_interval,
        min_gap=econ.min_gap,
    )


def build_calibrated_instance(
    name: str,
    machines: list[MachineEconomics],
    num_technicians: int,
    horizon: int,
    chains: list[ProductionChain] | None = None,
    blocked_periods: list[int] | None = None,
) -> ProblemInstance:
    """Assemble a ProblemInstance whose costs are constructed, not drawn.

    ``utils.generator`` is deliberately left untouched so every committed
    experiment still reproduces. Use this for the calibrated arm and report
    both.
    """
    specs = [to_machine_spec(e, i) for i, e in enumerate(machines)]
    return ProblemInstance(
        name=name,
        num_machines=len(specs),
        num_technicians=num_technicians,
        horizon=horizon,
        machines=specs,
        chains=list(chains or []),
        blocked_periods=list(blocked_periods or []),
    )


# ═══════════════════════════════════════════════════════════════════
#  The table you put in the thesis
# ═══════════════════════════════════════════════════════════════════

def calibration_table_markdown(parameters: list[Parameter] | None = None) -> str:
    """Render the calibration table as markdown, sorted by tier."""
    params = parameters if parameters is not None else ALL_PARAMETERS
    order = {t: i for i, t in enumerate(TIERS)}
    params = sorted(params, key=lambda p: (order[p.tier], p.name))

    lines = [
        "| Parameter | Symbol | Value | Units | Tier | Sensitivity range | Source |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in params:
        src = p.source if not p.url else f"[{p.source}]({p.url})"
        lines.append(
            f"| {p.name} | `{p.symbol}` | {p.value:g} | {p.units} | "
            f"{p.tier} | {p.low:g} – {p.high:g} | {src} |"
        )

    unsourced = [p for p in params if not p.is_sourced]
    lines.append("")
    lines.append(
        f"**{len(params) - len(unsourced)} of {len(params)} parameters are "
        f"sourced.** Every parameter in the `assumed` tier must appear in the "
        f"sensitivity analysis; a result that does not survive its range is "
        f"not a result."
    )
    if unsourced:
        lines.append("")
        lines.append("Still to source:")
        for p in unsourced:
            lines.append(f"- **{p.name}** — {p.source}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(calibration_table_markdown())
    print()
    print(f"One technician-period costs "
          f"${labor_cost_per_technician_period():,.2f}")
