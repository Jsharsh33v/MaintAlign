"""
MaintAlign - Instance Data Model (v2: Chains + Optional Tasks)
================================================================

LEVEL 3+ MODEL: Resource-Constrained PM with Production Chains
---------------------------------------------------------------

This is a 0/1 knapsack-style problem where the solver decides:
  - WHICH machines to maintain and HOW OFTEN (optional tasks)
  - WHEN to schedule each maintenance event (placement)
under limited technician resources and production chain dependencies.

KNAPSACK ANALOGY:
  "Items"    = potential maintenance tasks
  "Value"    = failure risk reduction (Weibull expected cost avoided)
  "Weight"   = PM cost + chain production loss + chain retooling cost
  "Capacity" = technician-hours available per period

═══════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULATION
═══════════════════════════════════════════════════════════════════

SETS:
  M = {0, ..., num_machines-1}       machines
  C = {0, ..., num_chains-1}         production chains
  K                                   number of technicians (pooled)
  T = {0, 1, ..., horizon-1}         time periods
  J_m = {0, ..., max_tasks-1}        potential tasks per machine m

  machines(c) ⊆ M                    machines belonging to chain c
  chain(m) ∈ C ∪ {∅}                 chain that machine m belongs to
  standalone = {m ∈ M : chain(m) = ∅} machines not in any chain

PARAMETERS:
  Per machine m:
    d_m         maintenance duration (periods)
    c_pm_m      planned maintenance cost ($)
    c_cm_m      corrective/breakdown cost ($)
    v_m         standalone production value ($/period, used if m ∉ chain)
    β_m         Weibull shape (>1 = wear-out, bigger = more predictable)
    η_m         Weibull scale (characteristic life in periods)
    W_m         maximum interval between PM events
    g_m         minimum gap between consecutive PMs on same machine

  Per chain c:
    V_c         chain production value ($/period when ALL machines running)
    R_c         chain retooling cost ($ per restart after interruption)

DECISION VARIABLES:
  p_{m,j} ∈ {0,1}    1 if maintenance task j on machine m is scheduled
  s_{m,j} ∈ [0, T)   start time of task (m,j), meaningful only when p_{m,j}=1

OBJECTIVE (minimize total expected cost):

  min  Σ_m Σ_j  c_pm_m × p_{m,j}                              ... (1) PM cost
     + Σ_{m∈standalone} Σ_j  v_m × d_m × p_{m,j}              ... (2) standalone prod loss
     + Σ_c  V_c × Σ_j Σ_{m∈machines(c)} d_m × p_{m,j}        ... (3) chain prod loss
     + Σ_c  R_c × Σ_j Σ_{m∈machines(c)} p_{m,j}               ... (4) chain retooling
     + Σ_m  E[failure_cost_m(gaps)]                             ... (5) expected breakdown

  Term (3): When any machine in chain c undergoes PM, the ENTIRE chain
  stops for d_m periods. Chain production loss = V_c × d_m per event.
  (Pessimistic: ignores overlapping PM within a chain. This gives the
  solver incentive to cluster same-chain maintenance together.)

  Term (4): Each PM event on a chain machine triggers a chain restart
  costing R_c. Clustering maintenance reduces effective restarts but
  this approximation is conservative and CP-SAT-friendly.

  Term (5): Weibull expected failures in each gap between PM events.
  For a gap of length g:  E[failures] = (g / η_m)^β_m
  Expected cost = E[failures] × c_cm_m

CONSTRAINTS:
  (C1) Technician capacity:    Cumulative(all intervals, demand=1, cap=K)
  (C2) No overlap per machine: NoOverlap(intervals on machine m)
  (C3) Contiguous numbering:   p_{m,j}=0 → p_{m,j+1}=0
  (C4) Ordering + min gap:     s_{m,j+1} ≥ s_{m,j} + d_m + g_m
                                (enforced when both present)
  (C5) Maximum interval:       first PM within W_m of start,
                                gap between consecutive PMs ≤ W_m,
                                last PM within W_m of horizon end
  (C6) Horizon bounds:         s_{m,j} + d_m ≤ T
"""

import json
from dataclasses import asdict, dataclass, field


@dataclass
class MachineSpec:
    """Specification for a single machine."""
    id: int
    name: str
    maintenance_duration: int       # d_m
    pm_cost: int                    # c_pm_m
    cm_cost: int                    # c_cm_m
    production_value: int           # v_m (used only if standalone)
    weibull_beta: float             # β_m
    weibull_eta: float              # η_m
    max_interval: int               # W_m
    min_gap: int                    # g_m
    repair_factor: float = 1.0     # 1.0 = perfect repair, 0.7 = restores to 70%

    def expected_failures(self, age: int) -> float:
        """E[N(t)] = (t / η)^β — Weibull power law process."""
        if age <= 0:
            return 0.0
        return (age / self.weibull_eta) ** self.weibull_beta

    def expected_failure_cost(self, age: int) -> float:
        """Expected CM cost for a gap of 'age' periods."""
        return self.expected_failures(age) * self.cm_cost

    def virtual_age_after_pm(self, age_before: float) -> float:
        """Virtual age after imperfect PM. Perfect repair → 0, imperfect → residual."""
        return (1.0 - self.repair_factor) * age_before

    def expected_failures_imperfect(self, gap: int, virtual_age: float) -> float:
        """Expected failures in a gap starting from a virtual age.
        E[failures] = ((virtual_age + gap) / η)^β - (virtual_age / η)^β
        """
        if gap <= 0:
            return 0.0
        new_age = virtual_age + gap
        return max(0.0,
            (new_age / self.weibull_eta) ** self.weibull_beta
            - (virtual_age / self.weibull_eta) ** self.weibull_beta
        )

    def optimal_interval_analytical(self) -> float:
        """Closed-form t* = η × (c_pm / (β × (c_cm - c_pm)))^(1/β)."""
        if self.weibull_beta <= 1 or self.cm_cost <= self.pm_cost:
            return float('inf')
        ratio = self.pm_cost / (self.weibull_beta * (self.cm_cost - self.pm_cost))
        return self.weibull_eta * (ratio ** (1.0 / self.weibull_beta))


@dataclass
class ProductionChain:
    """
    A production chain: ordered sequence of machines that together
    produce output. If ANY machine in the chain is down, the entire
    chain produces nothing.

    Economic insight: Maintaining a chain machine is more expensive
    than a standalone machine because you lose the chain's output,
    not just one machine's output. But skipping maintenance risks
    a breakdown that costs even more.
    """
    id: int
    name: str
    machine_ids: list[int]          # ordered: [upstream → downstream]
    chain_value: int                # V_c: $/period when chain is running
    retooling_cost: int             # R_c: $ to restart chain after PM


@dataclass
class ProblemInstance:
    """Complete MaintAlign problem instance."""
    name: str
    num_machines: int
    num_technicians: int            # K
    horizon: int                    # T
    machines: list[MachineSpec]
    chains: list[ProductionChain] = field(default_factory=list)
    blocked_periods: list[int] = field(default_factory=list)  # Calendar: no PM here
    max_tasks_per_machine: int = 0

    def __post_init__(self):
        """Compute derived values and build lookup maps."""
        if self.max_tasks_per_machine == 0:
            for m in self.machines:
                max_t = self.horizon // max(1, m.maintenance_duration + m.min_gap)
                self.max_tasks_per_machine = max(self.max_tasks_per_machine, max_t + 1)

        # Build machine→chain lookup
        self._machine_to_chain: dict[int, int] = {}
        for c in self.chains:
            for mid in c.machine_ids:
                self._machine_to_chain[mid] = c.id

    def get_chain_for_machine(self, machine_id: int) -> ProductionChain | None:
        """Return the chain this machine belongs to, or None."""
        cid = self._machine_to_chain.get(machine_id)
        if cid is not None:
            return self.chains[cid]
        return None

    def is_standalone(self, machine_id: int) -> bool:
        """True if machine is NOT part of any chain."""
        return machine_id not in self._machine_to_chain

    @property
    def standalone_machines(self) -> list[int]:
        return [m.id for m in self.machines if self.is_standalone(m.id)]

    @property
    def resource_constrainedness(self) -> float:
        """RC = total maintenance demand / total technician capacity."""
        total_demand = sum(
            m.maintenance_duration * max(1, self.horizon // m.max_interval)
            for m in self.machines
        )
        total_capacity = self.num_technicians * self.horizon
        return total_demand / total_capacity if total_capacity > 0 else float('inf')

    def task_cost(self, machine_id: int) -> dict:
        """
        Compute the full cost of scheduling one PM task on this machine.
        Returns breakdown: {pm, prod_loss, retooling, total}.

        This is the "weight" of the item in the knapsack analogy.
        """
        m = self.machines[machine_id]
        chain = self.get_chain_for_machine(machine_id)

        pm = m.pm_cost
        if chain:
            prod_loss = chain.chain_value * m.maintenance_duration
            retooling = chain.retooling_cost
        else:
            prod_loss = m.production_value * m.maintenance_duration
            retooling = 0

        return {
            "pm": pm,
            "prod_loss": prod_loss,
            "retooling": retooling,
            "total": pm + prod_loss + retooling
        }

    def save(self, filepath: str):
        data = {
            "name": self.name,
            "num_machines": self.num_machines,
            "num_technicians": self.num_technicians,
            "horizon": self.horizon,
            "max_tasks_per_machine": self.max_tasks_per_machine,
            "machines": [asdict(m) for m in self.machines],
            "chains": [asdict(c) for c in self.chains],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'ProblemInstance':
        with open(filepath) as f:
            data = json.load(f)
        machines = [MachineSpec(**m) for m in data["machines"]]
        chains = [ProductionChain(**c) for c in data.get("chains", [])]
        return cls(
            name=data["name"],
            num_machines=data["num_machines"],
            num_technicians=data["num_technicians"],
            horizon=data["horizon"],
            machines=machines,
            chains=chains,
            max_tasks_per_machine=data.get("max_tasks_per_machine", 0),
        )

    def summary(self) -> str:
        lines = [
            f"{'═'*70}",
            f" Instance: {self.name}",
            f" {self.num_machines} machines, {self.num_technicians} technicians, "
            f"{self.horizon} periods, {len(self.chains)} chains",
            f" RC = {self.resource_constrainedness:.2f}   "
            f"Max tasks/machine = {self.max_tasks_per_machine}",
            f"{'═'*70}",
        ]

        if self.chains:
            lines.append("\n Production Chains:")
            for c in self.chains:
                mnames = [self.machines[mid].name for mid in c.machine_ids]
                lines.append(
                    f"   Chain {c.id} '{c.name}': {' → '.join(mnames)}  "
                    f"V=${c.chain_value}/t  R=${c.retooling_cost}"
                )
            standalone = self.standalone_machines
            if standalone:
                snames = [self.machines[mid].name for mid in standalone]
                lines.append(f"   Standalone: {', '.join(snames)}")

        lines.append(f"\n {'ID':<4}{'Name':<12}{'Dur':<5}{'PM$':<6}{'CM$':<7}"
                     f"{'β':<5}{'η':<6}{'W':<5}{'TaskCost':<10}{'t*':<6}")
        lines.append(f" {'-'*62}")
        for m in self.machines:
            tc = self.task_cost(m.id)
            t_star = m.optimal_interval_analytical()
            ts = f"{t_star:.1f}" if t_star < float('inf') else "N/A"
            lines.append(
                f" {m.id:<4}{m.name:<12}{m.maintenance_duration:<5}"
                f"{m.pm_cost:<6}{m.cm_cost:<7}"
                f"{m.weibull_beta:<5.1f}{m.weibull_eta:<6.1f}{m.max_interval:<5}"
                f"${tc['total']:<9}{ts:<6}"
            )
        return "\n".join(lines)
