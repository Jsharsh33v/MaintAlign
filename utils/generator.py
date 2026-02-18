"""
MaintAlign - Instance Generator (v2: Chains + Feasibility)
============================================================
Generates problem instances with production chains and controlled difficulty.

CHAIN GENERATION STRATEGY:
  - Group machines into chains of 2-4 machines each
  - Some machines remain standalone (not in any chain)
  - Chain value > sum of individual machine values (synergy bonus)
  - Chain retooling cost is proportional to chain length

DIFFICULTY CONTROLS:
  Primary:   Resource Constrainedness (RC = demand / capacity)
  Secondary: Weibull β (higher = PM more valuable)
  Tertiary:  Cost ratio CM/PM (higher = bigger penalty for skipping PM)
  Chain:     Chain fraction (more chains = harder economic trade-offs)
"""

import random
import os
import logging
from core.instance import MachineSpec, ProductionChain, ProblemInstance

logger = logging.getLogger(__name__)

MACHINE_NAMES = [
    "CNC_Lathe", "Mill_Horiz", "Mill_Vert", "Drill_Press", "Grinder",
    "Welder_A", "Welder_B", "Press_Hyd", "Press_Mech", "Robot_Arm",
    "Conveyor", "Furnace", "Plater", "LaserCut", "PlasmaCut",
    "Bender", "Punch", "Polisher", "Coater", "Inspector",
    "Stamper", "Assembler", "Tester", "Packager", "Loader",
]

CHAIN_NAMES = [
    "Engine_Line", "Body_Line", "Axle_Line", "Gearbox_Line",
    "Frame_Line", "Exhaust_Line", "Brake_Line", "Wiring_Line",
    "Panel_Line", "Trim_Line",
]


def _validate_params(num_machines, num_technicians, horizon, num_chains):
    """Validate generator input parameters."""
    if num_machines < 1:
        raise ValueError(f"num_machines must be ≥ 1, got {num_machines}")
    if num_technicians < 1:
        raise ValueError(f"num_technicians must be ≥ 1, got {num_technicians}")
    if horizon < 4:
        raise ValueError(f"horizon must be ≥ 4, got {horizon}")
    if num_chains < 0:
        raise ValueError(f"num_chains must be ≥ 0, got {num_chains}")
    if num_chains > 0 and num_machines < 4:
        logger.warning("num_chains > 0 requires at least 4 machines; "
                       "chains will be empty")


def _generate_machine(mid: int, horizon: int, rng: random.Random,
                       beta_range=(1.5, 3.0), cost_ratio_range=(3, 10)) -> MachineSpec:
    """Generate one machine with feasible parameters."""
    name = MACHINE_NAMES[mid % len(MACHINE_NAMES)]
    if mid >= len(MACHINE_NAMES):
        name += f"_{mid // len(MACHINE_NAMES)}"

    dur = rng.randint(1, max(1, min(2, horizon // 6)))
    pm_cost = rng.randint(100, 500)
    cm_cost = int(pm_cost * rng.uniform(*cost_ratio_range))
    prod_val = rng.randint(50, 200)

    beta = round(rng.uniform(*beta_range), 2)

    # Max interval: at least 4×duration and at least horizon/3
    min_W = max(dur * 4, horizon // 3)
    max_W = max(min_W + 1, horizon * 2 // 3)
    W = rng.randint(min_W, max_W)

    # Eta: calibrate so expected failures at W ~ 0.5-1.5
    target = rng.uniform(0.5, 1.5)
    eta = round(W / (target ** (1.0 / beta)), 1)

    return MachineSpec(
        id=mid, name=name,
        maintenance_duration=dur,
        pm_cost=pm_cost, cm_cost=cm_cost,
        production_value=prod_val,
        weibull_beta=beta, weibull_eta=eta,
        max_interval=W, min_gap=1,
    )


def _generate_chains(machines: list, num_chains: int,
                      rng: random.Random) -> list:
    """
    Assign machines to production chains.
    Each chain has 2-4 machines. Remaining machines are standalone.
    """
    M = len(machines)
    if num_chains == 0 or M < 4:
        return []

    # Shuffle machine IDs and assign to chains
    ids = list(range(M))
    rng.shuffle(ids)

    chains = []
    idx = 0
    for c in range(num_chains):
        if idx >= M - 1:
            break
        chain_len = rng.randint(2, min(4, M - idx))
        chain_mids = ids[idx:idx + chain_len]
        idx += chain_len

        # Chain value: sum of individual machine values × 1.5-2.5 synergy
        base_val = sum(machines[mid].production_value for mid in chain_mids)
        synergy = rng.uniform(1.5, 2.5)
        chain_value = int(base_val * synergy)

        # Retooling cost: proportional to chain length
        retool = int(rng.uniform(100, 300) * chain_len)

        chains.append(ProductionChain(
            id=c,
            name=CHAIN_NAMES[c % len(CHAIN_NAMES)],
            machine_ids=chain_mids,
            chain_value=chain_value,
            retooling_cost=retool,
        ))

    logger.info("Generated %d chains covering %d/%d machines",
                len(chains), idx, M)
    return chains


def generate_instance(
    name: str, num_machines: int, num_technicians: int, horizon: int,
    num_chains: int = 0, seed: int = 42,
    beta_range=(1.5, 3.0), cost_ratio_range=(3, 10),
) -> ProblemInstance:
    """
    Generate a complete problem instance.

    Args:
        name: instance identifier
        num_machines: M
        num_technicians: K
        horizon: T
        num_chains: number of production chains (0 = no chains)
        seed: random seed
        beta_range: Weibull shape parameter range
        cost_ratio_range: CM/PM cost ratio range
    """
    _validate_params(num_machines, num_technicians, horizon, num_chains)

    rng = random.Random(seed)
    machines = [
        _generate_machine(i, horizon, rng, beta_range, cost_ratio_range)
        for i in range(num_machines)
    ]
    chains = _generate_chains(machines, num_chains, rng)

    inst = ProblemInstance(
        name=name,
        num_machines=num_machines,
        num_technicians=num_technicians,
        horizon=horizon,
        machines=machines,
        chains=chains,
    )
    logger.info("Instance '%s': %dM/%dK/%dT, RC=%.2f, %d chains",
                name, num_machines, num_technicians, horizon,
                inst.resource_constrainedness, len(chains))
    return inst


def generate_custom(
    difficulty: str = "medium",
    seed: int = 42,
) -> ProblemInstance:
    """
    Convenience function: generate an instance by difficulty string.

    Args:
        difficulty: one of "easy", "medium", "hard", "extreme"
        seed: random seed
    """
    presets = {
        "easy":    ("easy",    6,  3, 20, 1),
        "medium":  ("medium", 10,  4, 30, 2),
        "hard":    ("hard",   10,  2, 30, 2),
        "extreme": ("extreme", 20, 3, 50, 4),
    }
    difficulty = difficulty.lower()
    if difficulty not in presets:
        raise ValueError(f"Unknown difficulty '{difficulty}'. "
                         f"Choose from: {list(presets.keys())}")

    name, M, K, T, C = presets[difficulty]
    return generate_instance(name, M, K, T, num_chains=C, seed=seed)


# ═══ PRESETS ════════════════════════════════════════════════════════════

def generate_tiny(seed=42):
    """3M / 1K / 12T / 0 chains — debugging, hand-verifiable."""
    return generate_instance("tiny", 3, 1, 12, num_chains=0, seed=seed)

def generate_small(seed=42):
    """6M / 2K / 20T / 1 chain — unit testing with chains."""
    return generate_instance("small", 6, 2, 20, num_chains=1, seed=seed)

def generate_medium_easy(seed=42):
    """10M / 4K / 30T / 2 chains — baseline experiments."""
    return generate_instance("medium_easy", 10, 4, 30, num_chains=2, seed=seed)

def generate_medium_hard(seed=42):
    """10M / 2K / 30T / 2 chains — stress testing (tight resources)."""
    return generate_instance("medium_hard", 10, 2, 30, num_chains=2, seed=seed)

def generate_large(seed=42):
    """20M / 5K / 50T / 4 chains — scalability."""
    return generate_instance("large", 20, 5, 50, num_chains=4, seed=seed)

def generate_xl(seed=42):
    """40M / 8K / 80T / 6 chains — solver limits."""
    return generate_instance("xl", 40, 8, 80, num_chains=6, seed=seed)

def generate_industrial(seed=42):
    """50M / 10K / 60T / 8 chains — industrial scale."""
    return generate_instance("industrial", 50, 10, 60, num_chains=8, seed=seed)

def generate_factory(seed=42):
    """100M / 15K / 60T / 12 chains — full factory floor."""
    return generate_instance("factory", 100, 15, 60, num_chains=12, seed=seed)


def generate_experiment_suite(output_dir="instances", num_seeds=5):
    """Generate full experimental suite: all presets × multiple seeds."""
    os.makedirs(output_dir, exist_ok=True)
    presets = {
        "tiny":        (3, 1, 12, 0),
        "small":       (6, 2, 20, 1),
        "medium_easy": (10, 4, 30, 2),
        "medium_hard": (10, 2, 30, 2),
        "large":       (20, 5, 50, 4),
    }
    instances = []
    for pname, (M, K, T, C) in presets.items():
        for s in range(num_seeds):
            iname = f"{pname}_s{s}"
            inst = generate_instance(iname, M, K, T, num_chains=C, seed=s)
            inst.save(os.path.join(output_dir, f"{iname}.json"))
            instances.append(inst)
            print(f"  {iname}: RC={inst.resource_constrainedness:.2f}, "
                  f"{len(inst.chains)} chains")
    return instances


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(name)s | %(message)s")
    print("MaintAlign Instance Generator v2\n")
    for gen in [generate_tiny, generate_small, generate_medium_easy,
                generate_medium_hard, generate_large]:
        inst = gen()
        print(inst.summary())
        print()
