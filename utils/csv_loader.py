"""
MaintAlign - CSV Data Loader
==============================
Load real machine data from CSV files into ProblemInstance.

CSV FORMAT (machines):
  name, duration, pm_cost, cm_cost, prod_value, beta, eta, max_interval, min_gap

CSV FORMAT (chains, optional second file):
  chain_name, machine_names (semicolon-separated), chain_value, retooling_cost

First line of machines CSV must be a header row.
A special comment line starting with '#CONFIG' provides instance metadata:
  #CONFIG, instance_name, num_technicians, horizon

Example:
  python csv_loader.py data/example_machines.csv
  python csv_loader.py data/example_machines.csv --chains data/example_chains.csv
"""

import csv
import logging
import sys
from pathlib import Path

from core.instance import MachineSpec, ProblemInstance, ProductionChain

logger = logging.getLogger(__name__)


def load_machines_csv(filepath: str) -> tuple:
    """
    Parse machines CSV file.

    Returns:
        (instance_name, num_technicians, horizon, list of MachineSpec)
    """
    machines = []
    instance_name = Path(filepath).stem
    num_technicians = 2
    horizon = 30

    with open(filepath, newline='') as f:
        reader = csv.reader(f)

        for row in reader:
            if not row or not row[0].strip():
                continue

            # Parse config line
            if row[0].strip().upper().startswith('#CONFIG'):
                if len(row) >= 4:
                    instance_name = row[1].strip()
                    num_technicians = int(row[2].strip())
                    horizon = int(row[3].strip())
                    logger.info("Config: name=%s, K=%d, H=%d",
                                instance_name, num_technicians, horizon)
                continue

            # Skip comments
            if row[0].strip().startswith('#'):
                continue

            # Skip header row
            if row[0].strip().lower() == 'name':
                continue

            # Parse machine row
            if len(row) < 9:
                logger.warning("Skipping malformed row (need 9 cols): %s", row)
                continue

            try:
                m = MachineSpec(
                    id=len(machines),
                    name=row[0].strip(),
                    maintenance_duration=int(row[1].strip()),
                    pm_cost=int(row[2].strip()),
                    cm_cost=int(row[3].strip()),
                    production_value=int(row[4].strip()),
                    weibull_beta=float(row[5].strip()),
                    weibull_eta=float(row[6].strip()),
                    max_interval=int(row[7].strip()),
                    min_gap=int(row[8].strip()),
                )
                machines.append(m)
            except (ValueError, IndexError) as e:
                logger.warning("Skipping invalid row %s: %s", row, e)

    if not machines:
        raise ValueError(f"No valid machines found in {filepath}")

    logger.info("Loaded %d machines from %s", len(machines), filepath)
    return instance_name, num_technicians, horizon, machines


def load_chains_csv(filepath: str, machines: list) -> list:
    """
    Parse chains CSV file.

    Columns: chain_name, machine_names (semicolon-separated), chain_value, retooling_cost

    Machine names must match names from the machines CSV.
    """
    name_to_id = {m.name: m.id for m in machines}
    chains = []

    with open(filepath, newline='') as f:
        reader = csv.reader(f)

        for row in reader:
            if not row or not row[0].strip():
                continue
            if row[0].strip().startswith('#') or row[0].strip().lower() == 'chain_name':
                continue

            if len(row) < 4:
                logger.warning("Skipping malformed chain row: %s", row)
                continue

            try:
                chain_name = row[0].strip()
                machine_names = [n.strip() for n in row[1].split(';')]
                chain_value = int(row[2].strip())
                retooling_cost = int(row[3].strip())

                # Resolve machine names to IDs
                machine_ids = []
                for mname in machine_names:
                    if mname not in name_to_id:
                        logger.warning("Chain '%s': unknown machine '%s', skipping chain",
                                       chain_name, mname)
                        break
                    machine_ids.append(name_to_id[mname])
                else:
                    chains.append(ProductionChain(
                        id=len(chains),
                        name=chain_name,
                        machine_ids=machine_ids,
                        chain_value=chain_value,
                        retooling_cost=retooling_cost,
                    ))
            except (ValueError, IndexError) as e:
                logger.warning("Skipping invalid chain row %s: %s", row, e)

    logger.info("Loaded %d chains from %s", len(chains), filepath)
    return chains


def load_instance(machines_csv: str,
                  chains_csv: str | None = None) -> ProblemInstance:
    """
    Load a complete ProblemInstance from CSV file(s).

    Args:
        machines_csv: path to machines CSV file
        chains_csv: optional path to chains CSV file

    Returns:
        ProblemInstance ready for the solver
    """
    name, K, H, machines = load_machines_csv(machines_csv)

    chains = []
    if chains_csv:
        chains = load_chains_csv(chains_csv, machines)

    instance = ProblemInstance(
        name=name,
        num_machines=len(machines),
        num_technicians=K,
        horizon=H,
        machines=machines,
        chains=chains,
    )

    logger.info("Created instance '%s': %dM, %dK, %dT, %d chains",
                name, len(machines), K, H, len(chains))
    return instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python csv_loader.py <machines.csv> [--chains <chains.csv>]")
        print("\nRunning with example data...")

        # Create example data directory and files for testing
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        machines_file = data_dir / "example_machines.csv"
        chains_file = data_dir / "example_chains.csv"

        if not machines_file.exists():
            print(f"  No example files found. Create them in {data_dir}/")
            sys.exit(1)

        inst = load_instance(str(machines_file), str(chains_file))
        print(inst.summary())
    else:
        machines_path = sys.argv[1]
        chains_path = None
        if "--chains" in sys.argv:
            idx = sys.argv.index("--chains")
            if idx + 1 < len(sys.argv):
                chains_path = sys.argv[idx + 1]

        inst = load_instance(machines_path, chains_path)
        print(inst.summary())
