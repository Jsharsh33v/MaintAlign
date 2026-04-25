# MaintAlign — Maintenance Scheduling Optimizer

[![CI — Lint & Test](https://github.com/Jsharsh33v/MaintAlign/actions/workflows/ci.yml/badge.svg)](https://github.com/Jsharsh33v/MaintAlign/actions/workflows/ci.yml)

**Resource-constrained preventive maintenance scheduling with production chains.**

MaintAlign uses Google OR-Tools CP-SAT to find optimal maintenance schedules that
balance PM costs, production losses, chain retooling costs, and Weibull-modeled
failure risks — under limited technician capacity. It ships with a CLI, an
interactive Streamlit dashboard, runnable examples, and a full test suite.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Verifying the Installation](#verifying-the-installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Running the Examples](#running-the-examples)
- [Running the Tests](#running-the-tests)
- [Linting](#linting)
- [Input Validation](#input-validation)
- [Problem Overview](#problem-overview)
- [Key Concepts](#key-concepts)
- [Output Files](#output-files)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

**Requirements:** Python 3.10+ on macOS, Linux, or Windows.

```bash
# 1. Clone the repository
git clone https://github.com/Jsharsh33v/MaintAlign.git
cd MaintAlign

# 2. Create and activate a virtual environment (one-time setup)
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell

# 3. Install the package in editable mode
pip install -e .

# 4. Run the quick demo (~30 seconds)
python main.py
```

Results (Gantt charts, cost breakdowns, JSON) appear in `results/demo/`.

> **Note:** You must activate the virtual environment (`source .venv/bin/activate`)
> before running any commands. All dependencies (OR-Tools, matplotlib, numpy,
> Streamlit, Plotly, pandas) are installed inside `.venv`.

---

## Verifying the Installation

After installing, run these two checks to confirm everything works:

```bash
# 1. Run the quick-start example
python examples/01_quick_start.py
# → Should print a solver result with "OPTIMAL" status and cost breakdown

# 2. Run the test suite
pip install -e ".[dev]"          # install dev dependencies (pytest, ruff)
pytest
# → All tests should pass
```

---

## Usage

Always activate the virtual environment first:

```bash
source .venv/bin/activate
```

### CLI (`main.py`)

| Command | What it does |
|---|---|
| `python main.py` | Demo: 3 instance sizes with baselines + optimization |
| `python main.py --full` | Full suite: 8 sizes × 3 seeds each |
| `python main.py --sensitivity` | Sensitivity analysis: vary technicians & cost ratios |
| `python main.py --csv MACHINES.csv` | Solve from a CSV machines file |
| `python main.py --csv MACHINES.csv --chains CHAINS.csv` | Solve from CSV with production chains |
| `python main.py --simulate --num-sims 500` | Add Monte Carlo risk simulation |
| `python main.py --weekends` | Block every 6th and 7th period (weekends) |
| `python main.py --repair-factor 0.7` | Imperfect PM repair (Kijima Type I) |
| `python main.py --decompose` | Use decomposition for large instances |
| `python main.py --log-level DEBUG` | Verbose logging |

### Dashboard (`streamlit_app.py`)

```bash
streamlit run streamlit_app.py
```

The interactive dashboard lets you select generated instances or upload your own
CSV, configure solver parameters, and explore Gantt charts, cost breakdowns,
technician utilization, and Monte Carlo risk analysis — all from your browser.

---

## Project Structure

```
MaintAlign/
├── main.py                        # CLI entry point
├── streamlit_app.py               # Interactive Streamlit dashboard
├── pyproject.toml                 # Package metadata, Ruff & pytest config
├── requirements.txt               # Minimal pip dependencies
│
├── core/                          # Core data models & solvers
│   ├── instance.py                #   ProblemInstance, MachineSpec, ProductionChain
│   ├── solver.py                  #   CP-SAT solver (optional tasks + chain costs)
│   ├── baseline.py                #   4 baseline strategies for comparison
│   ├── decomposer.py              #   Problem decomposition for large instances
│   └── validators.py              #   Input validation & custom exceptions
│
├── analysis/                      # Simulation & evaluation
│   ├── simulator.py               #   Monte Carlo simulation engine
│   └── evaluator.py               #   Schedule evaluation & comparison
│
├── utils/                         # Utilities
│   ├── generator.py               #   Instance generator with difficulty presets
│   ├── csv_loader.py              #   CSV → ProblemInstance loader
│   └── visualizer.py              #   Gantt charts, cost breakdown plots
│
├── examples/                      # Runnable usage examples
│   ├── 01_quick_start.py          #   Minimal generate → solve → print
│   ├── 02_custom_instance.py      #   Build instance by hand
│   ├── 03_csv_workflow.py         #   CSV → solve → compare baselines
│   ├── 04_monte_carlo.py          #   Monte Carlo risk analysis
│   ├── sample_data/               #   Ready-to-use CSV files
│   └── README.md                  #   Example guide
│
├── tests/                         # pytest test suite
│   ├── conftest.py                #   Shared fixtures
│   ├── test_smoke.py              #   End-to-end smoke tests
│   ├── test_validators.py         #   Validator unit tests
│   └── test_csv_loader.py         #   CSV parsing tests
│
├── experiments/                   # Research experiment scripts & data
│   ├── scripts/                   #   Experiment runner scripts
│   ├── results/                   #   Raw CSV outputs
│   ├── figures/                   #   Generated publication figures
│   └── README.md                  #   Experiment guide
│
├── data/                          # Sample instance data (CSV)
├── docs/                          # Course documentation & journal entries
├── results/                       # Output (charts, JSON — gitignored)
│
├── .github/workflows/ci.yml       # CI: lint + test on every push
├── CONTRIBUTING.md                # Development workflow guide
└── README.md                      # This file
```

---

## Running the Examples

The `examples/` directory contains four progressive, runnable scripts. See the
[examples README](examples/README.md) for full details.

```bash
# Quick start — generate, solve, print results
python examples/01_quick_start.py

# Build an instance by hand with a production chain
python examples/02_custom_instance.py

# Load from CSV, solve, compare baselines
python examples/03_csv_workflow.py

# Monte Carlo risk analysis (~30-60 seconds)
python examples/04_monte_carlo.py
```

---

## Running the Tests

```bash
# Install dev dependencies (if not already)
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=core --cov=utils --cov=analysis --cov-report=term-missing

# Run a specific test file
pytest tests/test_validators.py
```

The CI workflow (`.github/workflows/ci.yml`) runs `ruff check .` and `pytest -v`
automatically on every push to `main` and on pull requests targeting `main`,
across Python 3.10, 3.11, and 3.12.

---

## Linting

All Python code is linted with [Ruff](https://docs.astral.sh/ruff/). Configuration
lives in `pyproject.toml`.

```bash
# Check for lint errors
ruff check .

# Auto-fix what can be auto-fixed
ruff check --fix .

# Format code (similar to Black)
ruff format .
```

---

## Input Validation

MaintAlign enforces input validation before the solver runs, surfacing clear error
messages instead of cryptic tracebacks. The validation module (`core/validators.py`)
defines a custom exception hierarchy:

| Exception | When it's raised |
|---|---|
| `MaintAlignError` | Base class for all validation errors |
| `InvalidMachineSpecError` | Machine parameters are invalid |
| `InvalidInstanceError` | Problem instance is inconsistent or infeasible |
| `InvalidCSVRowError` | A CSV row cannot be parsed into a valid machine |
| `InvalidSolverParamsError` | Solver parameters (time limit, repair factor, etc.) are bad |

### Checks enforced

- `max_interval ≥ maintenance_duration` — otherwise no PM can fit
- `cm_cost > pm_cost` — otherwise PM is never economical
- `repair_factor ∈ (0, 1]` — valid range for Kijima Type I imperfect repair
- Chain machine IDs all reference existing machines
- No machine appears in multiple chains
- `blocked_periods` are within `[0, horizon)`
- Horizon is at least as large as the longest maintenance duration
- CSV rows have the correct number of columns with valid types
- Solver time limit is positive; Monte Carlo `n_sims ≥ 1`

Both the CLI (`main.py`) and the Streamlit dashboard (`streamlit_app.py`) wrap
user-facing operations in try/except blocks that catch `MaintAlignError` and
present the validation message cleanly — no raw tracebacks are shown to the user.

---

## Problem Overview

- **Machines** have Weibull failure distributions — the longer since last PM, the higher the breakdown risk.
- **Production chains** link machines: if ANY machine in a chain is down, the ENTIRE chain stops.
- **Technicians** are a shared, limited resource — not all machines can be maintained simultaneously.
- The solver decides **which** machines to maintain, **when**, and **how often** — minimizing total expected cost.

---

## Key Concepts

| Term | Meaning |
|---|---|
| PM | Preventive Maintenance (planned) |
| CM | Corrective Maintenance (breakdown repair, much more expensive) |
| RC | Resource Constrainedness = total maintenance demand / technician capacity |
| β (beta) | Weibull shape: higher = more predictable wear-out |
| η (eta) | Weibull scale: characteristic life in periods |
| W | Max interval allowed between PMs |
| Kijima Type I | Imperfect repair model: PM restores machine to `repair_factor × age`, not fully new |

---

## Output Files

After a run, check the `results/` directory:

- `*_gantt_opt.png` — Gantt chart of optimized schedule
- `*_gantt_base.png` — Gantt chart of best baseline
- `*_cost_cmp.png` — Stacked bar: cost breakdown comparison
- `*_tech_util.png` — Technician utilization over time
- `*_chain_breakdown.png` — Per-chain cost breakdown (if chains exist)
- `*_results.json` — Full results with cost breakdown

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing,
linting, commit conventions, and the pull-request workflow.

---

## License

This is a research prototype developed for CMPSC 580 at Allegheny College.
It is **not** validated for safety-critical or production deployment.
Use at your own risk.
