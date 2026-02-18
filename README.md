# MaintAlign — Maintenance Scheduling Optimizer

**Resource-constrained preventive maintenance scheduling with production chains.**

MaintAlign uses Google OR-Tools CP-SAT to find optimal maintenance schedules that balance PM costs, production losses, chain retooling costs, and Weibull-modeled failure risks — under limited technician capacity.

## Quick Start

```bash
# 1. Create a virtual environment (one-time setup)
python3 -m venv .venv

# 2. Activate the virtual environment (run this every time you open a new terminal)
source .venv/bin/activate

# 3. Install dependencies (one-time, after activating venv)
pip install -r requirements.txt

# 4. Run demo (tiny + small + medium instances, ~30s)
python main.py

# 5. Results appear in results/demo/ (Gantt charts, cost comparisons, JSON)
```

> **Note:** You must activate the virtual environment (`source .venv/bin/activate`) before running any commands. The project's dependencies (OR-Tools, matplotlib, numpy) are installed inside `.venv` and won't be available otherwise.

## Usage

```bash
# Always activate venv first
source .venv/bin/activate
```

| Command | What it does |
|---|---|
| `python main.py` | Demo: 3 instance sizes with baselines + optimization |
| `python main.py --full` | Full suite: 5 sizes × 3 seeds each |
| `python main.py --sensitivity` | Sensitivity analysis: vary technicians & cost ratios |
| `python main.py --instance FILE` | Solve a specific JSON instance file |
| `python main.py --log-level DEBUG` | Enable verbose logging |

## Project Structure

```
MaintAlign/
├── main.py              # Experiment runner (entry point)
├── streamlit_app.py     # Interactive web dashboard
├── core/                # Core data models & solvers
│   ├── instance.py      # Data model (machines, chains, Weibull)
│   ├── solver.py        # CP-SAT solver (optional tasks + chain costs)
│   ├── baseline.py      # 4 baseline strategies for comparison
│   └── decomposer.py    # Problem decomposition
├── analysis/            # Simulation & evaluation
│   ├── simulator.py     # Monte Carlo simulation engine
│   └── evaluator.py     # Schedule evaluation & comparison
├── utils/               # Utilities
│   ├── generator.py     # Instance generator with difficulty presets
│   └── visualizer.py    # Gantt charts, cost breakdown plots
├── data/                # Instance data files
├── results/             # Output (Gantt charts, JSON, comparisons)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Problem Overview

- **Machines** have Weibull failure distributions — the longer since last PM, the higher the breakdown risk.
- **Production chains** link machines: if ANY machine in a chain is down, the ENTIRE chain stops.
- **Technicians** are a shared, limited resource — not all machines can be maintained simultaneously.
- The solver decides **which** machines to maintain, **when**, and **how often** — minimizing total expected cost.

## Key Concepts

| Term | Meaning |
|---|---|
| PM | Preventive Maintenance (planned) |
| CM | Corrective Maintenance (breakdown repair, much more expensive) |
| RC | Resource Constrainedness = total maintenance demand / technician capacity |
| β (beta) | Weibull shape: higher = more predictable wear-out |
| η (eta) | Weibull scale: characteristic life in periods |
| W | Max interval allowed between PMs |

## Output Files

After a run, check the `results/` directory:

- `*_gantt_opt.png` — Gantt chart of optimized schedule
- `*_gantt_base.png` — Gantt chart of best baseline
- `*_cost_cmp.png` — Stacked bar: cost breakdown comparison
- `*_tech_util.png` — Technician utilization over time
- `*_results.json` — Full results with cost breakdown
