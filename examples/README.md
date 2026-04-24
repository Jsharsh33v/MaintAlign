# MaintAlign — Examples

Four runnable scripts that demonstrate typical MaintAlign usage,
ordered from simplest to most involved. Each script is self-contained
and can be run directly from the project root.

## Setup

From the project root, activate the virtual environment:

```bash
source .venv/bin/activate
```

## Running the examples

| Script | What it demonstrates |
|---|---|
| `01_quick_start.py` | Generate a tiny instance, solve it, print the schedule |
| `02_custom_instance.py` | Build a problem instance by hand (machines + one chain) |
| `03_csv_workflow.py` | Load an instance from CSV, solve, compare to baselines |
| `04_monte_carlo.py` | Monte Carlo risk analysis across all strategies |

Run any example with:

```bash
python examples/01_quick_start.py
python examples/02_custom_instance.py
python examples/03_csv_workflow.py
python examples/04_monte_carlo.py
```

## Expected output

### 01_quick_start.py

Prints instance summary, solver status, total cost, and the
per-machine list of maintenance start periods. Runs in under 5 seconds.

### 02_custom_instance.py

Builds a 4-machine instance with a 2-machine production chain.
Validates the instance, solves it, and reports the percentage savings
of the CP-SAT schedule over the analytical baseline.

### 03_csv_workflow.py

Loads `sample_data/example_machines.csv` and `sample_data/example_chains.csv`,
runs all four baseline strategies, then runs the CP-SAT optimizer and
prints the savings over the best baseline.

### 04_monte_carlo.py

Runs 500 random breakdown simulations per strategy on a small instance
and prints a table with mean cost, standard deviation, VaR95 (tail risk),
and mean number of failures. Takes about 30-60 seconds.

## Sample data

`sample_data/example_machines.csv` contains six machines representing a
small job shop: CNC lathe, drill press, welder, hydraulic press, grinder,
and inspector. `sample_data/example_chains.csv` groups them into two
production chains (Body_Line and Finish_Line). These files are safe to
copy and modify as templates for your own factory data.

## Writing your own

To create a new example, use one of the four scripts as a template.
Make sure to add the project root to `sys.path` at the top:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
```

Then import what you need from `core`, `utils`, and `analysis`.
