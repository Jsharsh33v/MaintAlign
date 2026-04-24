# Contributing to MaintAlign

This document describes the development workflow for the MaintAlign
prototype. It covers local setup, the test suite, linting, commit
conventions, and the pull-request flow.

## Development setup

Clone the repo and create a virtual environment:

```bash
git clone https://github.com/Jsharsh33v/MaintAlign.git
cd MaintAlign

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e ".[dev]"
```

The `-e ".[dev]"` install pulls in `pytest`, `pytest-cov`, and `ruff`
from `pyproject.toml`.

## Running the tests

The project uses `pytest`. Run the full suite with:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=core --cov=utils --cov=analysis --cov-report=term-missing
```

Run a single test file or a single test:

```bash
pytest tests/test_smoke.py
pytest tests/test_validators.py::TestValidateMachineSpec::test_valid_spec_passes
```

## Linting

All Python code is linted with [Ruff](https://docs.astral.sh/ruff/).
The configuration lives in `pyproject.toml`.

```bash
# Check for lint errors
ruff check .

# Auto-fix what can be auto-fixed
ruff check --fix .

# Format files (Ruff also formats, similar to Black)
ruff format .
```

The CI workflow will fail if `ruff check .` reports any errors, so
please run it locally before pushing.

## Code organization

```
MaintAlign/
├── core/              # Data model, solver, baselines, validators
├── utils/             # CSV loader, instance generator, visualizer
├── analysis/          # Monte Carlo simulator and evaluator
├── examples/          # Runnable usage examples
├── experiments/       # Scripts + CSV outputs for the research experiments
├── tests/             # pytest test suite
├── main.py            # CLI entry point
└── streamlit_app.py   # Interactive dashboard
```

New features should include:

1. Code in the appropriate package (`core/`, `utils/`, or `analysis/`).
2. Input validation in `core/validators.py` (raise a `MaintAlignError`
   subclass, not a bare `ValueError`, so the UI can catch it cleanly).
3. At least one unit test in `tests/`.
4. A docstring on any public function explaining inputs and outputs.

## Commit conventions

Short, plain commit messages that describe what changed. Examples:

```
Add examples directory with four runnable demos
Fix Streamlit error when uploaded CSV has wrong columns
Wire validators into main.py CSV loader path
Bump ortools to 9.10
```

Avoid generic messages like "fixes" or "update code". If a commit
closes an issue, reference it (e.g., `Fix #12: Monte Carlo crashes on empty chains`).

## Releases

Research deliverables are tagged using semantic versioning with a
descriptive prefix. Example:

```bash
git tag junior_seminar_report-0.4.0
git push origin main --tags
```

The `render.yml` GitHub Action on the class repo listens for these
tags and compiles the Quarto report to a PDF release.

## Reporting bugs

Open an issue on GitHub with:

- A description of what you expected to happen.
- What actually happened (paste the traceback if there is one).
- A minimal reproduction (code snippet, CSV file, or command).
- Your Python version and OS.
