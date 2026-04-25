# MaintAlign — Research Experiments

This directory contains the scripts and data used to produce the figures and
tables in the CMPSC 580 junior-seminar report.

## Structure

```
experiments/
├── scripts/               # Experiment runner scripts
│   ├── run_all.sh          # Run the full experiment pipeline
│   ├── run_baselines.py    # Baseline strategy comparison
│   ├── run_scalability.py  # Scalability benchmarks (varying M, K, T)
│   ├── run_montecarlo.py   # Monte Carlo risk simulations
│   └── generate_figures.py # Generate publication-ready figures
├── results/               # Raw CSV outputs from experiment runs
│   ├── baseline_comparison.csv
│   ├── baselines_quick.csv
│   ├── montecarlo_results.csv
│   └── scalability_results.csv
└── figures/               # Generated figures (PNG)
    ├── baseline_bar_chart.png
    ├── montecarlo_summary.png
    ├── savings_summary.png
    └── scalability_plot.png
```

## Reproducing the experiments

From the project root, with the virtual environment activated:

```bash
# Run the full pipeline (baselines → scalability → Monte Carlo → figures)
bash experiments/scripts/run_all.sh

# Or run individual experiments
python experiments/scripts/run_baselines.py
python experiments/scripts/run_scalability.py
python experiments/scripts/run_montecarlo.py
python experiments/scripts/generate_figures.py
```

Results are saved to `experiments/results/` and figures to `experiments/figures/`.
