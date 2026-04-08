"""
Generate Figures for Chapter 4
================================
Reads experiment CSV results and creates publication-ready figures.

Produces:
  1. scalability_plot.png    — Line chart: solve time vs problem size
  2. baseline_bar_chart.png  — Grouped bar chart: optimized vs baselines
  3. savings_summary.png     — Bar chart: % savings over best baseline
  4. montecarlo_summary.png  — Grouped bar: Monte Carlo mean cost + VaR95
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})


def load_csv(filename):
    """Load a CSV file and return list of dicts."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping.")
        return []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_scalability():
    """Figure 1: Solve time vs number of machines."""
    data = load_csv("scalability_results.csv")
    if not data:
        return

    # Group by num_machines, compute mean and std of solve time
    by_size = defaultdict(list)
    for row in data:
        M = int(row["num_machines"])
        t = float(row["solve_time_sec"])
        by_size[M].append(t)

    sizes = sorted(by_size.keys())
    means = [np.mean(by_size[s]) for s in sizes]
    stds = [np.std(by_size[s]) for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sizes, means, yerr=stds, fmt='o-', color='#1976D2',
                capsize=5, capthick=1.5, linewidth=2, markersize=8,
                label='Mean ± Std Dev')
    ax.fill_between(sizes,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.15, color='#1976D2')

    ax.set_xlabel('Number of Machines', fontsize=12)
    ax.set_ylabel('Solve Time (seconds)', fontsize=12)
    ax.set_title('CP-SAT Solver Scalability', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(sizes)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "scalability_plot.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


def plot_baseline_comparison():
    """Figure 2: Grouped bar chart of optimized vs baselines per size."""
    data = load_csv("baseline_comparison.csv")
    if not data:
        return

    # Average total_cost per (label, strategy) across seeds
    costs = defaultdict(lambda: defaultdict(list))
    for row in data:
        label = row["label"]
        strat = row["strategy"]
        cost = float(row["total_cost"])
        if cost < 1e12:  # skip infeasible
            costs[label][strat].append(cost)

    labels = [l for l in ["small", "med_easy", "med_hard", "large"] if l in costs]
    strategies = ["max_interval", "half_max", "analytical", "condition_based", "optimized"]
    strat_labels = ["Max Interval", "Half Max", "Analytical", "Condition-Based", "CP-SAT Optimized"]
    colors = ['#90CAF9', '#A5D6A7', '#FFE082', '#CE93D8', '#E53935']

    x = np.arange(len(labels))
    width = 0.15
    offsets = np.arange(len(strategies)) - len(strategies) / 2 + 0.5

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (strat, slabel, color) in enumerate(zip(strategies, strat_labels, colors)):
        vals = []
        for label in labels:
            if strat in costs[label] and costs[label][strat]:
                vals.append(np.mean(costs[label][strat]))
            else:
                vals.append(0)
        bars = ax.bar(x + offsets[i] * width, vals, width, label=slabel,
                      color=color, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Problem Size', fontsize=12)
    ax.set_ylabel('Average Total Cost ($)', fontsize=12)
    ax.set_title('Optimization Quality: CP-SAT vs Baseline Strategies',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    display_labels = {"small": "Small\n(6M/2K)", "med_easy": "Medium Easy\n(10M/4K)",
                      "med_hard": "Medium Hard\n(10M/2K)", "large": "Large\n(20M/5K)"}
    ax.set_xticklabels([display_labels.get(l, l) for l in labels])
    ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "baseline_bar_chart.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


def plot_savings_summary():
    """Figure 3: Bar chart showing % savings of optimized over best baseline."""
    data = load_csv("baseline_comparison.csv")
    if not data:
        return

    # For each (label, seed), compute savings
    runs = defaultdict(lambda: defaultdict(list))
    for row in data:
        key = (row["label"], row["seed"])
        strat = row["strategy"]
        cost = float(row["total_cost"])
        if cost < 1e12:
            runs[key][strat].append(cost)

    savings_by_label = defaultdict(list)
    for (label, seed), strats in runs.items():
        baseline_costs = [np.mean(strats[s]) for s in strats if s != "optimized" and strats[s]]
        opt_costs = strats.get("optimized", [])
        if baseline_costs and opt_costs:
            best_base = min(baseline_costs)
            opt = np.mean(opt_costs)
            if best_base > 0:
                savings_by_label[label].append((1 - opt / best_base) * 100)

    labels = [l for l in ["small", "med_easy", "med_hard", "large"] if l in savings_by_label]
    means = [np.mean(savings_by_label[l]) for l in labels]
    stds = [np.std(savings_by_label[l]) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    display = {"small": "Small", "med_easy": "Med Easy", "med_hard": "Med Hard", "large": "Large"}
    x = np.arange(len(labels))
    bars = ax.bar(x, means, 0.5, yerr=stds, capsize=5,
                  color=['#42A5F5', '#66BB6A', '#FFA726', '#EF5350'],
                  edgecolor='white', linewidth=0.5)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.5, f'{m:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=11)

    ax.set_xlabel('Problem Size', fontsize=12)
    ax.set_ylabel('Cost Savings (%)', fontsize=12)
    ax.set_title('CP-SAT Savings vs Best Baseline Strategy',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([display.get(l, l) for l in labels])
    ax.axhline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "savings_summary.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


def plot_montecarlo_summary():
    """Figure 4: Grouped bar chart of Monte Carlo mean cost + VaR95."""
    data = load_csv("montecarlo_results.csv")
    if not data:
        return

    # Average across seeds per (label, strategy)
    mean_costs = defaultdict(lambda: defaultdict(list))
    var95s = defaultdict(lambda: defaultdict(list))
    failures = defaultdict(lambda: defaultdict(list))

    for row in data:
        label = row["label"]
        strat = row["strategy"]
        mean_costs[label][strat].append(float(row["mean_cost"]))
        var95s[label][strat].append(float(row["var95"]))
        failures[label][strat].append(float(row["mean_failures"]))

    labels = [l for l in ["small", "med_easy", "med_hard"] if l in mean_costs]
    strategies = ["max_interval", "half_max", "analytical", "condition_based", "optimized"]
    strat_display = ["Max Int.", "Half Max", "Analytical", "Cond-Based", "Optimized"]
    colors = ['#90CAF9', '#A5D6A7', '#FFE082', '#CE93D8', '#E53935']

    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5), sharey=False)
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        x = np.arange(len(strategies))
        means = [np.mean(mean_costs[label].get(s, [0])) for s in strategies]
        v95 = [np.mean(var95s[label].get(s, [0])) for s in strategies]

        width = 0.35
        ax.bar(x - width/2, means, width, label='Mean Cost', color=colors, alpha=0.8,
               edgecolor='white')
        ax.bar(x + width/2, v95, width, label='VaR95 (Tail Risk)', color=colors, alpha=0.4,
               edgecolor='white', hatch='//')

        ax.set_xticks(x)
        ax.set_xticklabels(strat_display, rotation=30, ha='right', fontsize=8)
        display = {"small": "Small (6M)", "med_easy": "Med Easy (10M/4K)",
                   "med_hard": "Med Hard (10M/2K)"}
        ax.set_title(display.get(label, label), fontweight='bold')
        ax.set_ylabel('Cost ($)')

    axes[0].legend(fontsize=8)
    fig.suptitle('Monte Carlo: Mean Cost vs Tail Risk (VaR95)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, "montecarlo_summary.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"{'='*50}")
    print(f" Generating Chapter 4 Figures")
    print(f"{'='*50}")

    plot_scalability()
    plot_baseline_comparison()
    plot_savings_summary()
    plot_montecarlo_summary()

    print(f"\n All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()