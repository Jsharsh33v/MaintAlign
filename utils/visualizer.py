"""
MaintAlign - Visualizer (v2: Chain-Aware)
==========================================
Gantt charts with chain grouping, cost breakdowns, utilization plots,
and per-chain cost analysis.
"""

import logging

import matplotlib

matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Use a modern style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.25,
})

# Color palette for chains
CHAIN_COLORS = [
    '#1976D2', '#388E3C', '#F57C00', '#7B1FA2',
    '#C62828', '#00838F', '#4E342E', '#283593',
]
STANDALONE_COLOR = '#4CAF50'
MAINT_COLOR = '#E53935'


def plot_gantt(instance, result, title=None, save_path=None, figsize=None,
               show=False):
    """
    Gantt chart with chain grouping.
    Machines in the same chain share a color.
    Standalone machines are green.
    Maintenance blocks are red.
    """
    M = instance.num_machines
    H = instance.horizon

    if figsize is None:
        figsize = (max(12, H * 0.35), max(4, M * 0.55 + 1))

    fig, ax = plt.subplots(figsize=figsize)

    # Build machine → color map
    m_color = {}
    m_label_suffix = {}
    for c in instance.chains:
        col = CHAIN_COLORS[c.id % len(CHAIN_COLORS)]
        for mid in c.machine_ids:
            m_color[mid] = col
            m_label_suffix[mid] = f" [{c.name}]"
    for m in instance.machines:
        if m.id not in m_color:
            m_color[m.id] = STANDALONE_COLOR
            m_label_suffix[m.id] = ""

    labels = []
    for m_idx in range(M):
        machine = instance.machines[m_idx]
        y = M - 1 - m_idx
        labels.append(f"M{m_idx}: {machine.name}{m_label_suffix[m_idx]}")
        starts = sorted(result.machine_schedules.get(m_idx, []))
        d = machine.maintenance_duration
        col = m_color[m_idx]

        # Production bars
        prev = 0
        for s in starts:
            if s > prev:
                ax.barh(y, s - prev, left=prev, height=0.6,
                       color=col, alpha=0.55, edgecolor='white', lw=0.5)
            prev = s + d
        if prev < H:
            ax.barh(y, H - prev, left=prev, height=0.6,
                   color=col, alpha=0.55, edgecolor='white', lw=0.5)

        # Maintenance bars
        for s in starts:
            ax.barh(y, d, left=s, height=0.6,
                   color=MAINT_COLOR, alpha=0.9, edgecolor='white', lw=0.5)
            ax.text(s + d / 2, y, 'PM', ha='center', va='center',
                   fontsize=6, color='white', fontweight='bold')

    ax.set_yticks(range(M))
    ax.set_yticklabels(list(reversed(labels)), fontsize=7)
    ax.set_xlabel('Time Period')
    ax.set_xlim(0, H)
    ax.set_ylim(-0.5, M - 0.5)

    if title is None:
        title = f"MaintAlign: {instance.name} ({result.status})"
    ax.set_title(f"{title}\nTotal Cost: ${result.objective_value:,.2f}",
                 fontsize=10, fontweight='bold')

    # Legend
    handles = [mpatches.Patch(color=MAINT_COLOR, alpha=0.9, label='Maintenance')]
    for c in instance.chains:
        col = CHAIN_COLORS[c.id % len(CHAIN_COLORS)]
        handles.append(mpatches.Patch(color=col, alpha=0.55, label=c.name))
    if instance.standalone_machines:
        handles.append(mpatches.Patch(color=STANDALONE_COLOR, alpha=0.55,
                                      label='Standalone'))
    ax.legend(handles=handles, loc='upper right', fontsize=7, ncol=2)
    ax.grid(axis='x', alpha=0.3, lw=0.5)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved Gantt chart → %s", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_cost_comparison(instance, baseline_results, optimized,
                         save_path=None, show=False):
    """Stacked bar chart: baseline strategies vs optimized."""
    labels = list(baseline_results.keys()) + ["Optimized"]
    results = list(baseline_results.values()) + [optimized]

    cats = ['PM Cost', 'Prod Loss', 'Retooling', 'Failure']
    colors = ['#2196F3', '#FF9800', '#9C27B0', '#F44336']
    data = [
        [r.total_pm_cost for r in results],
        [r.total_production_loss for r in results],
        [r.total_retooling_cost for r in results],
        [r.total_failure_cost for r in results],
    ]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(labels))

    for vals, col, name in zip(data, colors, cats, strict=False):
        ax.bar(x, vals, 0.6, bottom=bottom, label=name, color=col, alpha=0.85)
        bottom += np.array(vals)

    # Total cost labels on top
    for i, total in enumerate(bottom):
        ax.text(i, total + total * 0.01, f'${total:,.0f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Cost ($)')
    ax.set_title(f'Cost Comparison: {instance.name}', fontweight='bold')
    ax.legend(loc='upper left')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved cost comparison → %s", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_technician_utilization(instance, result, save_path=None, show=False):
    """Line chart: technician usage over time with capacity line."""
    H = instance.horizon
    K = instance.num_technicians
    usage = [0] * H

    for task in result.tasks:
        for t in range(task.start_time, task.end_time):
            if t < H:
                usage[t] += 1

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.fill_between(range(H), usage, alpha=0.4, color='#2196F3')
    ax.step(range(H), usage, where='mid', color='#1565C0', lw=1.5)
    ax.axhline(K, color='#E53935', ls='--', lw=1.5, label=f'Capacity (K={K})')

    ax.set_xlabel('Time Period')
    ax.set_ylabel('Technicians')
    ax.set_title(f'Technician Utilization: {instance.name}', fontweight='bold')
    ax.set_xlim(0, H)
    ax.set_ylim(0, K + 1)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved technician utilization → %s", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_sensitivity(results_by_param, param_name, save_path=None, show=False):
    """
    Line chart showing how total cost varies with a parameter.
    results_by_param: dict of {param_value: SolverResult}
    """
    params = sorted(results_by_param.keys())
    costs = [results_by_param[p].objective_value for p in params]
    pm = [results_by_param[p].total_pm_cost for p in params]
    fail = [results_by_param[p].total_failure_cost for p in params]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(params, costs, 'ko-', lw=2, label='Total Cost')
    ax.plot(params, pm, 's--', color='#2196F3', label='PM + Prod + Retool')
    ax.plot(params, fail, '^--', color='#F44336', label='Failure Cost')

    ax.set_xlabel(param_name)
    ax.set_ylabel('Cost ($)')
    ax.set_title(f'Sensitivity: Cost vs {param_name}', fontweight='bold')
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved sensitivity plot → %s", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_chain_breakdown(instance, result, save_path=None, show=False):
    """
    Per-chain cost breakdown: grouped bar chart showing prod loss,
    retooling, and event count for each chain.
    """
    if not result.chain_costs:
        logger.info("No chain costs to plot")
        return None

    cids = sorted(result.chain_costs.keys())
    if not cids:
        return None

    names = [instance.chains[c].name for c in cids]
    prod_loss = [result.chain_costs[c]["prod_loss"] for c in cids]
    retooling = [result.chain_costs[c]["retooling"] for c in cids]
    events = [result.chain_costs[c]["num_events"] for c in cids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: stacked costs
    x = np.arange(len(names))
    w = 0.5
    ax1.bar(x, prod_loss, w, label='Production Loss', color='#FF9800', alpha=0.85)
    ax1.bar(x, retooling, w, bottom=prod_loss, label='Retooling',
            color='#9C27B0', alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.set_ylabel('Cost ($)')
    ax1.set_title('Chain Costs Breakdown', fontweight='bold')
    ax1.legend()

    # Right: event count
    colors = [CHAIN_COLORS[c % len(CHAIN_COLORS)] for c in cids]
    ax2.bar(x, events, w, color=colors, alpha=0.85, edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylabel('Number of PM Events')
    ax2.set_title('Chain PM Events', fontweight='bold')
    for i, ev in enumerate(events):
        ax2.text(i, ev + 0.1, str(ev), ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved chain breakdown → %s", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
