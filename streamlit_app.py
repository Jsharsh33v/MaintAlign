"""
MaintAlign — Interactive Dashboard
=====================================
Clean, professional Streamlit UI for maintenance schedule optimization.

Launch:
    streamlit run streamlit_app.py
"""

import os
import tempfile
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.evaluator import compare_schedules
from core.baseline import ALL_STRATEGIES, fixed_interval_schedule

# ── Project imports ──────────────────────────────────────────
from core.instance import ProblemInstance
from core.solver import SolverResult, solve
from core.validators import (
    MaintAlignError,
    validate_instance,
    validate_solver_params,
)
from utils.csv_loader import load_instance
from utils.generator import (
    generate_factory,
    generate_industrial,
    generate_large,
    generate_medium_easy,
    generate_medium_hard,
    generate_small,
    generate_tiny,
    generate_xl,
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MaintAlign — Maintenance Optimizer",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — clean, minimal, professional ────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* Global font — do NOT use [class*="st-"] as it overrides Material icon fonts */
    html, body, p, h1, h2, h3, h4, h5, h6, span, div, label, button, input, select, textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Restore Material Symbols font for icon elements */
    .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* ═══ Metric cards — flat, clean, no truncation ═══ */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 12px 14px;
    }
    div[data-testid="stMetric"] label {
        color: #99aabb !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 500 !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #e0e0e0 !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }

    /* ═══ Tab styling ═══ */
    button[data-baseweb="tab"] {
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.3px;
    }

    /* ═══ Sidebar — higher contrast headers ═══ */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1.1rem !important;
        color: #e0e0e0 !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #42a5f5 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.4rem !important;
        font-weight: 600 !important;
    }

    /* ═══ Expander — clean, hide broken icon text ═══ */
    details[data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 8px !important;
        background: rgba(255,255,255,0.02) !important;
    }
    details[data-testid="stExpander"] summary span[class*="material"] {
        display: none !important;
    }

    /* ═══ Table ═══ */
    .dataframe {
        font-size: 0.85rem !important;
    }

    /* ═══ Plotly containers ═══ */
    .stPlotlyChart {
        border-radius: 8px;
        overflow: hidden;
    }

    /* ═══ Main header ═══ */
    .main-header {
        border-bottom: 1px solid rgba(255,255,255,0.08);
        padding: 16px 0 20px 0;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
        color: #e0e0e0;
        letter-spacing: -0.3px;
    }
    .main-header p {
        color: #778899;
        margin: 4px 0 0 0;
        font-size: 0.88rem;
        font-weight: 400;
    }

    /* ═══ Status badge ═══ */
    .status-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .status-optimal { background: rgba(76,175,80,0.15); color: #81c784; border: 1px solid rgba(76,175,80,0.3); }
    .status-feasible { background: rgba(255,167,38,0.15); color: #ffb74d; border: 1px solid rgba(255,167,38,0.3); }
    .status-unknown { background: rgba(158,158,158,0.15); color: #bdbdbd; border: 1px solid rgba(158,158,158,0.3); }

    /* ═══ Instance info bar ═══ */
    .instance-info {
        color: #778899;
        font-size: 0.82rem;
        letter-spacing: 0.3px;
    }

    /* ═══ Feature card on landing ═══ */
    .feature-card {
        text-align: center;
        padding: 28px 16px;
        background: rgba(255,255,255,0.02);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.06);
        transition: border-color 0.2s;
    }
    .feature-card:hover {
        border-color: rgba(66,165,245,0.3);
    }
    .feature-card .icon-circle {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: rgba(66,165,245,0.12);
        border: 1px solid rgba(66,165,245,0.25);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 12px;
    }
    .feature-card .icon-circle svg {
        width: 20px;
        height: 20px;
        stroke: #42a5f5;
        fill: none;
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .feature-card b {
        font-size: 0.9rem;
        color: #ccc;
        display: block;
        margin-bottom: 4px;
    }
    .feature-card p {
        color: #778899;
        font-size: 0.82rem;
        margin-top: 4px;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


# ── Color Palette ────────────────────────────────────────────
CHAIN_COLORS = [
    "#1976D2", "#388E3C", "#F57C00", "#7B1FA2",
    "#C62828", "#00838F", "#4E342E", "#283593",
]
STANDALONE_COLOR = "#4CAF50"
MAINT_COLOR = "#E53935"
PLOTLY_BG = "rgba(0,0,0,0)"
PLOTLY_GRID = "rgba(255,255,255,0.06)"
PLOTLY_TEXT = "#bbb"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PLOTLY_BG,
    plot_bgcolor=PLOTLY_BG,
    font=dict(color=PLOTLY_TEXT, family="Inter, -apple-system, sans-serif", size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0.2)", bordercolor="rgba(255,255,255,0.08)"),
)


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def build_gantt_figure(instance: ProblemInstance, result: SolverResult,
                       title: str = "Optimized Schedule") -> go.Figure:
    """Interactive Plotly Gantt chart."""
    M = instance.num_machines
    H = instance.horizon

    # Machine → color map
    m_color = {}
    m_chain_name = {}
    for c in instance.chains:
        col = CHAIN_COLORS[c.id % len(CHAIN_COLORS)]
        for mid in c.machine_ids:
            m_color[mid] = col
            m_chain_name[mid] = c.name
    for m in instance.machines:
        if m.id not in m_color:
            m_color[m.id] = STANDALONE_COLOR
            m_chain_name[m.id] = "Standalone"

    fig = go.Figure()

    for m_idx in range(M):
        machine = instance.machines[m_idx]
        y_label = f"{machine.name}"
        starts = sorted(result.machine_schedules.get(m_idx, []))
        d = machine.maintenance_duration
        col = m_color[m_idx]

        # Production bars (gaps between PMs)
        prev = 0
        for s in starts:
            if s > prev:
                fig.add_trace(go.Bar(
                    x=[s - prev], y=[y_label], orientation="h",
                    base=prev, marker=dict(color=col, opacity=0.45,
                                           line=dict(width=0)),
                    hovertemplate=(f"<b>{machine.name}</b><br>"
                                  f"Production: t={prev}→{s}<br>"
                                  f"Chain: {m_chain_name[m_idx]}"
                                  "<extra></extra>"),
                    showlegend=False,
                ))
            prev = s + d
        if prev < H:
            fig.add_trace(go.Bar(
                x=[H - prev], y=[y_label], orientation="h",
                base=prev, marker=dict(color=col, opacity=0.45,
                                       line=dict(width=0)),
                showlegend=False,
                hovertemplate=(f"<b>{machine.name}</b><br>"
                               f"Production: t={prev}→{H}"
                               "<extra></extra>"),
            ))

        # Maintenance bars
        for j, s in enumerate(starts):
            fig.add_trace(go.Bar(
                x=[d], y=[y_label], orientation="h",
                base=s, marker=dict(color=MAINT_COLOR, opacity=0.9,
                                    line=dict(width=0.5, color="white")),
                showlegend=False,
                hovertemplate=(f"<b>PM #{j+1}</b> — {machine.name}<br>"
                               f"Start: t={s}<br>"
                               f"Duration: {d} periods<br>"
                               f"PM Cost: ${machine.pm_cost:,}"
                               "<extra></extra>"),
            ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=14)),
        barmode="overlay",
        xaxis=dict(title="Time Period", range=[0, H], gridcolor=PLOTLY_GRID,
                   zeroline=False, dtick=max(1, H // 15)),
        yaxis=dict(autorange="reversed", gridcolor=PLOTLY_GRID, zeroline=False),
        height=max(350, M * 42 + 100),
        bargap=0.3,
    )
    return fig


def build_cost_comparison(instance, baselines, optimized) -> go.Figure:
    """Stacked bar chart: baselines vs optimized."""
    names = list(baselines.keys()) + ["Optimized"]
    pm_costs = [baselines[s].total_pm_cost for s in baselines] + [optimized.total_pm_cost]
    fail_costs = [baselines[s].total_failure_cost for s in baselines] + [optimized.total_failure_cost]
    prod_costs = [baselines[s].total_production_loss for s in baselines] + [optimized.total_production_loss]
    retool_costs = [baselines[s].total_retooling_cost for s in baselines] + [optimized.total_retooling_cost]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="PM Cost", x=names, y=pm_costs,
                         marker_color="#42A5F5"))
    fig.add_trace(go.Bar(name="Failure Cost", x=names, y=fail_costs,
                         marker_color="#EF5350"))
    fig.add_trace(go.Bar(name="Production Loss", x=names, y=prod_costs,
                         marker_color="#FFA726"))
    fig.add_trace(go.Bar(name="Retooling", x=names, y=retool_costs,
                         marker_color="#AB47BC"))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        title=dict(text="Cost Comparison: Baselines vs Optimized", font=dict(size=14)),
        xaxis=dict(gridcolor=PLOTLY_GRID, zeroline=False),
        yaxis=dict(title="Total Cost ($)", gridcolor=PLOTLY_GRID, zeroline=False),
        height=420,
    )
    return fig


def build_cost_donut(optimized: SolverResult) -> go.Figure:
    """Donut chart showing cost breakdown."""
    labels = ["PM Cost", "Failure Cost", "Production Loss", "Retooling"]
    values = [
        optimized.total_pm_cost,
        optimized.total_failure_cost,
        optimized.total_production_loss,
        optimized.total_retooling_cost,
    ]
    colors = ["#42A5F5", "#EF5350", "#FFA726", "#AB47BC"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(width=2, color="#0e1117")),
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Cost Breakdown", font=dict(size=14)),
        height=380,
        showlegend=False,
        annotations=[dict(
            text=f"${optimized.objective_value:,.0f}",
            x=0.5, y=0.5, font=dict(size=16, color="#ddd", family="Inter"),
            showarrow=False,
        )],
    )
    return fig


def build_technician_utilization(instance, result) -> go.Figure:
    """Technician utilization over time."""
    H = instance.horizon
    K = instance.num_technicians
    usage = [0] * H

    for task in result.tasks:
        for t in range(task.start_time, task.end_time):
            if t < H:
                usage[t] += 1

    times = list(range(H))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=usage, fill="tozeroy",
        fillcolor="rgba(66,165,245,0.15)",
        line=dict(color="#42A5F5", width=2),
        name="Usage",
        hovertemplate="t=%{x}<br>Technicians: %{y}<extra></extra>",
    ))
    fig.add_hline(y=K, line_dash="dash", line_color="#EF5350",
                  annotation_text=f"Capacity ({K})",
                  annotation_font_color="#EF5350")

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Technician Utilization", font=dict(size=14)),
        xaxis=dict(title="Time Period", gridcolor=PLOTLY_GRID, zeroline=False),
        yaxis=dict(title="Technicians In Use", range=[0, K + 1],
                   gridcolor=PLOTLY_GRID, zeroline=False),
        height=320,
    )
    return fig


def build_monte_carlo_violin(eval_results) -> go.Figure:
    """Violin plot of cost distributions from Monte Carlo."""
    fig = go.Figure()
    colors = ["#42A5F5", "#66BB6A", "#FFA726", "#AB47BC", "#EF5350"]

    for i, (name, er) in enumerate(eval_results.items()):
        col = colors[i % len(colors)]
        fig.add_trace(go.Violin(
            y=er.all_costs, name=name,
            box_visible=True, meanline_visible=True,
            fillcolor=col, opacity=0.6,
            line_color=col,
            hovertemplate="Cost: $%{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Monte Carlo Cost Distributions", font=dict(size=14)),
        xaxis=dict(gridcolor=PLOTLY_GRID, zeroline=False),
        yaxis=dict(title="Total Cost ($)", gridcolor=PLOTLY_GRID, zeroline=False),
        height=450,
        showlegend=False,
    )
    return fig


def build_chain_topology(instance: ProblemInstance) -> go.Figure:
    """Simple chain visualization bar chart."""
    if not instance.chains:
        return None

    chain_names = []
    chain_values = []
    chain_sizes = []
    retool_costs = []

    for c in instance.chains:
        chain_names.append(c.name)
        chain_values.append(c.chain_value)
        chain_sizes.append(len(c.machine_ids))
        retool_costs.append(c.retooling_cost)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=chain_names, y=chain_values,
        name="Chain Value ($/period)",
        marker_color="#42A5F5",
        text=[f"{v}" for v in chain_values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Value: $%{y}/period<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=chain_names, y=retool_costs,
        name="Retooling Cost ($)",
        marker_color="#FFA726",
        text=[f"{v}" for v in retool_costs],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Retooling: $%{y}<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        title=dict(text="Production Chain Values", font=dict(size=14)),
        xaxis=dict(gridcolor=PLOTLY_GRID, zeroline=False),
        yaxis=dict(title="Cost ($)", gridcolor=PLOTLY_GRID, zeroline=False),
        height=350,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════

# Instance presets
INSTANCE_PRESETS = {
    "Tiny (3M/1K/12T)":         generate_tiny,
    "Small (6M/2K/20T)":        generate_small,
    "Medium Easy (10M/4K/30T)":  generate_medium_easy,
    "Medium Hard (10M/2K/30T)":  generate_medium_hard,
    "Large (20M/5K/50T)":       generate_large,
    "XL (40M/8K/80T)":          generate_xl,
    "Industrial (50M/10K/60T)":  generate_industrial,
    "Factory (100M/15K/60T)":    generate_factory,
}

st.sidebar.markdown("## MaintAlign")
st.sidebar.caption("Maintenance Schedule Optimizer")
st.sidebar.divider()

# Data source
st.sidebar.markdown("### ▶ Data Source")
data_source = st.sidebar.radio(
    "Choose input:", ["Generated Instance", "CSV File"],
    horizontal=True, label_visibility="collapsed",
)

if data_source == "Generated Instance":
    preset_name = st.sidebar.selectbox(
        "Instance Preset",
        list(INSTANCE_PRESETS.keys()),
        index=2,  # default to Medium Easy
        help="Same presets used by `python main.py --full`",
    )
    seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=999, value=42)
    st.sidebar.success(f"Will generate: {preset_name}")
else:
    uploaded_machines = st.sidebar.file_uploader(
        "Machines CSV", type=["csv"], key="machines_upload"
    )
    uploaded_chains = st.sidebar.file_uploader(
        "Chains CSV (optional)", type=["csv"], key="chains_upload"
    )

st.sidebar.divider()

# Solver settings
st.sidebar.markdown("### ▶ Solver Settings")
time_limit = st.sidebar.slider("Time Limit (seconds)", 5, 120, 30, step=5)
block_weekends = st.sidebar.toggle("Block Weekends", value=False)
repair_factor = st.sidebar.slider(
    "Repair Factor", 0.5, 1.0, 1.0, step=0.05,
    help="1.0 = perfect repair, 0.7 = PM restores to 70% of new"
)

st.sidebar.divider()

# Run button
run_clicked = st.sidebar.button(
    "▶ Run Optimization", type="primary", use_container_width=True
)


# ═══════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>MaintAlign Dashboard</h1>
    <p>Maintenance scheduling powered by Constraint Programming and Weibull reliability models</p>
</div>
""", unsafe_allow_html=True)


# ── Run logic ────────────────────────────────────────────────
if run_clicked:
    with st.status("**Optimizing maintenance schedule...**", expanded=True) as status:
        # Step 1: Load / generate instance
        st.write("Generating problem instance...")
        inst = None
        tmp_machines = None
        tmp_chains = None
        try:
            if data_source == "Generated Instance":
                gen_fn = INSTANCE_PRESETS[preset_name]
                inst = gen_fn(seed=seed)
            else:
                if uploaded_machines is None:
                    st.error("Please upload a machines CSV file first.")
                    st.stop()
                # Write uploaded bytes to temp files
                m_bytes = uploaded_machines.getvalue()
                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                    f.write(m_bytes.decode("utf-8"))
                    tmp_machines = f.name
                if uploaded_chains is not None:
                    c_bytes = uploaded_chains.getvalue()
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                        f.write(c_bytes.decode("utf-8"))
                        tmp_chains = f.name
                inst = load_instance(tmp_machines, tmp_chains)
            validate_instance(inst)
        except UnicodeDecodeError:
            st.error("CSV file is not valid UTF-8. Please re-save it as UTF-8 and try again.")
            st.stop()
        except FileNotFoundError as e:
            st.error(f"File not found: {e}")
            st.stop()
        except MaintAlignError as e:
            st.error(f"Invalid instance: {e}")
            st.stop()
        except ValueError as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error loading instance: {e}")
            st.stop()
        finally:
            # Clean up temp files
            for tmp in (tmp_machines, tmp_chains):
                if tmp is not None:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

        st.write(f"Instance loaded: **{inst.num_machines}** machines, "
                 f"**{inst.num_technicians}** technicians, "
                 f"**{inst.horizon}** periods, "
                 f"**{len(inst.chains)}** chains")

        # Apply options
        if block_weekends:
            blocked = [t for t in range(inst.horizon) if t % 7 in (5, 6)]
            inst.blocked_periods = blocked
            st.write(f"Blocked **{len(blocked)}** weekend periods")

        if repair_factor < 1.0:
            for m in inst.machines:
                m.repair_factor = repair_factor
            st.write(f"Repair factor set to **{repair_factor:.0%}**")

        # Step 2: Compute baselines
        st.write("Computing baseline strategies...")
        baselines = {}
        for strat in ALL_STRATEGIES:
            baselines[strat] = fixed_interval_schedule(inst, strat)
        best_b_name = min(baselines, key=lambda k: baselines[k].objective_value)
        hint = baselines[best_b_name].machine_schedules
        st.write(f"{len(baselines)} baselines computed. "
                 f"Best: **{best_b_name}** (${baselines[best_b_name].objective_value:,.0f})")

        # Step 3: Solve with CP-SAT
        st.write(f"Running CP-SAT solver (time limit: **{time_limit}s**)...")
        t0 = time.time()
        try:
            validate_solver_params(time_limit_seconds=time_limit, repair_factor=repair_factor)
            result = solve(inst, time_limit_seconds=time_limit, hint_schedule=hint)
        except MaintAlignError as e:
            st.error(f"Bad solver parameter: {e}")
            st.stop()
        except Exception as e:
            st.error(
                f"Solver failed: {e}\n\n"
                "Try reducing the problem size, increasing the time limit, "
                "or enabling decomposition for large instances."
            )
            st.stop()
        solve_elapsed = time.time() - t0

        # Check solver status
        if result.status not in ("OPTIMAL", "FEASIBLE"):
            st.warning(
                f"Solver returned status **{result.status}**. "
                "No feasible schedule was found. Try increasing the time limit "
                "or reducing problem complexity."
            )
            st.stop()
        if result.status == "FEASIBLE":
            st.warning(
                "Solver found a feasible solution, but could not prove optimality "
                "within the time limit. The result is an upper bound, not a proven optimum."
            )

        st.write(f"Solver finished in **{solve_elapsed:.1f}s** — "
                 f"Status: **{result.status}** — "
                 f"Cost: **${result.objective_value:,.0f}**")

        status.update(label="**Optimization complete**", state="complete", expanded=False)

    # Store results in session state
    st.session_state["solved"] = True
    st.session_state["inst"] = inst
    st.session_state["baselines"] = baselines
    st.session_state["result"] = result
    st.session_state["best_b_name"] = best_b_name
    st.session_state["solve_elapsed"] = solve_elapsed


# ── Display results ──────────────────────────────────────────
if st.session_state.get("solved"):
    inst = st.session_state["inst"]
    baselines = st.session_state["baselines"]
    result = st.session_state["result"]
    best_b_name = st.session_state["best_b_name"]
    solve_elapsed = st.session_state["solve_elapsed"]

    best_b_cost = baselines[best_b_name].objective_value
    savings_pct = (1 - result.objective_value / best_b_cost) * 100 if best_b_cost > 0 else 0

    # Status badge
    status_class = "status-optimal" if "OPTIMAL" in result.status else (
        "status-feasible" if "FEASIBLE" in result.status else "status-unknown"
    )

    # ── Top metrics row ──────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Optimized Cost", f"${result.objective_value:,.0f}")
    col2.metric("Best Baseline", f"${best_b_cost:,.0f}", help=best_b_name)
    col3.metric("Savings", f"{savings_pct:+.1f}%")
    col4.metric("Tasks Scheduled", f"{len(result.tasks)}")
    col5.metric("Solve Time", f"{solve_elapsed:.1f}s")

    st.markdown(
        f'<p style="text-align:center; margin-top:4px;">'
        f'<span class="status-badge {status_class}">{result.status}</span> '
        f'&nbsp; <span class="instance-info">{inst.num_machines} machines · '
        f'{inst.num_technicians} technicians · {inst.horizon} periods · '
        f'{len(inst.chains)} chains</span></p>',
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Schedule", "Cost Analysis", "Risk Analysis", "Factory"
    ])

    # ────────────────── TAB 1: Schedule ──────────────────────
    with tab1:
        st.plotly_chart(
            build_gantt_figure(inst, result, "Optimized Schedule"),
            use_container_width=True,
        )

        # Comparison Gantt
        with st.expander("Compare with best baseline"):
            st.plotly_chart(
                build_gantt_figure(inst, baselines[best_b_name],
                                  f"Baseline: {best_b_name}"),
                use_container_width=True,
            )

        # Machine details table
        with st.expander("Machine Specifications"):
            df = pd.DataFrame([{
                "Name": m.name,
                "Duration": m.maintenance_duration,
                "PM Cost ($)": m.pm_cost,
                "CM Cost ($)": m.cm_cost,
                "Prod Value ($)": m.production_value,
                "Beta": m.weibull_beta,
                "Eta": m.weibull_eta,
                "Max Interval": m.max_interval,
                "t* (optimal)": round(m.optimal_interval_analytical(), 1),
            } for m in inst.machines])
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Schedule details
        with st.expander("Schedule Details"):
            sched_rows = []
            for task in sorted(result.tasks, key=lambda t: (t.machine_id, t.start_time)):
                m = inst.machines[task.machine_id]
                chain = inst.get_chain_for_machine(task.machine_id)
                sched_rows.append({
                    "Machine": m.name,
                    "PM #": task.task_index + 1,
                    "Start": task.start_time,
                    "End": task.end_time,
                    "PM Cost ($)": task.cost_pm,
                    "Prod Loss ($)": round(task.cost_prod_loss, 0),
                    "Chain": chain.name if chain else "—",
                })
            st.dataframe(pd.DataFrame(sched_rows), use_container_width=True, hide_index=True)

    # ────────────────── TAB 2: Cost Analysis ─────────────────
    with tab2:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(
                build_cost_comparison(inst, baselines, result),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                build_cost_donut(result),
                use_container_width=True,
            )

        # Baseline comparison table
        st.markdown("#### Strategy Comparison")
        comp_rows = []
        for name, b in baselines.items():
            sav = (1 - result.objective_value / b.objective_value) * 100 if b.objective_value > 0 else 0
            comp_rows.append({
                "Strategy": name,
                "Total Cost ($)": f"${b.objective_value:,.0f}",
                "PM ($)": f"${b.total_pm_cost:,.0f}",
                "Failure ($)": f"${b.total_failure_cost:,.0f}",
                "Tasks": len(b.tasks),
                "vs Optimized": f"{sav:+.1f}%",
            })
        comp_rows.append({
            "Strategy": "Optimized",
            "Total Cost ($)": f"${result.objective_value:,.0f}",
            "PM ($)": f"${result.total_pm_cost:,.0f}",
            "Failure ($)": f"${result.total_failure_cost:,.0f}",
            "Tasks": len(result.tasks),
            "vs Optimized": "—",
        })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # ────────────────── TAB 3: Risk Analysis ─────────────────
    with tab3:
        st.markdown("#### Monte Carlo Simulation")
        st.caption("Test how each schedule survives random machine failures (Weibull model)")

        mc_col1, mc_col2 = st.columns([1, 3])
        with mc_col1:
            n_sims = st.slider("Simulations", 100, 2000, 500, step=100)
            run_mc = st.button("Run Simulation", type="primary",
                               use_container_width=True)
        with mc_col2:
            st.info(
                "Monte Carlo tests each schedule against random breakdowns. "
                "VaR95 = average cost in the worst 5% of scenarios (tail risk)."
            )

        if run_mc or st.session_state.get("mc_results"):
            if run_mc:
                try:
                    validate_solver_params(n_sims=n_sims)
                    with st.spinner(f"Running {n_sims} simulations per strategy..."):
                        schedules = {
                            name: b.machine_schedules for name, b in baselines.items()
                        }
                        schedules["Optimized"] = result.machine_schedules
                        mc_results = compare_schedules(inst, schedules, n_sims=n_sims)
                    st.session_state["mc_results"] = mc_results
                except MaintAlignError as e:
                    st.error(f"Invalid simulation parameter: {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"Monte Carlo simulation failed: {e}")
                    st.stop()

            mc_results = st.session_state["mc_results"]

            # Violin plot
            st.plotly_chart(
                build_monte_carlo_violin(mc_results),
                use_container_width=True,
            )

            # Stats table
            mc_rows = []
            for name, er in mc_results.items():
                mc_rows.append({
                    "Strategy": name,
                    "Mean ($)": f"${er.mean_cost:,.0f}",
                    "Std ($)": f"${er.std_cost:,.0f}",
                    "Median ($)": f"${er.median_cost:,.0f}",
                    "VaR95 ($)": f"${er.var95:,.0f}",
                    "Avg Failures": f"{er.mean_failures:.1f}",
                    "Avg Downtime": f"{er.mean_downtime:.1f}",
                })
            st.dataframe(pd.DataFrame(mc_rows), use_container_width=True,
                         hide_index=True)

            # Risk-return scatter
            fig_scatter = go.Figure()
            colors_mc = ["#42A5F5", "#66BB6A", "#FFA726", "#AB47BC", "#EF5350"]
            for i, (name, er) in enumerate(mc_results.items()):
                is_opt = name == "Optimized"
                fig_scatter.add_trace(go.Scatter(
                    x=[er.mean_cost], y=[er.var95],
                    mode="markers+text",
                    text=[name],
                    textposition="top center",
                    marker=dict(
                        size=18 if is_opt else 10,
                        color=colors_mc[i % len(colors_mc)],
                        symbol="diamond" if is_opt else "circle",
                        line=dict(width=1.5, color="white") if is_opt else dict(width=0),
                    ),
                    showlegend=False,
                    hovertemplate=(f"<b>{name}</b><br>"
                                  f"Mean: $%{{x:,.0f}}<br>"
                                  f"VaR95: $%{{y:,.0f}}<extra></extra>"),
                ))
            fig_scatter.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Risk vs Cost (lower-left = better)", font=dict(size=14)),
                xaxis_title="Mean Cost ($)",
                yaxis_title="VaR95 — Tail Risk ($)",
                height=400,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ────────────────── TAB 4: Factory ───────────────────────
    with tab4:
        f1, f2 = st.columns(2)

        with f1:
            st.markdown("#### Production Chains")
            if inst.chains:
                for c in inst.chains:
                    machines_in_chain = [inst.machines[mid].name for mid in c.machine_ids]
                    col = CHAIN_COLORS[c.id % len(CHAIN_COLORS)]
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.03); '
                        f'border-left:3px solid {col}; padding:12px 16px; '
                        f'border-radius:6px; margin-bottom:8px;">'
                        f'<b style="color:{col}; font-size:0.9rem;">{c.name}</b>'
                        f'<br><span style="color:#8899aa;font-size:0.82rem;">'
                        f'{" → ".join(machines_in_chain)}</span>'
                        f'<br><span style="color:#778899; font-size:0.8rem;">Value: ${c.chain_value}/period '
                        f'&nbsp; Retooling: ${c.retooling_cost}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Standalone machines
                standalone = [m for m in inst.machines if inst.is_standalone(m.id)]
                if standalone:
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.03); '
                        f'border-left:3px solid {STANDALONE_COLOR}; padding:12px 16px; '
                        f'border-radius:6px; margin-bottom:8px;">'
                        f'<b style="color:{STANDALONE_COLOR}; font-size:0.9rem;">Standalone</b>'
                        f'<br><span style="color:#8899aa;font-size:0.82rem;">'
                        f'{", ".join(m.name for m in standalone)}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No production chains defined. All machines are independent.")

        with f2:
            chain_fig = build_chain_topology(inst)
            if chain_fig:
                st.plotly_chart(chain_fig, use_container_width=True)
            else:
                st.info("Chain topology chart requires at least one production chain.")

        # Technician utilization
        st.plotly_chart(
            build_technician_utilization(inst, result),
            use_container_width=True,
        )

        # Chain cost breakdown
        if inst.chains and result.chain_costs:
            st.markdown("#### Per-Chain Cost Breakdown")
            chain_rows = []
            for cid, cc in result.chain_costs.items():
                c = inst.chains[int(cid)] if int(cid) < len(inst.chains) else None
                chain_rows.append({
                    "Chain": c.name if c else f"Chain {cid}",
                    "Production Loss ($)": f"${cc.get('prod_loss', 0):,.0f}",
                    "Retooling ($)": f"${cc.get('retooling', 0):,.0f}",
                    "Events": cc.get("num_events", 0),
                })
            st.dataframe(pd.DataFrame(chain_rows), use_container_width=True,
                         hide_index=True)

else:
    # Landing page when no results yet
    st.markdown("---")
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown("""
        <div style="text-align:center; padding: 50px 20px;">
            <h2 style="color:#c0c8d0; font-weight:600; font-size:1.4rem;">
                Welcome to MaintAlign
            </h2>
            <p style="color:#667788; font-size:0.95rem; max-width:480px; margin:8px auto 0;">
                Upload your factory data or select a built-in instance,
                configure the solver, and click <b>Run Optimization</b>
                to generate an optimal maintenance schedule.
            </p>
            <br>
            <p style="color:#556677; font-size:0.85rem;">
                Use the sidebar to get started
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Quick feature overview
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown("""
            <div class="feature-card">
                <div class="icon-circle">
                    <svg viewBox="0 0 24 24"><path d="M12 20V10M18 20V4M6 20v-4"/></svg>
                </div>
                <b>Smart Scheduling</b>
                <p>CP-SAT constraint solver finds optimal PM timing under resource limits</p>
            </div>
            """, unsafe_allow_html=True)
        with f2:
            st.markdown("""
            <div class="feature-card">
                <div class="icon-circle">
                    <svg viewBox="0 0 24 24"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                </div>
                <b>Risk Analysis</b>
                <p>Monte Carlo simulation with Weibull failure modeling</p>
            </div>
            """, unsafe_allow_html=True)
        with f3:
            st.markdown("""
            <div class="feature-card">
                <div class="icon-circle">
                    <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
                </div>
                <b>Chain-Aware</b>
                <p>Production chain dependencies and opportunistic grouping</p>
            </div>
            """, unsafe_allow_html=True)

