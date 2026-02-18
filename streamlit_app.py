"""
MaintAlign — Interactive Dashboard
=====================================
Premium Streamlit UI for maintenance schedule optimization.

Launch:
    streamlit run streamlit_app.py
"""

import os
import sys
import time
import tempfile
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Project imports ──────────────────────────────────────────
from core.instance import ProblemInstance, MachineSpec, ProductionChain
from core.solver import solve, SolverResult
from core.baseline import fixed_interval_schedule, ALL_STRATEGIES
from analysis.evaluator import evaluate_schedule, compare_schedules
from utils.csv_loader import load_instance, load_machines_csv, load_chains_csv

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MaintAlign — Maintenance Optimizer",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium dark look ─────────────────────────
st.markdown("""
<style>
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #8899aa !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Tab styling */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    /* Sidebar header */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }

    /* Expander styling */
    details[data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        background: rgba(255,255,255,0.02) !important;
    }

    /* Table styling */
    .dataframe {
        font-size: 0.85rem !important;
    }

    /* Plotly chart containers */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        background: linear-gradient(120deg, #e94560, #0f3460, #53a8b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: #8899aa;
        margin: 6px 0 0 0;
        font-size: 0.95rem;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-optimal { background: #1b5e20; color: #a5d6a7; }
    .status-feasible { background: #e65100; color: #ffcc80; }
    .status-unknown { background: #424242; color: #bdbdbd; }
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
PLOTLY_TEXT = "#ccc"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PLOTLY_BG,
    plot_bgcolor=PLOTLY_BG,
    font=dict(color=PLOTLY_TEXT, family="Inter, sans-serif"),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)"),
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
        title=dict(text=title, font=dict(size=16)),
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
        title=dict(text="Cost Comparison: Baselines vs Optimized", font=dict(size=16)),
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
        marker=dict(colors=colors, line=dict(width=2, color="#1a1a2e")),
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Cost Breakdown", font=dict(size=16)),
        height=380,
        showlegend=False,
        annotations=[dict(
            text=f"${optimized.objective_value:,.0f}",
            x=0.5, y=0.5, font=dict(size=18, color="#fff", family="Inter"),
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
        fillcolor="rgba(66,165,245,0.2)",
        line=dict(color="#42A5F5", width=2),
        name="Usage",
        hovertemplate="t=%{x}<br>Technicians: %{y}<extra></extra>",
    ))
    fig.add_hline(y=K, line_dash="dash", line_color="#EF5350",
                  annotation_text=f"Capacity ({K})",
                  annotation_font_color="#EF5350")

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Technician Utilization", font=dict(size=16)),
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
        title=dict(text="Monte Carlo Cost Distributions", font=dict(size=16)),
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
        title=dict(text="Production Chain Values", font=dict(size=16)),
        xaxis=dict(gridcolor=PLOTLY_GRID, zeroline=False),
        yaxis=dict(title="Cost ($)", gridcolor=PLOTLY_GRID, zeroline=False),
        height=350,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
from utils.generator import (
    generate_tiny, generate_small, generate_medium_easy,
    generate_medium_hard, generate_large, generate_xl,
    generate_industrial, generate_factory, generate_instance,
)

# Instance presets (same as --full mode in main.py)
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

st.sidebar.markdown("## 🔧 MaintAlign")
st.sidebar.caption("Maintenance Schedule Optimizer")
st.sidebar.divider()

# Data source
st.sidebar.markdown("### 📂 Data Source")
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
st.sidebar.markdown("### ⚙️ Solver Settings")
time_limit = st.sidebar.slider("Time Limit (seconds)", 5, 120, 30, step=5)
block_weekends = st.sidebar.toggle("Block Weekends", value=False)
repair_factor = st.sidebar.slider(
    "Repair Factor", 0.5, 1.0, 1.0, step=0.05,
    help="1.0 = perfect repair, 0.7 = PM restores to 70% of new"
)

st.sidebar.divider()

# Run button
run_clicked = st.sidebar.button(
    "🚀 Optimize Schedule", type="primary", use_container_width=True
)


# ═══════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>🔧 MaintAlign Dashboard</h1>
    <p>Intelligent maintenance scheduling powered by Constraint Programming &amp; Weibull reliability models</p>
</div>
""", unsafe_allow_html=True)


# ── Run logic ────────────────────────────────────────────────
if run_clicked:
    with st.status("🔧 **Optimizing maintenance schedule...**", expanded=True) as status:
        # Step 1: Load / generate instance
        st.write("📦 Generating problem instance...")
        if data_source == "Generated Instance":
            gen_fn = INSTANCE_PRESETS[preset_name]
            inst = gen_fn(seed=seed)
        else:
            if uploaded_machines is None:
                st.error("❌ Please upload a machines CSV file first.")
                st.stop()
            # Write uploaded bytes to temp files
            m_bytes = uploaded_machines.getvalue()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                f.write(m_bytes.decode("utf-8"))
                tmp_machines = f.name
            tmp_chains = None
            if uploaded_chains is not None:
                c_bytes = uploaded_chains.getvalue()
                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                    f.write(c_bytes.decode("utf-8"))
                    tmp_chains = f.name
            inst = load_instance(tmp_machines, tmp_chains)
            os.unlink(tmp_machines)
            if tmp_chains:
                os.unlink(tmp_chains)

        st.write(f"✅ Instance loaded: **{inst.num_machines}** machines, "
                 f"**{inst.num_technicians}** technicians, "
                 f"**{inst.horizon}** periods, "
                 f"**{len(inst.chains)}** chains")

        # Apply options
        if block_weekends:
            blocked = [t for t in range(inst.horizon) if t % 7 in (5, 6)]
            inst.blocked_periods = blocked
            st.write(f"🚫 Blocked **{len(blocked)}** weekend periods")

        if repair_factor < 1.0:
            for m in inst.machines:
                m.repair_factor = repair_factor
            st.write(f"🔧 Repair factor set to **{repair_factor:.0%}**")

        # Step 2: Compute baselines
        st.write("📊 Computing baseline strategies...")
        baselines = {}
        for strat in ALL_STRATEGIES:
            baselines[strat] = fixed_interval_schedule(inst, strat)
        best_b_name = min(baselines, key=lambda k: baselines[k].objective_value)
        hint = baselines[best_b_name].machine_schedules
        st.write(f"✅ {len(baselines)} baselines computed. "
                 f"Best: **{best_b_name}** (${baselines[best_b_name].objective_value:,.0f})")

        # Step 3: Solve with CP-SAT
        st.write(f"🧠 Running CP-SAT solver (time limit: **{time_limit}s**)...")
        t0 = time.time()
        result = solve(inst, time_limit_seconds=time_limit, hint_schedule=hint)
        solve_elapsed = time.time() - t0
        st.write(f"✅ Solver finished in **{solve_elapsed:.1f}s** — "
                 f"Status: **{result.status}** — "
                 f"Cost: **${result.objective_value:,.0f}**")

        status.update(label="✅ **Optimization complete!**", state="complete", expanded=False)

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
    col1.metric("💰 Optimized Cost", f"${result.objective_value:,.0f}")
    col2.metric("📊 Best Baseline", f"${best_b_cost:,.0f}", help=best_b_name)
    col3.metric("💪 Savings", f"{savings_pct:+.1f}%")
    col4.metric("🔧 Tasks", f"{len(result.tasks)}")
    col5.metric("⏱️ Solve Time", f"{solve_elapsed:.1f}s")

    st.markdown(
        f'<p style="text-align:center; margin-top:4px;">'
        f'<span class="status-badge {status_class}">{result.status}</span> '
        f'&nbsp; {inst.num_machines} machines · {inst.num_technicians} technicians '
        f'· {inst.horizon} periods · {len(inst.chains)} chains</p>',
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Schedule", "💰 Cost Analysis", "🎲 Risk Analysis", "🏭 Factory"
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
                "β (shape)": m.weibull_beta,
                "η (scale)": m.weibull_eta,
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
            "Strategy": "✅ Optimized",
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
            run_mc = st.button("🎲 Run Simulation", type="primary",
                               use_container_width=True)
        with mc_col2:
            st.info(
                "Monte Carlo tests each schedule against random breakdowns. "
                "VaR₉₅ = average cost in the worst 5% of scenarios (tail risk)."
            )

        if run_mc or st.session_state.get("mc_results"):
            if run_mc:
                with st.spinner(f"Running {n_sims} simulations per strategy..."):
                    schedules = {
                        name: b.machine_schedules for name, b in baselines.items()
                    }
                    schedules["Optimized"] = result.machine_schedules
                    mc_results = compare_schedules(inst, schedules, n_sims=n_sims)
                st.session_state["mc_results"] = mc_results

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
                    "VaR₉₅ ($)": f"${er.var95:,.0f}",
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
                        size=20 if is_opt else 12,
                        color=colors_mc[i % len(colors_mc)],
                        symbol="star" if is_opt else "circle",
                        line=dict(width=2, color="white") if is_opt else dict(width=0),
                    ),
                    showlegend=False,
                    hovertemplate=(f"<b>{name}</b><br>"
                                  f"Mean: $%{{x:,.0f}}<br>"
                                  f"VaR95: $%{{y:,.0f}}<extra></extra>"),
                ))
            fig_scatter.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Risk vs Cost (lower-left = better)", font=dict(size=16)),
                xaxis_title="Mean Cost ($)",
                yaxis_title="VaR₉₅ — Tail Risk ($)",
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
                        f'<div style="background:rgba(255,255,255,0.04); '
                        f'border-left:4px solid {col}; padding:12px 16px; '
                        f'border-radius:8px; margin-bottom:10px;">'
                        f'<b style="color:{col};">{c.name}</b>'
                        f'<br><span style="color:#8899aa;font-size:0.85rem;">'
                        f'{" → ".join(machines_in_chain)}</span>'
                        f'<br><span style="color:#aaa;">Value: ${c.chain_value}/period '
                        f'&nbsp; Retooling: ${c.retooling_cost}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Standalone machines
                standalone = [m for m in inst.machines if inst.is_standalone(m.id)]
                if standalone:
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.04); '
                        f'border-left:4px solid {STANDALONE_COLOR}; padding:12px 16px; '
                        f'border-radius:8px; margin-bottom:10px;">'
                        f'<b style="color:{STANDALONE_COLOR};">Standalone</b>'
                        f'<br><span style="color:#8899aa;font-size:0.85rem;">'
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
        <div style="text-align:center; padding: 60px 20px;">
            <h2 style="color:#8899aa;">Welcome to MaintAlign</h2>
            <p style="color:#667788; font-size:1.1rem; max-width:500px; margin:0 auto;">
                Upload your factory data or use the built-in example,
                configure your solver, and click <b>🚀 Optimize Schedule</b>
                to find the optimal maintenance plan.
            </p>
            <br>
            <p style="color:#556677;">
                ⬅️ Use the sidebar to get started
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Quick feature overview
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown("""
            <div style="text-align:center; padding:20px; background:rgba(255,255,255,0.03);
                        border-radius:12px; border:1px solid rgba(255,255,255,0.06);">
                <div style="font-size:2rem;">📊</div>
                <b>Smart Scheduling</b>
                <p style="color:#667788; font-size:0.85rem;">
                    CP-SAT solver finds optimal PM timing
                </p>
            </div>
            """, unsafe_allow_html=True)
        with f2:
            st.markdown("""
            <div style="text-align:center; padding:20px; background:rgba(255,255,255,0.03);
                        border-radius:12px; border:1px solid rgba(255,255,255,0.06);">
                <div style="font-size:2rem;">🎲</div>
                <b>Risk Analysis</b>
                <p style="color:#667788; font-size:0.85rem;">
                    Monte Carlo simulation with Weibull failure
                </p>
            </div>
            """, unsafe_allow_html=True)
        with f3:
            st.markdown("""
            <div style="text-align:center; padding:20px; background:rgba(255,255,255,0.03);
                        border-radius:12px; border:1px solid rgba(255,255,255,0.06);">
                <div style="font-size:2rem;">🏭</div>
                <b>Chain-Aware</b>
                <p style="color:#667788; font-size:0.85rem;">
                    Production chain dependencies & grouping
                </p>
            </div>
            """, unsafe_allow_html=True)
