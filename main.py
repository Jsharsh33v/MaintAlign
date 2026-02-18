"""
MaintAlign - Main Experiment Runner (v3)
==========================================
Full pipeline: generate → solve → compare baselines → visualize → analyze.

Usage:
    python main.py                        # Demo (tiny + small + medium)
    python main.py --full                 # Full experiment suite
    python main.py --sensitivity          # Run sensitivity analysis
    python main.py --instance FILE        # Solve specific JSON instance
    python main.py --csv MACHINES.csv     # Solve from CSV data
    python main.py --simulate             # Add Monte Carlo simulation
    python main.py --weekends             # Block weekends
    python main.py --decompose            # Use decomposition for large instances
    python main.py --log-level DEBUG      # Verbose logging
"""

import argparse
import logging
import os
import sys
import json
from utils.generator import (
    generate_tiny, generate_small, generate_medium_easy,
    generate_medium_hard, generate_large, generate_xl,
    generate_industrial, generate_factory, generate_instance,
)
from core.solver import solve
from core.baseline import fixed_interval_schedule, ALL_STRATEGIES
from utils.visualizer import (
    plot_gantt, plot_cost_comparison,
    plot_technician_utilization, plot_sensitivity,
    plot_chain_breakdown,
)
from analysis.evaluator import compare_schedules
from core.decomposer import solve_decomposed

logger = logging.getLogger(__name__)


def run_single(instance, output_dir="results", time_limit=60, verbose=True):
    """Run baselines + optimized for one instance. Return results dict."""
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(instance.summary())

    # ── Baselines ────────────────────────────────────────────
    baselines = {}
    for strat in ALL_STRATEGIES:
        b = fixed_interval_schedule(instance, strat)
        baselines[strat] = b
        if verbose:
            print(f"  Baseline {strat:<14}: ${b.objective_value:>10,.2f}  "
                  f"tasks={len(b.tasks)}")

    # ── Warm-start: use best baseline as hint ────────────────
    best_b_name = min(baselines, key=lambda k: baselines[k].objective_value)
    hint_schedule = baselines[best_b_name].machine_schedules

    # ── Optimized ────────────────────────────────────────────
    if verbose:
        print(f"  Solving (limit={time_limit}s)...", end=" ", flush=True)

    opt = solve(instance, time_limit_seconds=time_limit,
                hint_schedule=hint_schedule)

    if verbose:
        print(f"{opt.status}")
        print(f"  Optimized:          ${opt.objective_value:>10,.2f}  "
              f"tasks={len(opt.tasks)}  time={opt.solve_time_seconds:.2f}s")

    # ── Savings ──────────────────────────────────────────────
    best_b_cost = baselines[best_b_name].objective_value
    savings_pct = 0.0
    if best_b_cost > 0 and opt.objective_value < float('inf'):
        savings_pct = (1 - opt.objective_value / best_b_cost) * 100
        if verbose:
            print(f"  Savings vs {best_b_name}: {savings_pct:+.1f}%")

    # ── Visualizations ───────────────────────────────────────
    pfx = os.path.join(output_dir, instance.name)

    plot_gantt(instance, opt, save_path=f"{pfx}_gantt_opt.png")
    plot_gantt(instance, baselines[best_b_name],
              title=f"Baseline ({best_b_name}): {instance.name}",
              save_path=f"{pfx}_gantt_base.png")
    plot_cost_comparison(instance, baselines, opt,
                        save_path=f"{pfx}_cost_cmp.png")
    plot_technician_utilization(instance, opt,
                               save_path=f"{pfx}_tech_util.png")
    if instance.chains:
        plot_chain_breakdown(instance, opt,
                            save_path=f"{pfx}_chain_breakdown.png")

    # ── Results JSON ─────────────────────────────────────────
    data = {
        "instance": instance.name,
        "M": instance.num_machines,
        "K": instance.num_technicians,
        "T": instance.horizon,
        "chains": len(instance.chains),
        "RC": round(instance.resource_constrainedness, 3),
        "optimized": {
            "status": opt.status,
            "cost": round(opt.objective_value, 2),
            "pm": round(opt.total_pm_cost, 2),
            "prod_loss": round(opt.total_production_loss, 2),
            "retooling": round(opt.total_retooling_cost, 2),
            "failure": round(opt.total_failure_cost, 2),
            "solve_time": round(opt.solve_time_seconds, 3),
            "num_tasks": len(opt.tasks),
            "schedule": {str(k): v for k, v in opt.machine_schedules.items()},
        },
        "baselines": {
            name: {"cost": round(r.objective_value, 2), "tasks": len(r.tasks)}
            for name, r in baselines.items()
        },
        "savings_pct": round(savings_pct, 2),
        "chain_costs": {
            str(k): {kk: round(vv, 2) if isinstance(vv, float) else vv
                     for kk, vv in v.items()}
            for k, v in opt.chain_costs.items()
        } if opt.chain_costs else {},
    }
    with open(f"{pfx}_results.json", 'w') as f:
        json.dump(data, f, indent=2)

    return data


def run_demo(time_limit=30):
    """Quick demo with three instance sizes."""
    print("=" * 60)
    print(" MaintAlign v2 — Demo Run")
    print("=" * 60)

    results = []
    generators = [
        ("tiny",        generate_tiny),
        ("small",       generate_small),
        ("medium_easy", generate_medium_easy),
    ]
    total = len(generators)

    for idx, (label, gen) in enumerate(generators, 1):
        inst = gen()
        print(f"\n[{idx}/{total}] ── {label.upper()} ──")
        r = run_single(inst, output_dir="results/demo", time_limit=time_limit)
        results.append(r)

    # Summary table
    print(f"\n{'═'*60}")
    print(f" SUMMARY")
    print(f"{'═'*60}")
    print(f" {'Instance':<16}{'Chains':<7}{'Optimized':>11}{'Baseline':>11}{'Save':>8}")
    print(f" {'─'*52}")
    for r in results:
        best_b = min(b['cost'] for b in r['baselines'].values())
        print(f" {r['instance']:<16}{r['chains']:<7}"
              f"${r['optimized']['cost']:>9,.0f}"
              f"${best_b:>9,.0f}"
              f"{r['savings_pct']:>7.1f}%")

    print(f"\n Results saved to results/demo/")


def run_sensitivity():
    """
    Sensitivity analysis: vary technician count and observe cost impact.
    This demonstrates the 'shadow price' of adding a technician.
    """
    print("=" * 60)
    print(" MaintAlign v2 — Sensitivity Analysis")
    print("=" * 60)

    base_M = 10
    base_T = 30
    base_chains = 2
    results_by_k = {}

    print("\n  Varying Technicians (K):")
    for K in range(1, 7):
        inst = generate_instance(
            f"sens_K{K}", base_M, K, base_T,
            num_chains=base_chains, seed=42
        )
        result = solve(inst, time_limit_seconds=30)
        results_by_k[K] = result
        print(f"  K={K}: ${result.objective_value:>10,.2f}  "
              f"tasks={len(result.tasks)}  "
              f"PM=${result.total_pm_cost:,.0f}  "
              f"Fail=${result.total_failure_cost:,.0f}")

    os.makedirs("results/sensitivity", exist_ok=True)
    plot_sensitivity(results_by_k, "Technicians (K)",
                    save_path="results/sensitivity/cost_vs_K.png")

    # Shadow price analysis
    print(f"\n  Shadow Price of Technicians:")
    ks = sorted(results_by_k.keys())
    for i in range(1, len(ks)):
        prev_cost = results_by_k[ks[i-1]].objective_value
        curr_cost = results_by_k[ks[i]].objective_value
        if prev_cost < float('inf') and curr_cost < float('inf'):
            delta = prev_cost - curr_cost
            print(f"    K: {ks[i-1]}→{ks[i]}  Δcost = ${delta:>+10,.2f}")

    # Also vary cost ratio
    print(f"\n  Cost Ratio Sensitivity (CM/PM):")
    for ratio in [2, 5, 8, 12, 20]:
        inst = generate_instance(
            f"sens_ratio{ratio}", base_M, 3, base_T,
            num_chains=base_chains, seed=42,
            cost_ratio_range=(ratio, ratio),
        )
        result = solve(inst, time_limit_seconds=20)
        print(f"    CM/PM={ratio:>2}x: ${result.objective_value:>10,.2f}  "
              f"tasks={len(result.tasks)}")

    print(f"\n  Results saved to results/sensitivity/")


def run_full(time_limit=60):
    """Full experiment suite."""
    print("=" * 60)
    print(" MaintAlign v2 — Full Suite")
    print("=" * 60)

    presets = [
        ("tiny", generate_tiny),
        ("small", generate_small),
        ("medium_easy", generate_medium_easy),
        ("medium_hard", generate_medium_hard),
        ("large", generate_large),
        ("xl", generate_xl),
        ("industrial", generate_industrial),
        ("factory", generate_factory),
    ]
    all_r = []
    total = len(presets) * 3  # 3 seeds each
    run_idx = 0

    for pname, gen in presets:
        for seed in range(3):
            run_idx += 1
            inst = gen(seed=seed)
            inst.name = f"{pname}_s{seed}"
            print(f"\n  [{run_idx}/{total}] ── {inst.name} ──")
            r = run_single(inst, output_dir="results/full",
                          time_limit=time_limit, verbose=False)
            all_r.append(r)
            best_b = min(b['cost'] for b in r['baselines'].values())
            print(f"  {inst.name:<18} Opt=${r['optimized']['cost']:>9,.0f}  "
                  f"Base=${best_b:>9,.0f}  "
                  f"Save={r['savings_pct']:>+6.1f}%  "
                  f"Time={r['optimized']['solve_time']:.2f}s")

    print(f"\n{'═'*70}")
    print(f" FULL RESULTS")
    print(f"{'═'*70}")
    print(f" {'Instance':<18}{'M':<4}{'K':<4}{'T':<5}{'Ch':<4}{'RC':<6}"
          f"{'Opt$':>10}{'Base$':>10}{'Save%':>8}{'Time':>8}")
    print(f" {'─'*68}")
    for r in all_r:
        best_b = min(b['cost'] for b in r['baselines'].values())
        print(f" {r['instance']:<18}{r['M']:<4}{r['K']:<4}{r['T']:<5}"
              f"{r['chains']:<4}{r['RC']:<6.2f}"
              f"${r['optimized']['cost']:>8,.0f}"
              f"${best_b:>8,.0f}"
              f"{r['savings_pct']:>7.1f}%"
              f"{r['optimized']['solve_time']:>7.2f}s")

    print(f"\n Results saved to results/full/")


def main():
    parser = argparse.ArgumentParser(
        description="MaintAlign — Maintenance Scheduling Optimizer (v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                        Quick demo
  python main.py --full                                 Full experiment suite
  python main.py --csv data/example_machines.csv        Solve from CSV
  python main.py --simulate --num-sims 500              Monte Carlo
  python main.py --weekends                             Block weekends
  python main.py --decompose                            Decomposition mode
  python main.py --log-level DEBUG                      Verbose output
        """
    )
    parser.add_argument("--demo", action="store_true",
                       help="Run quick demo (5 machines, default)")
    parser.add_argument("--full", action="store_true",
                       help="Run full experiment suite (5 sizes × 3 seeds)")
    parser.add_argument("--sensitivity", action="store_true",
                       help="Run sensitivity analysis")
    parser.add_argument("--instance", type=str, metavar="FILE",
                       help="Solve a specific JSON instance file")
    parser.add_argument("--csv", type=str, metavar="FILE",
                       help="Load machines from CSV file")
    parser.add_argument("--chains", type=str, metavar="FILE",
                       help="Load chains from CSV file (use with --csv)")
    parser.add_argument("--log-level", type=str, default="WARNING",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: WARNING)")
    parser.add_argument("--time-limit", type=int, default=60,
                       help="Solver time limit in seconds (default: 60)")
    parser.add_argument("--simulate", action="store_true",
                       help="Run Monte Carlo simulation after solving")
    parser.add_argument("--num-sims", type=int, default=500,
                       help="Number of Monte Carlo simulations (default: 500)")
    parser.add_argument("--weekends", action="store_true",
                       help="Block every 6th and 7th period (weekends)")
    parser.add_argument("--decompose", action="store_true",
                       help="Use decomposition for large instances")
    parser.add_argument("--repair-factor", type=float, default=1.0,
                       help="PM repair factor: 1.0=perfect, 0.7=imperfect (default: 1.0)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)-12s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.csv:
        from utils.csv_loader import load_instance as csv_load
        inst = csv_load(args.csv, args.chains)

        # Apply options
        if args.weekends:
            inst.blocked_periods = [t for t in range(inst.horizon)
                                    if t % 7 in (5, 6)]
            print(f"  Blocked {len(inst.blocked_periods)} weekend periods")

        if args.repair_factor < 1.0:
            for m in inst.machines:
                m.repair_factor = args.repair_factor
            print(f"  Imperfect repair: factor={args.repair_factor}")

        if args.decompose and inst.num_machines > 15:
            result = solve_decomposed(inst, time_limit_seconds=args.time_limit)
        else:
            result = solve(inst, time_limit_seconds=args.time_limit)

        print(inst.summary())
        print(f"\n  Solver: {result.status}  Cost=${result.objective_value:,.0f}  "
              f"Tasks={len(result.tasks)}  Time={result.solve_time_seconds:.2f}s")

        if args.simulate:
            print(f"\n  Running {args.num_sims} Monte Carlo simulations...")
            baselines = {}
            for strat in ALL_STRATEGIES:
                b = fixed_interval_schedule(inst, strat)
                baselines[strat] = b.machine_schedules
            schedules = {"Optimized": result.machine_schedules}
            schedules.update(baselines)
            mc_results = compare_schedules(inst, schedules, n_sims=args.num_sims)
            print(f"\n  {'Schedule':<20} {'Mean$':>10} {'±Std':>10} {'VaR95':>10} {'Failures':>10}")
            print(f"  {'─'*60}")
            for name, r in mc_results.items():
                print(f"  {name:<20} ${r.mean_cost:>9,.0f} ±${r.std_cost:>8,.0f} "
                      f"${r.var95:>9,.0f} {r.mean_failures:>9.1f}")

    elif args.full:
        run_full(time_limit=args.time_limit)
    elif args.sensitivity:
        run_sensitivity()
    elif args.instance:
        from core.instance import ProblemInstance
        inst = ProblemInstance.load(args.instance)
        run_single(inst, time_limit=args.time_limit)
    else:
        run_demo(time_limit=args.time_limit)


if __name__ == "__main__":
    main()
