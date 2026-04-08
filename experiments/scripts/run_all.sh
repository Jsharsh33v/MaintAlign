#!/bin/bash
# MaintAlign Experiment Runner
# Run all 3 experiments and generate figures
#
# Usage (from MaintAlign project root):
#   bash experiments/scripts/run_all.sh
#
# Make sure venv is activated first:
#   source .venv/bin/activate

set -e

echo "============================================="
echo " MaintAlign — Running All Experiments"
echo "============================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/4] Running Experiment 1: Solver Scalability..."
python "$SCRIPT_DIR/run_scalability.py"
echo ""

echo "[2/4] Running Experiment 2: Baseline Comparison..."
python "$SCRIPT_DIR/run_baselines.py"
echo ""

echo "[3/4] Running Experiment 3: Monte Carlo Risk Analysis..."
python "$SCRIPT_DIR/run_montecarlo.py"
echo ""

echo "[4/4] Generating Figures..."
python "$SCRIPT_DIR/generate_figures.py"
echo ""

echo "============================================="
echo " All experiments complete!"
echo " Results:  experiments/results/*.csv"
echo " Figures:  experiments/figures/*.png"
echo "============================================="