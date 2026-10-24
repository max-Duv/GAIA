#!/bin/bash
# =============================================================================
# PROJECT GAIA: Full Experiment Suite for Publication
# Run this ON the Jetson AGX Thor from the GAIA/ directory (no sudo needed —
# T3 invokes sudo internally only for nvidia-smi -pl)
# Usage: bash scripts/run_full_experiments.sh
# =============================================================================
set -e

GAIA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$GAIA_DIR"

DURATION=15        # seconds per phase
TRIALS=10          # repeated trials for statistical significance
SWEEP_TRIALS=5     # trials for mechanism-isolation sweep

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$GAIA_DIR/results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "====================================================================="
echo "  PROJECT GAIA: FULL EXPERIMENT SUITE FOR PUBLICATION"
echo "  $(date)"
echo "  Duration/phase: ${DURATION}s | Trials: ${TRIALS} | Sweep trials: ${SWEEP_TRIALS}"
echo "  Results dir: $RESULTS_DIR"
echo "====================================================================="

# --- Sanity checks (non-fatal — sklearn/matplotlib only needed for T2 batch plotting) ---
echo ""
echo "[prereq] Checking environment..."
nvidia-smi --query-gpu=name,compute_cap,power.limit,driver_version --format=csv
python3 -c "import numpy, pandas, scipy; print('[prereq] Core packages (numpy, pandas, scipy) OK')"
python3 -c "import sklearn; print('[prereq] scikit-learn OK')" 2>/dev/null || echo "[prereq] WARNING: scikit-learn missing — T2 batch plots will be skipped (run analyze_results.py on workstation instead)"
python3 -c "import matplotlib; print('[prereq] matplotlib OK')" 2>/dev/null || echo "[prereq] WARNING: matplotlib missing — plots will be skipped"
python3 -c "import pynvml; pynvml.nvmlInit(); print('[prereq] pynvml OK')"
nvcc --version | tail -1
echo ""

# --- Record system info ---
echo "[info] Recording system information..."
{
    echo "=== System Info ==="
    date
    uname -a
    cat /proc/device-tree/model 2>/dev/null || echo "No device-tree model"
    echo ""
    nvidia-smi
    echo ""
    nvcc --version
    echo ""
    python3 --version
    pip3 list 2>/dev/null | grep -E "numpy|pandas|scipy|scikit|matplotlib|nvidia-ml"
} > "$RESULTS_DIR/system_info.txt" 2>&1
echo "[info] System info saved to $RESULTS_DIR/system_info.txt"

# =============================================================================
# THRUST 1: Attack & Fault Assessment (Main 3-phase experiment)
# =============================================================================
echo ""
echo "====================================================================="
echo "  THRUST 1: 5-Condition Controlled Experiment (${TRIALS} trials)"
echo "  Conditions: alone, sustained_stress, cache_only, compute_only, pulsed_stress"
echo "====================================================================="

cd "$GAIA_DIR/T1"

# Force recompile CUDA binaries (source may have changed)
echo "[T1] Removing old CUDA binaries and recompiling..."
rm -f gpu_power_poc_v3 gpu_victim_kernel
python3 controlled_3phase_runner_v2.py --duration 1 --trials 1

echo "[T1] Running main 3-phase experiment (randomized phase order)..."
python3 controlled_3phase_runner_v2.py \
    --duration "$DURATION" \
    --trials "$TRIALS" \
    --randomize \
    --out "$RESULTS_DIR/t1_3phase_experiment.csv"

# Copy the summary CSV too
cp -f controlled_3phase_experiment_v2_summary.csv \
    "$RESULTS_DIR/t1_3phase_summary.csv" 2>/dev/null || true

echo ""
echo "====================================================================="
echo "  THRUST 1: Mechanism-Isolation Sweep (${SWEEP_TRIALS} trials)"
echo "====================================================================="

echo "[T1] Running mechanism-isolation sweep..."
python3 controlled_3phase_runner_v2.py \
    --duration "$DURATION" \
    --trials "$SWEEP_TRIALS" \
    --sweep \
    --out "$RESULTS_DIR/t1_sweep_experiment.csv"

cp -f controlled_3phase_experiment_v2_summary.csv \
    "$RESULTS_DIR/t1_sweep_summary.csv" 2>/dev/null || true

# Generate T1 publication figure
echo "[T1] Generating publication figures..."
TDP_RAW=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits 2>/dev/null | head -1)
# Thor reports [N/A]; default to 130W (Thor TDP)
if echo "$TDP_RAW" | grep -qE '^[0-9]+\.?[0-9]*$'; then
    TDP="$TDP_RAW"
else
    TDP=130
fi
cd "$GAIA_DIR"
python3 power_analysis_plotter.py "$RESULTS_DIR/t1_3phase_experiment.csv" "$TDP" || echo "[T1] WARNING: Plot generation failed. Will generate on workstation."
# Move the generated PNG
mv -f T1/power_sag_analysis_v2.png "$RESULTS_DIR/t1_power_sag_figure.png" 2>/dev/null || true

echo "[T1] Complete."

# =============================================================================
# THRUST 2: Signal Processing & Anomaly Detection
# =============================================================================
echo ""
echo "====================================================================="
echo "  THRUST 2: Batch Signal Processing & Anomaly Detection"
echo "====================================================================="

cd "$GAIA_DIR"

echo "[T2] Processing main experiment data..."
python3 T2/gpu_defense_filters_v2.py \
    --mode batch \
    --file "$RESULTS_DIR/t1_3phase_experiment.csv" \
    --output "$RESULTS_DIR/t2_defense_diagnostics.png" || echo "[T2] WARNING: Batch processing failed (likely missing scikit-learn). Run analyze_results.py on workstation."

echo "[T2] Processing sweep data..."
python3 T2/gpu_defense_filters_v2.py \
    --mode batch \
    --file "$RESULTS_DIR/t1_sweep_experiment.csv" \
    --output "$RESULTS_DIR/t2_sweep_diagnostics.png" || echo "[T2] WARNING: Sweep processing failed. Will process on workstation."

echo "[T2] Complete."

# =============================================================================
# THRUST 3: Closed-Loop Active Mitigation
# =============================================================================
echo ""
echo "====================================================================="
echo "  THRUST 3: Closed-Loop Mitigation (5 trials)"
echo "====================================================================="

cd "$GAIA_DIR"

echo "[T3] Running closed-loop mitigation experiment..."
python3 T3/closed_loop_mitigator_v2.py \
    --duration "$DURATION" \
    --trials 5 \
    --out "$RESULTS_DIR/t3_closed_loop_mitigation.csv"

echo "[T3] Complete."

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "====================================================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "  $(date)"
echo "====================================================================="
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Files produced:"
ls -lh "$RESULTS_DIR/"
echo ""
echo "Transfer these files to your workstation for paper writing:"
echo "  scp -r $(whoami)@$(hostname -I | awk '{print $1}'):$RESULTS_DIR ."
echo ""
echo "====================================================================="
