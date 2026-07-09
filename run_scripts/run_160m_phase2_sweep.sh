#!/usr/bin/env bash
# Serial sweep: 160M LAR phase2, 3x3 grid of (entropy_lambda, p_sq_lambda)
# Run: nohup bash run_160m_phase2_sweep.sh > run_160m_phase2_sweep.log 2>&1 &
#
# entropy_lambda: 0.0, 0.05, 0.10
# p_sq_lambda:   0.02, 0.05, 0.10
# All load from phase1 checkpoint; each run is ~500 steps.

set -e

CONFIGS=(
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_e00_psq02.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_e00_psq05.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_e00_psq10.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_e05_psq02.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_e05_psq05.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_e05_psq10.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_e10_psq02.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_e10_psq05.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_e10_psq10.yml"
)

TOTAL=${#CONFIGS[@]}

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  name=$(basename "$cfg" .yml)
  TS=$(date +%m%d_%H%M%S)
  logfile="pythia_layer_wise/${name}_output_${TS}.txt"

  echo "=========================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] RUN $((i+1))/$TOTAL: $name"
  echo "=========================================="

  python deepy.py train.py "$cfg" 2>&1 | tee "$logfile"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: $name"
  echo ""
done

echo "All $TOTAL sweep runs complete."
