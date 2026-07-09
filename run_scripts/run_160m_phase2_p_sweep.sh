#!/usr/bin/env bash
# Serial sweep: 160M phase2 linear-p regularisation (vs p² comparison)
# λ_p ∈ {0.001, 0.02, 0.05, 0.10}, no entropy, with anneal
# nohup bash run_160m_phase2_p_sweep.sh > run_160m_phase2_p_sweep.log 2>&1 &

set -e

CONFIGS=(
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_p0001.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_p002.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_p005.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_p010.yml"
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

echo "All $TOTAL runs complete."
