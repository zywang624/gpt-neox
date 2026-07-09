#!/usr/bin/env bash
# Serial sweep: 70M phase2 p_sq ranking stability test
# λ_psq ∈ {0.001, 0.005, 0.01, 0.05}, no entropy
# nohup bash run_70m_phase2_psq_sweep.sh > run_70m_phase2_psq_sweep.log 2>&1 &

set -e

CONFIGS=(
  "pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_phase2_psq0001.yml"
  "pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_phase2_psq0005.yml"
  "pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_phase2_psq001.yml"
  "pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_phase2_psq005.yml"
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
