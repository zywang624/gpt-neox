#!/usr/bin/env bash
# Serial: 160M LAR phase2, psq=0.001, e=0.0 then e=0.05
# nohup bash run_160m_phase2_psq001.sh > run_160m_phase2_psq001.log 2>&1 &

set -e

CONFIGS=(
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_e00_psq001.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_e05_psq001.yml"
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
