#!/usr/bin/env bash
# Serial sweep: 160M phase2 log(p) regularisation
# λ ∈ {0.0001, 0.0005, 0.001, 0.02, 0.05, 0.10}, with anneal τ 1.0→0.01
# nohup bash run_160m_phase2_logp_sweep.sh > run_160m_phase2_logp_sweep.log 2>&1 &

set -e

CONFIGS=(
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_logp00001.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_logp00005.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_logp0001.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_logp002.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_logp005.yml"
  "pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_logp010.yml"
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
