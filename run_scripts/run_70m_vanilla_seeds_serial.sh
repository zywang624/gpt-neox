#!/usr/bin/env bash
# 70M Vanilla seed controls (1235 then 1236), SERIAL -- each needs all 8 GPUs, so they
# run one after another. Standard transformer (NO lskv args) -> on the lskv branch
# gpt2_model.py routes to transformer.py, not transformer_lskv.
#
# Foreground runner (tee'd per-seed logs). To run in background yourself:
#   cd /workspace/gpt-neox
#   nohup bash run_scripts/run_70m_vanilla_seeds_serial.sh > nohup_70m_van_seeds.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."                       # -> gpt-neox repo root

for SEED in 1235 1236; do
  CFG="pythia_no_gelu_act/pythia-70m-deduped_vanilla_seed${SEED}.yml"
  TS=$(date +%m%d_%H%M%S)
  echo "======================================================================"
  echo "[$(date '+%F %T')] START vanilla seed ${SEED}  ($CFG)"
  echo "======================================================================"
  python deepy.py train.py "$CFG" \
    2>&1 | tee "pythia_no_gelu_act/pythia-70m-deduped_vanilla_seed${SEED}_output_${TS}.txt"
  echo "[$(date '+%F %T')] DONE vanilla seed ${SEED}"
done
echo "ALL VANILLA SEEDS DONE"
