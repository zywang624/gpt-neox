#!/bin/bash
# Vanilla Pythia-70m (no LSKV/DDIM) — 3 seeds for variance estimation, all at the
# main-experiment budget (bs256 / iters19073 / ~10B tokens), matching the TAR runs.
# NOTE: all three seeds are trained fresh — the pre-existing on-disk vanilla ckpt
# (pythia_70m_vanilla_checkpoints) is a different budget (bs1024/iters5000) and is
# NOT reused. Runs sequentially (each uses all 8 GPUs).
#
# Launch (detached):
#   nohup bash run_vanilla_seeds.sh > run_vanilla_seeds.log 2>&1 &

set -u

CONFIGS=(
  "pythia_no_gelu_act/pythia-70m-deduped_seed1234.yml"
  "pythia_no_gelu_act/pythia-70m-deduped_seed1235.yml"
  "pythia_no_gelu_act/pythia-70m-deduped_seed1236.yml"
)

for cfg in "${CONFIGS[@]}"; do
  TS=$(date +%m%d_%H%M%S)
  base="${cfg%.yml}"
  log="${base}_output_${TS}.txt"
  echo "==================================================================="
  echo "[$(date '+%F %T')] START  $cfg   -> $log"
  echo "==================================================================="
  python deepy.py train.py "$cfg" 2>&1 | tee "$log"
  echo "[$(date '+%F %T')] DONE   $cfg"
done

echo "[$(date '+%F %T')] all vanilla seed runs finished."
