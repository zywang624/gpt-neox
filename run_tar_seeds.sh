#!/bin/bash
# TAR (~50:50 balanced, th=0.491937) — extra seeds for variance estimation.
# seed 1 (1234) already trained: pythia_70m_ddim_d128_w128_bs256_frozen_alpha_hard_balanced_checkpoints
# This script runs the 2 additional seeds (1235, 1236) sequentially (each uses all 8 GPUs).
#
# Launch (detached):
#   nohup bash run_tar_seeds.sh > run_tar_seeds.log 2>&1 &

set -u

CONFIGS=(
  # Repro of the original balanced (seed-1) run — same config (default seed 1234,
  # th=0.491937) but a FRESH save dir (*_repro) so it trains from step 0 without
  # touching the original ckpt. Run first to diff its loss trajectory against the
  # original logs (pythia-70m-deduped_ddim_d128_w128_frozen_alpha_lookup_output_0518_*.txt)
  # and confirm the current `tar`-branch code reproduces the original behavior.
  # "pythia-70m-deduped_ddim_d128_w128_frozen_alpha_lookup_repro.yml"
  "pythia-70m-deduped_ddim_d128_w128_frozen_alpha_lookup_seed1235.yml"
  "pythia-70m-deduped_ddim_d128_w128_frozen_alpha_lookup_seed1236.yml"
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

echo "[$(date '+%F %T')] all TAR seed runs finished."
