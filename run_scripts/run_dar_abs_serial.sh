#!/usr/bin/env bash
# Serial runner: DAR-abs A run, then Q3 Uniform-matched (one at a time, 8 GPU each).
# Both bias-free MLA / decoupled / d_r=32 / same recipe; only allocation differs.
# deepy.py blocks per run, so the loop is strictly serial.
#
# Foreground runner. To run in background yourself, from the gpt-neox repo root:
#   nohup bash run_scripts/run_dar_abs_serial.sh > nohup_dar_abs_serial.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."                       # -> gpt-neox repo root

CONFIGS=(
  pythia-70m-deduped_lskv_d128_w128_dar_abs   # DAR-abs (distance-adaptive)
  pythia-70m-deduped_lskv_d192_w0_dar_abs     # Q3 Uniform-matched (no window, d_u=192)
)

declare -a RESULTS
for cfg in "${CONFIGS[@]}"; do
  TS=$(date +%m%d_%H%M%S)
  LOG="pythia_lskv_configs/${cfg}_output_${TS}.txt"
  echo "======== [$(date '+%F %T')] START ${cfg}  ->  ${LOG} ========"
  python deepy.py train.py "pythia_lskv_configs/${cfg}.yml" 2>&1 | tee "${LOG}"
  rc=${PIPESTATUS[0]}
  echo "======== [$(date '+%F %T')] DONE  ${cfg}  (exit ${rc}) ========"
  RESULTS+=("${cfg}: exit ${rc}")
  sleep 30    # let GPUs/NCCL fully release before the next run
done

echo "================ SUMMARY ================"
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo "ALL_DONE"
