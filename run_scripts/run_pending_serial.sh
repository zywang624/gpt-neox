#!/usr/bin/env bash
# Serial runner for the 5 not-yet-trained lskv configs (one after another).
# Order: 70M seed variants first (DAR 1235/1236, Uniform 1235/1236), 160M LAST.
#
# Foreground runner. To run in background yourself:
#   cd /workspace/gpt-neox
#   nohup bash run_scripts/run_pending_serial.sh > nohup_pending_serial.out 2>&1 &
#
# ⚠️ 70M configs use GPUs 0-7, 160M uses 1-7 — both conflict with the currently
# running 410M Uniform (GPUs 1-7). START THIS ONLY AFTER the 410M finishes, else
# the first run fails to grab GPUs. deepy.py blocks until each run completes, so
# the loop is strictly serial (one training at a time).
set -uo pipefail
cd "$(dirname "$0")/.."                       # -> gpt-neox repo root

CONFIGS=(
  pythia-70m-deduped_lskv_d128_w128_seed1235   # 70M DAR   seed 1235
  pythia-70m-deduped_lskv_d128_w128_seed1236   # 70M DAR   seed 1236
  pythia-70m-deduped_lskv_d128_w0_seed1235     # 70M Uniform seed 1235
  pythia-70m-deduped_lskv_d128_w0_seed1236     # 70M Uniform seed 1236
  pythia-160m-deduped_lskv_d192_w0             # 160M Uniform  (LAST)
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
echo "ALL PENDING RUNS COMPLETE"
