#!/usr/bin/env bash
# 70M DAR d128 w128 seed 1235. Mirror of run_410m_lskv_d256_w0.sh.

# Foreground runner (writes a tee'd log). To run in background yourself:
#   cd /workspace/gpt-neox
#   nohup bash run_scripts/run_70m_lskv_d128_w128_seed1235.sh > nohup_70m_dar_s1235.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."                       # -> gpt-neox repo root
TS=$(date +%m%d_%H%M%S)

python deepy.py train.py pythia_lskv_configs/pythia-70m-deduped_lskv_d128_w128_seed1235.yml \
  2>&1 | tee pythia_lskv_configs/pythia-70m-deduped_lskv_d128_w128_seed1235_output_${TS}.txt
