#!/usr/bin/env bash
# 160M Uniform (lskv d192, window=0, GELU+bias). Mirror of run_410m_lskv_d256_w0.sh.

# Foreground runner (writes a tee'd log). To run in background yourself:
#   cd /workspace/gpt-neox
#   nohup bash run_scripts/run_160m_lskv_d192_w0.sh > nohup_160m_uniform.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."                       # -> gpt-neox repo root
TS=$(date +%m%d_%H%M%S)

python deepy.py train.py pythia_lskv_configs/pythia-160m-deduped_lskv_d192_w0.yml \
  2>&1 | tee pythia_lskv_configs/pythia-160m-deduped_lskv_d192_w0_output_${TS}.txt
