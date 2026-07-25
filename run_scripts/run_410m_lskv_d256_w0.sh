#!/usr/bin/env bash
# 410M Uniform (lskv d256, window=0, GELU+bias). Mirror of run_70m_lskv_d256_w0.sh.

# Foreground runner (writes a tee'd log). To run in background yourself:
#   cd /workspace/gpt-neox
#   nohup bash run_scripts/run_410m_lskv_d256_w0.sh > nohup_410m_uniform.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."                       # -> gpt-neox repo root
TS=$(date +%m%d_%H%M%S)

python deepy.py train.py pythia_lskv_configs/pythia-410m-deduped_lskv_d256_w0.yml \
  2>&1 | tee pythia_lskv_configs/pythia-410m-deduped_lskv_d256_w0_output_${TS}.txt
