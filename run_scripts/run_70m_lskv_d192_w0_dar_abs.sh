#!/usr/bin/env bash
# 70M Q3 Uniform-matched baseline (MLA-form, bias-free, d_u=192, no window -> every
# token through the bottleneck). Mirror of run_410m_lskv_d256_w0.sh.

# Foreground runner (writes a tee'd log). To run in background yourself, from the
# gpt-neox repo root:
#   nohup bash run_scripts/run_70m_lskv_d192_w0_dar_abs.sh > nohup_70m_uniform_matched.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."                       # -> gpt-neox repo root
TS=$(date +%m%d_%H%M%S)

python deepy.py train.py pythia_lskv_configs/pythia-70m-deduped_lskv_d192_w0_dar_abs.yml \
  2>&1 | tee pythia_lskv_configs/pythia-70m-deduped_lskv_d192_w0_dar_abs_output_${TS}.txt
