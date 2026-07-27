#!/usr/bin/env bash
# 70M Vanilla seed 1236 (seed control). Standard transformer (NO lskv args) -- on the
# lskv branch gpt2_model.py routes this to transformer.py, not transformer_lskv.
# Mirror of the submitted vanilla baseline, seed 1234->1236 + own save dir.
#
# Foreground runner (writes a tee'd log). To run in background yourself:
#   cd /workspace/gpt-neox
#   nohup bash run_scripts/run_70m_vanilla_seed1236.sh > nohup_70m_van_s1236.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."                       # -> gpt-neox repo root
TS=$(date +%m%d_%H%M%S)

python deepy.py train.py pythia_no_gelu_act/pythia-70m-deduped_vanilla_seed1236.yml \
  2>&1 | tee pythia_no_gelu_act/pythia-70m-deduped_vanilla_seed1236_output_${TS}.txt
