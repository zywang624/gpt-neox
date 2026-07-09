#!/usr/bin/env bash
# Invoke from the repo root: bash run_scripts/run_70m_lskv_d256_w128.sh
TS=$(date +%m%d_%H%M%S)

python deepy.py train.py pythia_lskv_configs/pythia-70m-deduped_lskv_d256_w128.yml \
  2>&1 | tee pythia_lskv_configs/pythia-70m-deduped_lskv_d256_w128_output_${TS}.txt
