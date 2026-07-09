#!/usr/bin/env bash
# Invoke from the repo root: bash run_scripts/run_70m_vanilla.sh
TS=$(date +%m%d_%H%M%S)

python deepy.py train.py pythia_lskv_configs/pythia-70m-deduped.yml \
  2>&1 | tee pythia_lskv_configs/pythia-70m-deduped_output_${TS}.txt
