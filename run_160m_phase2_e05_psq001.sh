#!/usr/bin/env bash
# 160M LAR phase2: entropy=0.05, p_sq=0.001, GPUs 8-15
# nohup bash run_160m_phase2_e05_psq001.sh > run_160m_phase2_e05_psq001.log 2>&1 &

TS=$(date +%m%d_%H%M%S)
python deepy.py train.py \
  pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_e05_psq001.yml \
  2>&1 | tee pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_e05_psq001_output_${TS}.txt
