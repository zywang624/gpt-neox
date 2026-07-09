#!/usr/bin/env bash
# 160M static routing baseline: single-layer swap, L5 -> L4, L9/L11/L12 unchanged
# (minimal perturbation of the ranking-based static_4h run).
# train from scratch, 5000 steps, matches vanilla 5000-step token budget
# nohup bash run_160m_static_4h_swap_l5.sh > run_160m_static_4h_swap_l5.log 2>&1 &

TS=$(date +%m%d_%H%M%S)

python deepy.py train.py pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_static_4h_swap_l5.yml \
  2>&1 | tee pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_static_4h_swap_l5_output_${TS}.txt
