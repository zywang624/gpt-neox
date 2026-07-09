#!/usr/bin/env bash
# 160M static routing baseline: 4H + 8L with H-layer choice shifted back by 1
# from the ranking-based static_4h run (L5,L9,L11,L12 → L4,L8,L10,L11).
# train from scratch, 5000 steps, matches vanilla 5000-step token budget
# nohup bash run_160m_static_4h_shift1.sh > run_160m_static_4h_shift1.log 2>&1 &

TS=$(date +%m%d_%H%M%S)

python deepy.py train.py pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_static_4h_shift1.yml \
  2>&1 | tee pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_static_4h_shift1_output_${TS}.txt
