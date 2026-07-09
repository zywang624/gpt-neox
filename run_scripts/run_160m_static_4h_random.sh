#!/usr/bin/env bash
# 160M static routing baseline: 4H + 8L with FULLY RANDOM H-layer choice
# (random.seed(42): L1,L2,L5,L11), contrasted against the ranking-based
# static_4h run (L5,L9,L11,L12), shift1 (systematic -1 shift), and swap_l5
# (single-layer swap).
# train from scratch, 5000 steps, matches vanilla 5000-step token budget
# nohup bash run_160m_static_4h_random.sh > run_160m_static_4h_random.log 2>&1 &

TS=$(date +%m%d_%H%M%S)

python deepy.py train.py pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_static_4h_random.yml \
  2>&1 | tee pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_static_4h_random_output_${TS}.txt
