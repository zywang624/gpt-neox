#!/usr/bin/env bash
# 160M static routing ablation: DROP L5 entirely (not swapped) -> 3H + 9L
# (H = L9, L11, L12 only), contrasted against swap_l5 (L5->L4, still 4H+8L)
# and the ranking-based static_4h (L5,L9,L11,L12).
# train from scratch, 5000 steps, matches vanilla 5000-step token budget
# nohup bash run_160m_static_3h_drop_l5.sh > run_160m_static_3h_drop_l5.log 2>&1 &

TS=$(date +%m%d_%H%M%S)

python deepy.py train.py pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_static_3h_drop_l5.yml \
  2>&1 | tee pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_static_3h_drop_l5_output_${TS}.txt
