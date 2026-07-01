#!/usr/bin/env bash
# 160M static routing: L5, L9, L11, L12 → H (d=192), others → L (d=96)
# train from scratch, 5000 steps, matches vanilla 5000-step token budget
# nohup bash run_160m_static_4h.sh > run_160m_static_4h.log 2>&1 &

TS=$(date +%m%d_%H%M%S)

python deepy.py train.py pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_static_4h.yml \
  2>&1 | tee pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_static_4h_output_${TS}.txt
