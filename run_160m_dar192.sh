#!/usr/bin/env bash
# 160M DAR (all-H): every layer forced to the high-rank branch (d=192), no routing
# train from scratch, 5000 steps, matches vanilla 5000-step token budget
# nohup bash run_160m_dar192.sh > run_160m_dar192.log 2>&1 &

TS=$(date +%m%d_%H%M%S)

python deepy.py train.py pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_dar192.yml \
  2>&1 | tee pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_dar192_output_${TS}.txt
