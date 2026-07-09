#!/usr/bin/env bash
# 160M DAR bracket: all-H (d=192) then all-L (d=96), run serially (same 8 GPUs).
# nohup bash run_160m_dar_sweep.sh > run_160m_dar_sweep.log 2>&1 &

set -e

TS=$(date +%m%d_%H%M%S)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START DAR-192 (all-H)"
python deepy.py train.py pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_dar192.yml \
  2>&1 | tee pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_dar192_output_${TS}.txt
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  DAR-192 (all-H)"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START DAR-96 (all-L)"
python deepy.py train.py pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_dar96.yml \
  2>&1 | tee pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_dar96_output_${TS}.txt
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  DAR-96 (all-L)"
