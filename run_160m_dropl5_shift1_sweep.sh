#!/usr/bin/env bash
# 160M layer-choice ablations, run serially (same 8 GPUs): drop_l5 first,
# then shift1. drop_l5 was previously killed at step 4700 -- save/load point
# at the same checkpoint dir with a `latest` file at global_step4000, so this
# will auto-resume from there instead of starting over.
# nohup bash run_160m_dropl5_shift1_sweep.sh > run_160m_dropl5_shift1_sweep.log 2>&1 &

set -e

TS=$(date +%m%d_%H%M%S)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START drop_l5 (3H+9L, L5 dropped)"
python deepy.py train.py pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_static_3h_drop_l5.yml \
  2>&1 | tee pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_static_3h_drop_l5_output_${TS}.txt
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  drop_l5"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START shift1 (4H+8L, -1 layer shift)"
python deepy.py train.py pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_static_4h_shift1.yml \
  2>&1 | tee pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_static_4h_shift1_output_${TS}.txt
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  shift1"
