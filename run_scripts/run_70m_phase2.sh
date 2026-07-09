TS=$(date +%m%d_%H%M%S)

# nohup bash run_70m_phase2.sh > run_70m_phase2.log 2>&1 &
# Run AFTER run_70m_phase1.sh completes (loads from phase1 checkpoint).

# ===== LAR 70M Phase 2: routing search, 2000 steps (~1B tokens) =====
python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_phase2.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_phase2_output_${TS}.txt
