TS=$(date +%m%d_%H%M%S)

# nohup bash run_160m_phase2.sh > run_160m_phase2.log 2>&1 &
# Run AFTER run_160m_phase1.sh completes (loads from phase1 checkpoint).

# ===== LAR 160M Phase 2: routing search, 500 steps (~1B tokens) =====
python deepy.py train.py pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2.yml 2>&1 | tee pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase2_output_${TS}.txt
