TS=$(date +%m%d_%H%M%S)

# nohup bash run_160m_phase1.sh > run_160m_phase1.log 2>&1 &

# ===== LAR 160M Phase 1: frozen alpha, pure CE, 4500 steps (~9.4B tokens) =====
python deepy.py train.py pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase1.yml 2>&1 | tee pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_phase1_output_${TS}.txt
