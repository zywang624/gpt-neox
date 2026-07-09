TS=$(date +%m%d_%H%M%S)

# nohup bash run_70m_phase1.sh > run_70m_phase1.log 2>&1 &

# ===== LAR 70M Phase 1: frozen alpha, pure CE, 17000 steps (~8.9B tokens) =====
python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_phase1.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_phase1_output_${TS}.txt
