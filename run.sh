TS=$(date +%m%d_%H%M%S)

# nohup bash run.sh > run.log 2>&1 &

# LAR — no annealing (tau fixed at 1.0)
# python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_output_${TS}.txt

# LAR — with temperature annealing (tau: 1.0 → 0.1 over train_iters)
# python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_output_${TS}.txt

# LAR — no annealing + entropy regularisation (lambda=0.01)
python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_reg.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_reg_output_${TS}.txt

# LAR — annealing (tau: 1.0 → 0.1) + entropy regularisation (lambda=0.01)
python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_reg.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_reg_output_${TS}.txt
