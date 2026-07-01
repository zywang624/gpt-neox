TS=$(date +%m%d_%H%M%S)

# nohup bash run_temp.sh > run_temp.log 2>&1 &
# # # LAR — with temperature annealing (tau: 1.0 → 0.1 over train_iters)
# python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_output_${TS}.txt
# python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_reg.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_reg_output_${TS}.txt

# LAR — STE + p² reg (lambda=0.1)
# python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_ste.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_ste_output_${TS}.txt

# LAR — annealing (tau: 1.0 → 0.1) + entropy reg (lambda=0.01)
# python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_reg.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_reg_output_${TS}.txt

# LAR — static 4H+2L (layers 0-3 → d_high=128, layers 4-5 → d_low=64)
# python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_static_4h2l.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_static_4h2l_output_${TS}.txt

# LAR — delayed reg/anneal at 90% (step 17165): CE-only for first 90%, then H(p) reg + annealing for last ~1908 steps
python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_rentropy_tau001_dreg90pct.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_rentropy_tau001_dreg90pct_output_${TS}.txt
