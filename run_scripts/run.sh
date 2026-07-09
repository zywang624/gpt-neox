TS=$(date +%m%d_%H%M%S)

# nohup bash run.sh > run.log 2>&1 &

# LAR — no annealing (tau fixed at 1.0)
# python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_output_${TS}.txt

# LAR — with temperature annealing (tau: 1.0 → 0.1 over train_iters)
# python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_output_${TS}.txt


# LAR — annealing (tau: 1.0 → 0.01) + p² reg (lambda=0.1)
# python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal2.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal2_output_${TS}.txt

# LAR — no annealing + entropy reg (lambda=0.01)
# python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_reg.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_reg_output_${TS}.txt

# # LAR — STE + p² regularisation (lambda=1.0)
# python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_ste.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_ste_output_${TS}.txt

# LAR — annealing (tau: 1.0 → 0.01) only, no reg
# python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_tau001.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_tau001_output_${TS}.txt

# LAR — annealing (tau: 1.0 → 0.01) + entropy reg (lambda=0.01)
# python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_reg_tau001.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_reg_tau001_output_${TS}.txt

# LAR — annealing (tau: 1.0 → 0.01) + routing entropy H(p) (lambda=0.004, ~= p*(1-p) lambda=0.01 in strength)
# python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_rentropy_tau001.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_rentropy_tau001_output_${TS}.txt

# LAR — delayed reg/anneal at 10% (step 1907): CE-only for first 10%, then H(p) reg + annealing for remaining 90%
python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_rentropy_tau001_dreg10pct.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lar_d128_d64_w128_anneal_rentropy_tau001_dreg10pct_output_${TS}.txt
