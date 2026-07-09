TS=$(date +%m%d_%H%M%S)

# nohup bash run_160m.sh > run_160m.log 2>&1 &

# ===== LAR 160M (best 70M config: Anneal+Reg tau001) =====
python deepy.py train.py pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_anneal_reg_tau001.yml 2>&1 | tee pythia_layer_wise/pythia-160m-deduped_lar_d192_d96_w128_anneal_reg_tau001_output_${TS}.txt

# ===== old experiments (already done) =====
# python deepy.py train.py pythia_no_gelu_act/pythia-160m-deduped_lskv_d192_w128_no_gelu_act.yml 2>&1 | tee pythia_no_gelu_act/pythia-160m-deduped_lskv_d192_w128_no_gelu_act_output_${TS}.txt

