TS=$(date +%m%d_%H%M%S)

# nohup bash run_a.sh > run_a.log 2>&1 &

python deepy.py train.py pythia_no_gelu_act/pythia-70m-deduped_lskv_d256_w128_no_gelu_act.yml 2>&1 | tee pythia_no_gelu_act/pythia-70m-deduped_lskv_d256_w128_no_gelu_act_output_${TS}.txt
python deepy.py train.py pythia_no_gelu_act/pythia-70m-deduped_lskv_d128_w256_no_gelu_act.yml 2>&1 | tee pythia_no_gelu_act/pythia-70m-deduped_lskv_d128_w256_no_gelu_act_output_${TS}.txt
