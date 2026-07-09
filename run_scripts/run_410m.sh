TS=$(date +%m%d_%H%M%S)

# nohup bash run_410m.sh > run_410m.log 2>&1 &

python deepy.py train.py pythia_no_gelu_act/pythia-410m-deduped_lskv_d256_w128_no_gelu_act.yml   2>&1 | tee pythia_no_gelu_act/pythia-410m-deduped_lskv_d256_w128_no_gelu_act_output_${TS}.txt
