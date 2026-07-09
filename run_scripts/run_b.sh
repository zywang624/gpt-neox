TS=$(date +%m%d_%H%M%S)

# nohup bash run_b.sh > run_b.log 2>&1 &

python deepy.py train.py pythia_no_gelu_act/pythia-70m-deduped_lskv_d64_w128_no_gelu_act.yml  2>&1 | tee pythia_no_gelu_act/pythia-70m-deduped_lskv_d64_w128_no_gelu_act_output_${TS}.txt
python deepy.py train.py pythia_no_gelu_act/pythia-70m-deduped_lskv_d128_w64_no_gelu_act.yml  2>&1 | tee pythia_no_gelu_act/pythia-70m-deduped_lskv_d128_w64_no_gelu_act_output_${TS}.txt
