TS=$(date +%m%d_%H%M%S)

# nohup bash run.sh > run.log 2>&1 &
# python deepy.py train.py pythia-70m-deduped_ddim_d128_w128_random_alpha_lookup.yml  2>&1 | tee pythia-70m-deduped_ddim_d128_w128_random_alpha_lookup_output_${TS}.txt
# python deepy.py train.py pythia-70m-deduped_ddim_d128_w128_frozen_alpha_lookup_inverted.yml  2>&1 | tee pythia-70m-deduped_ddim_d128_w128_frozen_alpha_lookup_inverted_output_${TS}.txt

# python deepy.py train.py pythia-70m-deduped_lskv_d100_w128.yml 2>&1 | tee pythia-70m-deduped_lskv_d100_w128_output_${TS}.txt

# python deepy.py train.py pythia-160m-deduped_lskv_d148_w128.yml 2>&1 | tee pythia-160m-deduped_lskv_d148_w128_output_${TS}.txt

# python deepy.py train.py pythia-410m-deduped_lskv_d194_w128.yml 2>&1 | tee pythia-410m-deduped_lskv_d194_w128_output_${TS}.txt

# python deepy.py train.py pythia_no_gelu_act/pythia-70m-deduped_lskv_d128_w128_no_gelu_act.yml 2>&1 | tee pythia_no_gelu_act/pythia-70m-deduped_lskv_d128_w128_no_gelu_act_output_${TS}.txt
# python deepy.py train.py pythia_no_gelu_act/pythia-70m-deduped_lskv_d128_w1_no_gelu_act.yml   2>&1 | tee pythia_no_gelu_act/pythia-70m-deduped_lskv_d128_w1_no_gelu_act_output_${TS}.txt
# python deepy.py train.py pythia_no_gelu_act/pythia-70m-deduped_lskv_d128_w16_no_gelu_act.yml   2>&1 | tee pythia_no_gelu_act/pythia-70m-deduped_lskv_d128_w16_no_gelu_act_output_${TS}.txt
# python deepy.py train.py pythia_no_gelu_act/pythia-70m-deduped_lskv_d128_w0_no_gelu_act.yml   2>&1 | tee pythia_no_gelu_act/pythia-70m-deduped_lskv_d128_w0_no_gelu_act_output_${TS}.txt
# python deepy.py train.py pythia_no_gelu_act/pythia-70m-deduped_lskv_d128_w4_no_gelu_act.yml   2>&1 | tee pythia_no_gelu_act/pythia-70m-deduped_lskv_d128_w4_no_gelu_act_output_${TS}.txt

# python deepy.py train.py pythia_no_gelu_act/pythia-70m-deduped_lskv_d64_w0_no_gelu_act.yml   2>&1 | tee pythia_no_gelu_act/pythia-70m-deduped_lskv_d64_w0_no_gelu_act_output_${TS}.txt

# python deepy.py train.py pythia_no_gelu_act/pythia-70m-deduped_lskv_d256_w0_no_gelu_act.yml   2>&1 | tee pythia_no_gelu_act/pythia-70m-deduped_lskv_d256_w0_no_gelu_act_output_${TS}.txt

# pythia-70m w_act_wo_bias experiments
python deepy.py train.py pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d256_w128_w_act_wo_bias.yml 2>&1 | tee pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d256_w128_w_act_wo_bias_output_${TS}.txt
python deepy.py train.py pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d256_w1_w_act_wo_bias.yml   2>&1 | tee pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d256_w1_w_act_wo_bias_output_${TS}.txt
python deepy.py train.py pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w128_w_act_wo_bias.yml 2>&1 | tee pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w128_w_act_wo_bias_output_${TS}.txt
python deepy.py train.py pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w1_w_act_wo_bias.yml   2>&1 | tee pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w1_w_act_wo_bias_output_${TS}.txt

python deepy.py train.py pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w0_w_act_wo_bias.yml   2>&1 | tee pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w0_w_act_wo_bias_output_${TS}.txt
python deepy.py train.py pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w4_w_act_wo_bias.yml   2>&1 | tee pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w4_w_act_wo_bias_output_${TS}.txt
python deepy.py train.py pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w16_w_act_wo_bias.yml  2>&1 | tee pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w16_w_act_wo_bias_output_${TS}.txt
python deepy.py train.py pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w32_w_act_wo_bias.yml  2>&1 | tee pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w32_w_act_wo_bias_output_${TS}.txt

# move to 160m
# python deepy.py train.py pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w256_w_act_wo_bias.yml 2>&1 | tee pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w256_w_act_wo_bias_output_${TS}.txt
# python deepy.py train.py pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d256_w0_w_act_wo_bias.yml   2>&1 | tee pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d256_w0_w_act_wo_bias_output_${TS}.txt
# python deepy.py train.py pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w64_w_act_wo_bias.yml  2>&1 | tee pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w64_w_act_wo_bias_output_${TS}.txt
# python deepy.py train.py pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w2_w_act_wo_bias.yml   2>&1 | tee pythia_w_act_wo_bias/pythia-70m-deduped_lskv_d128_w2_w_act_wo_bias_output_${TS}.txt


#   nohup bash -c '
#   while true; do
#     max_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
#     if [ "$max_mem" -lt 5000 ]; then
#       cd /share/edc/home/zwang796/code/ls_kv/ppl/train_pythia/gpt-neox
#       bash run_temp.sh
#       break
#     fi
#     sleep 30
#   done
#   ' > /tmp/wait_and_run.log 2>&1 &
