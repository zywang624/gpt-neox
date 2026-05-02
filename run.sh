TS=$(date +%m%d_%H%M%S)

# python deepy.py train.py pythia-160m-deduped.yml  2>&1 | tee output.txt
# python deepy.py train.py pythia-70m-deduped.yml  2>&1 | tee output.txt
# python deepy.py train.py pythia-70m-deduped_lskv.yml  2>&1 | tee pythia-70m-deduped_lskv_d256_w128_wo_act_bias_output_${TS}.txt
# python deepy.py train.py pythia-70m-deduped_lskv_d256_w1.yml  2>&1 | tee pythia-70m-deduped_lskv_d256_w1_wo_act_bias_output_${TS}.txt
# python deepy.py train.py pythia-70m-deduped_lskv_d128_w128.yml  2>&1 | tee pythia-70m-deduped_lskv_d128_w128_wo_act_bias_init_output_${TS}.txt
# python deepy.py train.py pythia-70m-deduped_lskv_d64_w128.yml  2>&1 | tee pythia-70m-deduped_lskv_d64_w128_wo_act_bias_output_${TS}.txt
# python deepy.py train.py pythia-70m-deduped_lskv_d128_w1.yml  2>&1 | tee pythia-70m-deduped_lskv_d128_w1_wo_act_bias_output_${TS}.txt
# python deepy.py train.py pythia-70m-deduped_lskv_d256_w96.yml  2>&1 | tee pythia-70m-deduped_lskv_d128_w96_wo_act_bias_output_${TS}.txt
# # python deepy.py train.py pythia-70m-deduped_lskv_d256_w160.yml  2>&1 | tee pythia-70m-deduped_lskv_d128_w160_wo_act_bias_output_${TS}.txt
# python deepy.py train.py pythia-70m-deduped_lskv_d128_w160.yml  2>&1 | tee pythia-70m-deduped_lskv_d128_w160_wo_act_bias_output_${TS}.txt

# python deepy.py train.py pythia-160m-deduped.yml  2>&1 | tee pythia-160m-deduped_vanilla_output_${TS}.txt

# python deepy.py train.py pythia-410m-deduped.yml  2>&1 | tee pythia-410m-deduped_vanilla_output_${TS}.txt

# python deepy.py train.py pythia-1b-deduped_lskv_d512_w128.yml  2>&1 | tee pythia-1b-deduped_lskv_d512_w128_wo_act_bias_output_${TS}.txt


# python deepy.py train.py pythia-70m-deduped_lskv_d256_w96.yml  2>&1 | tee pythia-70m-deduped_lskv_d256_w96_wo_act_bias_output_${TS}.txt
# python deepy.py train.py pythia-70m-deduped_lskv_d256_w160.yml  2>&1 | tee pythia-70m-deduped_lskv_d256_w160_wo_act_bias_output_${TS}.txt

# python deepy.py train.py pythia-160m-deduped_lskv_d256_w128.yml  2>&1 | tee pythia-160m-deduped_lskv_d256_w128_wo_act_bias_output_${TS}.txt
# python deepy.py train.py pythia-160m-deduped_lskv_d192_w128.yml  2>&1 | tee pythia-160m-deduped_lskv_d192_w128_wo_act_bias_output_${TS}.txt

# python deepy.py train.py pythia-410m-deduped_lskv_d256_w128.yml  2>&1 | tee pythia-410m-deduped_lskv_d256_w128_wo_act_bias_output_${TS}.txt

# python deepy.py train.py pythia-70m-deduped_lskv_d128_w0_wo_act_bias.yml  2>&1 | tee pythia-70m-deduped_lskv_d128_w0_wo_act_bias_output_${TS}.txt

python deepy.py train.py pythia-70m-deduped.yml  2>&1 | tee pythia-70m-deduped_vanilla_aligned_config_output_${TS}.txt
