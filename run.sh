TS=$(date +%m%d_%H%M%S)

# python deepy.py train.py pythia-70m-deduped.yml  2>&1 | tee pythia-70m-deduped_vanilla_aligned_config_output_${TS}.txt

# python deepy.py train.py pythia-70m-deduped_lskv_d128_w128.yml  2>&1 | tee pythia-70m-deduped_lskv_d128_w128_aligned_config_output_${TS}.txt

# python deepy.py train.py pythia-70m-deduped_lskv_d128_w0.yml  2>&1 | tee pythia-70m-deduped_lskv_d128_w0_aligned_config_output_${TS}.txt

# python deepy.py train.py pythia-70m-deduped_lskv_d256_w128.yml  2>&1 | tee pythia-70m-deduped_lskv_d256_w128_aligned_config_output_${TS}.txt

# python deepy.py train.py pythia-70m-deduped_lskv_d64_w128.yml  2>&1 | tee pythia-70m-deduped_lskv_d64_w128_aligned_config_output_${TS}.txt

# python deepy.py train.py pythia-70m-deduped_lskv_d256_w0.yml  2>&1 | tee pythia-70m-deduped_lskv_d256_w0_aligned_config_output_${TS}.txt

python deepy.py train.py pythia-70m-deduped_lskv_d1_w128.yml  2>&1 | tee pythia-70m-deduped_lskv_d1_w128_aligned_config_output_${TS}.txt
