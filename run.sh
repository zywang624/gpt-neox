TS=$(date +%m%d_%H%M%S)

# nohup bash run.sh > run.log 2>&1 &
# python deepy.py train.py pythia-70m-deduped_ddim_d128_w128_random_alpha_lookup.yml  2>&1 | tee pythia-70m-deduped_ddim_d128_w128_random_alpha_lookup_output_${TS}.txt
# python deepy.py train.py pythia-70m-deduped_ddim_d128_w128_frozen_alpha_lookup_inverted.yml  2>&1 | tee pythia-70m-deduped_ddim_d128_w128_frozen_alpha_lookup_inverted_output_${TS}.txt

# python deepy.py train.py pythia-70m-deduped_lskv_d100_w128.yml 2>&1 | tee pythia-70m-deduped_lskv_d100_w128_output_${TS}.txt

python deepy.py train.py pythia-160m-deduped_lskv_d148_w128.yml 2>&1 | tee pythia-160m-deduped_lskv_d148_w128_output_${TS}.txt