TS=$(date +%m%d_%H%M%S)
MASTER_PORT=29637 python deepy.py train.py pythia-70m-deduped_lskv_d128_w0.yml  2>&1 | tee pythia-70m-deduped_lskv_d128_w0_aligned_config_output_${TS}.txt
