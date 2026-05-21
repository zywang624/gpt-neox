TS=$(date +%m%d_%H%M%S)

MASTER_PORT=29638 python deepy.py train.py pythia-70m-deduped_ddim_d128_w128_alpha_ste.yml  2>&1 | tee pythia-70m-deduped_ddim_d128_w128_alpha_ste_output_${TS}.txt
