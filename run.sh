TS=$(date +%m%d_%H%M%S)

python deepy.py train.py pythia-70m-deduped_ddim_d128_w128.yml  2>&1 | tee pythia-70m-deduped_ddim_d128_w128_arc1e-3_output_${TS}.txt
