TS=$(date +%m%d_%H%M%S)

# nohup bash run_70m_per_layer.sh > run_70m_per_layer.log 2>&1 &

MASTER_PORT=29501 python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lskv_d128_w128_per_layer_2low.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lskv_d128_w128_per_layer_2low_output_${TS}.txt
MASTER_PORT=29501 python deepy.py train.py pythia_layer_wise/pythia-70m-deduped_lskv_d128_w128_per_layer_4low.yml 2>&1 | tee pythia_layer_wise/pythia-70m-deduped_lskv_d128_w128_per_layer_4low_output_${TS}.txt

