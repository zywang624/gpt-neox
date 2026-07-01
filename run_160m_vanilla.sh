TS=$(date +%m%d_%H%M%S)

# nohup bash run_160m_vanilla.sh > run_160m_vanilla.log 2>&1 &

python deepy.py train.py pythia_no_gelu_act/pythia-160m-deduped_lskv_vanilla.yml 2>&1 | tee pythia_no_gelu_act/pythia-160m-deduped_lskv_vanilla_output_${TS}.txt
