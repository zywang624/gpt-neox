TS=$(date +%m%d_%H%M%S)


# nohup bash run_vanilla.sh > run_vanilla.log 2>&1 &
python deepy.py train.py pythia_w_act_wo_bias/pythia-70-deduped.yml 2>&1 | tee pythia_w_act_wo_bias/pythia-70-deduped_vanilla_output_${TS}.txt
