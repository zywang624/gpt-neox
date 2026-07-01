docker run --gpus '"device=0,1,2,3,4,5,6,7"' --user root -it \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --memory=160g \
    --memory-swap=160g \
    --mount type=bind,src=$PWD,dst=/gpt-neox \
    -v $(pwd):/workspace/ \
    -v /share/edc/home/zwang796/data:/share/edc/home/zwang796/data \
    pythia:latest bash