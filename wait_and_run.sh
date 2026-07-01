#!/bin/bash
# Usage:
#   bash wait_and_run.sh <container_name_or_id>   # wait for docker container to stop
#   bash wait_and_run.sh                           # monitor GPU memory, run when all GPUs < 5GB

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_temp.sh"
MEM_THRESHOLD=5000  # MB
CHECK_INTERVAL=30   # seconds

if [ -n "$1" ]; then
    CONTAINER=$1
    echo "[$(date)] Waiting for container '$CONTAINER' to stop..."
    docker wait "$CONTAINER"
    echo "[$(date)] Container stopped. Waiting 30s for GPU memory to clear..."
    sleep 30
else
    echo "[$(date)] No container given. Monitoring GPU memory (threshold: ${MEM_THRESHOLD}MB)..."
    while true; do
        max_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        echo "[$(date)] Max GPU memory used: ${max_mem}MB"
        if [ "$max_mem" -lt "$MEM_THRESHOLD" ]; then
            break
        fi
        sleep $CHECK_INTERVAL
    done
fi

echo "[$(date)] GPU free. Starting run_temp.sh..."
cd "$SCRIPT_DIR"
bash run_temp.sh
