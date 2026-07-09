#!/usr/bin/env bash
# C4-only eval sweep for official EleutherAI pythia-* checkpoints (deduped and
# non-deduped), across attn_implementation x dtype combinations.
#
# Motivation: these checkpoints are numerically unstable under reduced
# precision, but HOW unstable depends heavily on attn_implementation, not
# just dtype. Quick sanity check on a fixed text sample (loss/ppl, not full
# C4) showed:
#   attn=eager dtype=float16   -> NaN on every model
#   attn=sdpa  dtype=float16   -> fine, close to fp32 (e.g. 160m-deduped:
#                                 78.37 vs fp32's 78.70)
#   attn=*     dtype=bfloat16  -> degraded on both, sdpa less bad than eager
# So this sweep exists to check whether that pattern holds on the real C4
# eval set, not just one sentence.
#
# Output naming encodes both knobs so combinations never clobber each other:
#   eval_results/official_pythia_reference/<model>/c4_<attn>_<dtype>/...
#   eval_results/official_pythia_reference/<model>/eval_c4_<attn>_<dtype>_<timestamp>.log
#
# CHECKPOINT_STEP picks an intermediate Pythia training checkpoint instead of
# the final release -- EleutherAI publishes these as HF revisions named
# "step<N>" (step0, step1, ..., step143000) on the SAME repo, loaded via
# revision=stepN. Leave unset to use the final checkpoint (revision=main).
#   CHECKPOINT_STEP=5000 bash eval_official_pythia_c4.sh   # step5000 checkpoint
#   bash eval_official_pythia_c4.sh                        # final checkpoint
#
# nohup bash eval_official_pythia_c4.sh > eval_official_pythia_c4.log 2>&1 &

set -euo pipefail

# HF Hub repo ids -- no local download needed, lm_eval/from_pretrained
# fetches + caches these automatically (~/.cache/huggingface/hub).
MODEL_PATHS=(
  "EleutherAI/pythia-70m-deduped"
  "EleutherAI/pythia-160m-deduped"
  "EleutherAI/pythia-70m"
  "EleutherAI/pythia-160m"
)

ATTN_IMPLS=(eager sdpa)
DTYPES=(float32 bfloat16 float16)
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"

REVISION_ARG=""
STEP_TAG=""
if [[ -n "$CHECKPOINT_STEP" ]]; then
  REVISION_ARG=",revision=step${CHECKPOINT_STEP}"
  STEP_TAG="_step${CHECKPOINT_STEP}"
fi

TS=$(date +%m%d_%H%M%S)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH")${STEP_TAG}"
  RESULT_DIR="./eval_results/official_pythia_reference/$MODEL_NAME"
  mkdir -p "$RESULT_DIR"

  for ATTN in "${ATTN_IMPLS[@]}"; do
    for DTYPE in "${DTYPES[@]}"; do
      TAG="${ATTN}_${DTYPE}"
      LOGFILE="$RESULT_DIR/eval_c4_${TAG}_${TS}.log"

      echo "==============================" | tee -a "$LOGFILE"
      echo "Running C4 eval for: $MODEL_PATH  (attn=$ATTN dtype=$DTYPE step=${CHECKPOINT_STEP:-final})" | tee -a "$LOGFILE"
      echo "==============================" | tee -a "$LOGFILE"

      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        /share/edc/home/zwang796/miniconda3/envs/llm_eval/bin/accelerate launch --num_processes 8 \
        -m lm_eval --model hf \
        --model_args pretrained="$MODEL_PATH",trust_remote_code=True,max_length=2048,attn_implementation="$ATTN",dtype="$DTYPE"${REVISION_ARG} \
        --tasks c4 \
        --batch_size 8 \
        --output_path "$RESULT_DIR/c4_${TAG}" \
        2>&1 | tee -a "$LOGFILE"
    done
  done
done
