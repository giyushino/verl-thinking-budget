#!/bin/bash

set -x
export HYDRA_FULL_ERROR=1

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."
echo "Working directory: $PWD"

# Add src to PYTHONPATH so 'thinking' module can be found
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Configuration
MODEL_NAME=Qwen/Qwen3-0.6B
LORA_PATH=/home/allanz/verl-thinking-budget/checkpoints/thinking/new_poly_2048_2048/global_step_112/actor/lora_adapter/
THINKING_BUDGET=2048
RESPONSE_BUDGET=2048
THINKING_END_TOKEN="</think>"
THINKING_END_TOKEN_ID=151668
BATCH_SIZE=32
TEMPERATURE=1.0
N_SAMPLES=10
K_VALUES="1 10"
MODE=eval
SAVE_NAME=calc_mixed
SAVE_FOLDER=new_poly_0_6
SAVE_PATH=$PWD/benchmarks
GPU_MEM_UTIL=0.9
DATASET_NAMES=calc_mixed
DATASET_PATHS=none

CUDA_VISIBLE_DEVICES=1 python -m thinking.eval \
    --model_name $MODEL_NAME \
    --lora_path $LORA_PATH \
    --thinking_budget $THINKING_BUDGET \
    --response_budget $RESPONSE_BUDGET \
    --thinking_end_token "$THINKING_END_TOKEN" \
    --thinking_end_token_id $THINKING_END_TOKEN_ID \
    --batch_size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --n_samples $N_SAMPLES \
    --k_values $K_VALUES \
    --mode $MODE \
    --save_name $SAVE_NAME \
    --save_path $SAVE_PATH/$SAVE_FOLDER \
    --gpu_memory_utilization $GPU_MEM_UTIL \
    --dataset_names $DATASET_NAMES \
    --dataset_paths $DATASET_PATHS \
    --force
    $@
