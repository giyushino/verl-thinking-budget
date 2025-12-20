# verl-thinking-budget

Train LLMs with constrained chain-of-thought reasoning using GRPO. Built off of verl 

## Overview

This project implements **two-phase generation with budget constraints**:

1. **Thinking Phase**: Generate up to `thinking_budget` tokens for chain-of-thought reasoning
2. **Response Phase**: Generate up to `response_budget` tokens for the final answer

## Key Files

- `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py` - Two-phase generation implementation

## Installation

```bash
git clone git@github.com:giyushino/verl-thinking-budget.git
cd verl-thinking-budget
conda create -n thinking python==3.12
conda activate thinking

# Install VeRL framework
cd verl
bash scripts/install_vllm_sglang_mcore.sh
pip install -e .
cd ..

# Install project and dependencies
pip install -e .
uv pip install vllm==0.11.0 --no-cache
uv pip install flash-attn --no-build-isolation
```

## Usage

### Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python verl/verl/trainer/main_ppo.py \
    algorithm.adv_estimator=grpo \
    data.train_files=$PWD/datasets/new_poly/train.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.rollout.thinking_budget=2048 \
    actor_rollout_ref.rollout.thinking_delimiter_id=151668 \
    data.max_response_length=4096 \
    custom_reward_function.path=$PWD/src/thinking/grader.py \
    trainer.project_name='thinking' \
    trainer.experiment_name='my_experiment'
```

See `scripts/budget_train.sh` for a full example.

### Evaluation

```bash
python -m thinking.eval_faster \
    --model_name Qwen/Qwen3-0.6B \
    --lora_path /path/to/lora/adapter/ \
    --thinking_budget 2048 \
    --response_budget 2048 \
    --thinking_end_token "</think>" \
    --thinking_end_token_id 151668 \
    --batch_size 32 \
    --n_samples 10 \
    --k_values 1 10 \
    --dataset_names math12k gsm8k \
    --save_path ./benchmarks
```

See `scripts/eval.sh` for more options.

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `thinking_budget` | Max tokens for the thinking/reasoning phase |
| `response_budget` | Max tokens for the response phase |
| `thinking_delimiter_id` | Token ID for `</think>` (151668 for Qwen3) |
| `n_samples` | Samples per problem for pass@k evaluation |


## Project Structure

```
├── src/thinking/
│   ├── eval_faster.py    # Evaluation with budget constraints
│   ├── data.py           # Dataset loaders
│   └── grader.py         # Math answer grading
├── scripts/
│   ├── budget_train.sh   # Training script
│   └── eval.sh           # Evaluation script
├── verl/                  # VeRL framework
└── datasets/              # Training data (parquet format)
```

## Notes

- If the model doesn't generate `</think>` within the thinking budget, it's automatically appended
- LoRA adapters are saved in the checkpoint directory and can be loaded via `--lora_path`
