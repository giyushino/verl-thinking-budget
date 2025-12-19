#!/bin/bash

set -x
#export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."
echo $PWD

CUDA_VISIBLE_DEVICES=0,1,2,3 python $PWD/verl/verl/trainer/main_ppo.py \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files=$PWD/datasets/new_poly/train.parquet \
    data.val_files=$PWD/datasets/new_poly/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=6000 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.thinking_budget=2048 \
    actor_rollout_ref.rollout.thinking_delimiter_id=151668 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    actor_rollout_ref.actor.checkpoint.save_contents=[optimizer,extra] \
    custom_reward_function.path=$PWD/src/thinking/grader.py \
    custom_reward_function.name=compute_score \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='thinking' \
    trainer.experiment_name='new_poly_0_6' \
    trainer.total_epochs=2 $@


