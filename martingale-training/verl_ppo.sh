#!/usr/bin/env bash
set -eux


export WANDB_API_KEY="<YOUR_WANDB_API_KEY_HERE>"
export WANDB_ENTITY="k0r1g"

export VERL_USE_MODELSCOPE=${VERL_USE_MODELSCOPE:-False}

# Prepare GSM8K dataset 
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

# (Optional) preload model
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); 
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
"

# Run PPO training
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files=~/data/gsm8k/train.parquet \
  data.val_files=~/data/gsm8k/test.parquet \
  data.train_batch_size=64 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  critic.ppo_micro_batch_size_per_gpu=2 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger='[\"console\",\"wandb\"]' \
  trainer.project_name=ppo_gsm8k_minimal \
  trainer.experiment_name=qwen_light_run \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=5 \
  trainer.test_freq=5 \
  trainer.total_epochs=2 \
  2>&1 | tee verl_light.log
