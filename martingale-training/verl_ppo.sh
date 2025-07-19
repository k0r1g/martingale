#!/usr/bin/env bash
set -eux

# Load environment variables from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "Missing .env file with WANDB_API_KEY and HF_TOKEN" && exit 1
fi

# logging
echo "Loaded credentials for WandB: $WANDB_ENTITY and HuggingFace user: Kogero"

# Prepare dataset
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

# prefetch the model to avoid download during training
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
"

# PPO training run (2 epochs)
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
  trainer.logger='["console","wandb"]' \
  trainer.project_name=ppo_gsm8k_minimal \
  trainer.experiment_name=qwen_light_run \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=5 \
  trainer.test_freq=5 \
  trainer.total_epochs=2 \
  2>&1 | tee verl_light.log

# Merge actor checkpoint to Hugging Face-compatible format
CHECKPOINT_DIR=$(ls -td checkpoints/ppo_gsm8k_minimal/qwen_light_run/global_step_*/ | head -n1)
HF_DIR="$CHECKPOINT_DIR/actor/hf_model"

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$CHECKPOINT_DIR/actor" \
  --target_dir "$HF_DIR"

# Push model + tokenizer to Hugging Face Hub
python3 - <<EOF
from huggingface_hub import HfApi, login
from transformers import AutoModelForCausalLM, AutoTokenizer

login(token="${HF_TOKEN}")
api = HfApi()

repo_id = "Kogero/ppo-qwen-gsm8k"
local_dir = "$HF_DIR"

try:
    api.create_repo(repo_id, private=False)
except:
    pass  # already exists

AutoModelForCausalLM.from_pretrained(local_dir).push_to_hub(repo_id)
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").push_to_hub(repo_id)
EOF
