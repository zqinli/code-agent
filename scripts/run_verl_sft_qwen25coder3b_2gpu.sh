#!/usr/bin/env bash
set -xeuo pipefail

cd /home/zhenqinli/rl-workplace

export CUDA_VISIBLE_DEVICES=0,1
unset LD_LIBRARY_PATH

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export CUDA_MODULE_LOADING=LAZY

MODEL=/home/zhenqinli/rl-workplace/models/Qwen2.5-Coder-3B-Instruct-sdpa
TRAIN=/home/zhenqinli/rl-workplace/dataset/data/swegym_full_project/verl_sft_8192/train.parquet
VAL=/home/zhenqinli/rl-workplace/dataset/data/swegym_full_project/verl_sft_8192/val.parquet
OUT=/home/zhenqinli/rl-workplace/outputs/qwen25_coder_3b_swegym_verl_lora_sft_8192

mkdir -p "$OUT"

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m verl.trainer.sft_trainer \
  data.train_files="$TRAIN" \
  data.val_files="$VAL" \
  data.messages_key=messages \
  data.train_batch_size=4 \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=8192 \
  data.truncation=right \
  model.path="$MODEL" \
  model.trust_remote_code=True \
  model.enable_gradient_checkpointing=True \
  model.lora_rank=16 \
  optim.lr=2e-4 \
  optim.weight_decay=0.0 \
  optim.lr_warmup_steps_ratio=0.03 \
  trainer.project_name=swegym-code-agent-sft \
  trainer.experiment_name=qwen25-coder-3b-lora-sft-8192 \
  trainer.total_epochs=2 \
  trainer.default_local_dir="$OUT" \
  trainer.logger='["console"]'
