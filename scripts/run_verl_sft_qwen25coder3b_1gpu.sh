#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_SHM_DISABLE=0
export CUDA_MODULE_LOADING=LAZY

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/dataset/data/swegym_full_project}"
MODEL="${MODEL:-${PROJECT_ROOT}/models/Qwen2.5-Coder-3B-Instruct-sdpa}"
TRAIN="${TRAIN:-${DATA_ROOT}/verl_sft_8192/train.parquet}"
VAL="${VAL:-${DATA_ROOT}/verl_sft_8192/val.parquet}"
OUT="${OUT:-${PROJECT_ROOT}/outputs/qwen25_coder_3b_swegym_verl_lora_sft_8192}"

mkdir -p "$OUT"

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m verl.trainer.sft_trainer \
  data.train_files="$TRAIN" \
  data.val_files="$VAL" \
  data.messages_key=messages \
  data.train_batch_size=1 \
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
