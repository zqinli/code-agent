#!/usr/bin/env bash
# SFT | code-agent protocol | LoRA | FSDP engine
#
# Default data:
#   /root/autodl-tmp/datasets/processed/verl_sft_answer_code_train150/train.parquet
#
# Usage:
#   bash /root/rl-workplace/verl/code-agent/scripts/run_sft_lora.sh
#
# Common overrides:
#   NPROC_PER_NODE=1 MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-Coder-0.5B-Instruct bash .../run_sft_lora.sh
#   TOTAL_EPOCHS=3 TRAIN_BATCH_SIZE=8 MICRO_BATCH_SIZE_PER_GPU=1 bash .../run_sft_lora.sh

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ---- paths ----
MODEL_DIR=${MODEL_DIR:-/root/autodl-tmp/models}
MODEL_PATH=${MODEL_PATH:-${MODEL_DIR}/Qwen2.5-Coder-3B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-/root/autodl-tmp/datasets/processed/verl_sft_answer_code_train150/train.parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/outputs}

# ---- distributed ----
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29501}
TORCHRUN=${TORCHRUN:-}
if [ -z "${TORCHRUN}" ]; then
    if [ -x /root/miniconda3/envs/verl/bin/torchrun ]; then
        TORCHRUN=/root/miniconda3/envs/verl/bin/torchrun
    else
        TORCHRUN=torchrun
    fi
fi

# ---- training ----
PROJECT_NAME=${PROJECT_NAME:-code-agent-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_coder_lora_sft}
SAVE_PATH=${SAVE_PATH:-${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
LR=${LR:-1e-4}

# Global batch = data.train_batch_size.
# Effective examples per optimizer step depend on verl dynamic batching and GPU count.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-8192}

SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}
LOGGER=${LOGGER:-'["console"]'}
SEED=${SEED:-20260512}

# ---- LoRA ----
USE_PEFT=${USE_PEFT:-1}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_TARGETS=${LORA_TARGETS:-all-linear}

# ---- memory knobs ----
SP_SIZE=${SP_SIZE:-1}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-true}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-true}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
FSDP_PARAM_OFFLOAD=${FSDP_PARAM_OFFLOAD:-false}
FSDP_OPTIMIZER_OFFLOAD=${FSDP_OPTIMIZER_OFFLOAD:-false}

if [ ! -d "${MODEL_PATH}" ]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    echo "Set MODEL_PATH to a local model under ${MODEL_DIR}." >&2
    exit 1
fi

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "TRAIN_FILE does not exist: ${TRAIN_FILE}" >&2
    exit 1
fi

mkdir -p "${SAVE_PATH}"

extra_args=()
if [ "${USE_PEFT}" = "1" ]; then
    extra_args+=(
        "model.lora_rank=${LORA_RANK}"
        "model.lora_alpha=${LORA_ALPHA}"
        "model.target_modules=${LORA_TARGETS}"
    )
fi

cd "${VERL_ROOT}"

"${TORCHRUN}" --standalone --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files=null \
    data.messages_key=messages \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.max_length="${MAX_LENGTH}" \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    data.truncation=error \
    data.ignore_input_ids_mismatch=True \
    data.num_workers=4 \
    optim.lr="${LR}" \
    model.path="${MODEL_PATH}" \
    +model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
    model.use_remove_padding="${USE_REMOVE_PADDING}" \
    model.enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING}" \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size="${SP_SIZE}" \
    engine.param_offload="${FSDP_PARAM_OFFLOAD}" \
    engine.optimizer_offload="${FSDP_OPTIMIZER_OFFLOAD}" \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="${LOGGER}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.seed="${SEED}" \
    trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.resume_mode=auto \
    "${extra_args[@]}" \
    "$@"
