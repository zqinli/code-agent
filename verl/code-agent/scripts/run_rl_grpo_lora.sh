#!/usr/bin/env bash
# RL | code-agent | GRPO + LoRA | low-disk / low-checkpoint variant
#
# Goals:
# 1) keep LoRA RL training behavior
# 2) avoid large mid-run checkpoints by default
# 3) if checkpointing is enabled, save only model shards and keep very few ckpts
#
# Default behavior in this version:
# - SAVE_FREQ=-1            -> disable trainer checkpoints by default
# - RESUME_MODE=disable     -> avoid accidental auto-resume from old checkpoints
# - CHECKPOINT_SAVE_CONTENTS=[model] -> if you enable checkpointing, do not save optimizer/extra
# - MAX_ACTOR_CKPT_TO_KEEP=1 -> if you enable checkpointing, keep only the latest actor checkpoint
#
# Examples:
#   # safest: no checkpoints, just train
#   bash run_rl_grpo_lora_lowdisk.sh
#
#   # sparse model-only checkpoints every 100 steps, keep only 1
#   SAVE_FREQ=100 RESUME_MODE=auto bash run_rl_grpo_lora_lowdisk.sh
#
#   # resume from a previous model-only checkpoint
#   RESUME_FROM_PATH=/root/autodl-tmp/outputs/code-agent-rl/qwen2_5_coder_grpo_lora_agent/global_step_100 \
#   RESUME_MODE=resume_path SAVE_FREQ=100 bash run_rl_grpo_lora_lowdisk.sh
#
# Notes:
# - In verl FSDP, checkpoint save_contents must include 'model'.
# - Model-only checkpoints are smaller, but they are not the same as PEFT adapter-only export.
# - If you need a final adapter artifact for inference, export it separately after training.

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_AGENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERL_ROOT="$(cd "${CODE_AGENT_ROOT}/.." && pwd)"

export PYTHONPATH="${VERL_ROOT}:${CODE_AGENT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ---- paths ----
MODEL_DIR=${MODEL_DIR:-/root/autodl-tmp/models}
MODEL_PATH=${MODEL_PATH:-${MODEL_DIR}/Qwen2.5-Coder-3B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-/root/autodl-tmp/datasets/processed/verl_rl/train.parquet}
VAL_FILE=${VAL_FILE:-/root/autodl-tmp/datasets/processed/verl_eval/test.parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/outputs}
AGENT_LOOP_CONFIG=${AGENT_LOOP_CONFIG:-${CODE_AGENT_ROOT}/configs/agent_loop_config.yaml}
REWARD_PATH=${REWARD_PATH:-${CODE_AGENT_ROOT}/code_agent/rewards/code_agent_reward.py}
RAG_CORPUS=${RAG_CORPUS:-/root/autodl-tmp/datasets/processed/rag_final/corpus.jsonl}
RAG_INDEX_DIR=${RAG_INDEX_DIR:-/root/autodl-tmp/datasets/processed/rag_final/index}
BGE_MODEL=${BGE_MODEL:-/root/autodl-tmp/models/bge-m3}
MILVUS_URI=${MILVUS_URI:-${RAG_INDEX_DIR}/milvus_lite.db}
MILVUS_COLLECTION=${MILVUS_COLLECTION:-code_rag_bge_m3}

# ---- distributed / engine ----
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
INFER_BACKEND=${INFER_BACKEND:-vllm}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.75}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-true}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-2}

# ---- data / rollout ----
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-7168}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_N=${ROLLOUT_N:-4}

# ---- optimization ----
ACTOR_LR=${ACTOR_LR:-5e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-false}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-false}

# ---- LoRA ----
USE_PEFT=${USE_PEFT:-1}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_TARGETS=${LORA_TARGETS:-all-linear}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-/root/autodl-tmp/outputs/code-agent-sft/qwen2_5_coder_lora_sft/merged_step_27/lora_adapter}

# ---- checkpoint / resume (low-disk defaults) ----
# Disable trainer checkpoints by default. Set SAVE_FREQ>0 to enable sparse checkpoints.
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:--1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}

# Safer default: do not auto-resume from whatever is in the output dir.
RESUME_MODE=${RESUME_MODE:-disable}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

# If you explicitly provide a resume path, switch to resume_path automatically.
if [ -n "${RESUME_FROM_PATH}" ] && [ "${RESUME_MODE}" = "disable" ]; then
    RESUME_MODE=resume_path
fi

# If checkpointing is enabled, keep it minimal.
CHECKPOINT_SAVE_CONTENTS=${CHECKPOINT_SAVE_CONTENTS:-[model]}
CHECKPOINT_LOAD_CONTENTS=${CHECKPOINT_LOAD_CONTENTS:-[model]}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}

# ---- code-agent env knobs ----
export CODE_AGENT_RAG_CORPUS="${RAG_CORPUS}"
export CODE_AGENT_RAG_INDEX_DIR="${RAG_INDEX_DIR}"
export CODE_AGENT_BGE_MODEL="${BGE_MODEL}"
export CODE_AGENT_BGE_DEVICE="${CODE_AGENT_BGE_DEVICE:-cuda:0}"
export CODE_AGENT_BGE_USE_FP16="${CODE_AGENT_BGE_USE_FP16:-1}"
export CODE_AGENT_MILVUS_URI="${MILVUS_URI}"
export CODE_AGENT_MILVUS_COLLECTION="${MILVUS_COLLECTION}"
export CODE_AGENT_MAX_TURNS="${CODE_AGENT_MAX_TURNS:-2}"
export CODE_AGENT_MAX_OBS_LENGTH="${CODE_AGENT_MAX_OBS_LENGTH:-1024}"
export CODE_AGENT_ENABLE_FINAL_ROLLOUT="${CODE_AGENT_ENABLE_FINAL_ROLLOUT:-1}"
export CODE_AGENT_TOOL_TIMEOUT="${CODE_AGENT_TOOL_TIMEOUT:-10}"
export CODE_AGENT_REWARD_TIMEOUT="${CODE_AGENT_REWARD_TIMEOUT:-10}"
export CODE_AGENT_SEARCH_TOP_K="${CODE_AGENT_SEARCH_TOP_K:-3}"

# ---- logging / output ----
PROJECT_NAME=${PROJECT_NAME:-code-agent-rl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_coder_grpo_lora_agent}
SAVE_PATH=${SAVE_PATH:-${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}}
LOGGER=${LOGGER:-'["console"]'}

if [ ! -d "${MODEL_PATH}" ]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    echo "Set MODEL_PATH to a base/HF model directory with tokenizer files." >&2
    exit 1
fi
if [ -n "${RESUME_FROM_PATH}" ] && [ ! -d "${RESUME_FROM_PATH}" ]; then
    echo "RESUME_FROM_PATH does not exist: ${RESUME_FROM_PATH}" >&2
    echo "Set RESUME_FROM_PATH to a verl RL checkpoint directory like global_step_100." >&2
    exit 1
fi
if [ -n "${LORA_ADAPTER_PATH}" ] && [ ! -d "${LORA_ADAPTER_PATH}" ]; then
    echo "LORA_ADAPTER_PATH does not exist: ${LORA_ADAPTER_PATH}" >&2
    echo "Export the SFT checkpoint to a PEFT adapter first, then point LORA_ADAPTER_PATH to that adapter directory." >&2
    exit 1
fi
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "TRAIN_FILE does not exist: ${TRAIN_FILE}" >&2
    exit 1
fi
if [ ! -f "${VAL_FILE}" ]; then
    echo "VAL_FILE does not exist: ${VAL_FILE}" >&2
    exit 1
fi
if [ ! -f "${AGENT_LOOP_CONFIG}" ]; then
    echo "AGENT_LOOP_CONFIG does not exist: ${AGENT_LOOP_CONFIG}" >&2
    exit 1
fi
if [ ! -f "${REWARD_PATH}" ]; then
    echo "REWARD_PATH does not exist: ${REWARD_PATH}" >&2
    exit 1
fi

mkdir -p "${SAVE_PATH}"

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.return_raw_chat=True
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

if [ -n "${LORA_ADAPTER_PATH}" ]; then
    MODEL+=(
        actor_rollout_ref.model.lora_adapter_path="${LORA_ADAPTER_PATH}"
    )
fi

if [ "${USE_PEFT}" = "1" ]; then
    MODEL+=(
        actor_rollout_ref.model.lora_rank="${LORA_RANK}"
        actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
        actor_rollout_ref.model.target_modules="${LORA_TARGETS}"
    )
fi

ACTOR=(
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}"
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
    actor_rollout_ref.actor.checkpoint.save_contents="${CHECKPOINT_SAVE_CONTENTS}"
    actor_rollout_ref.actor.checkpoint.load_contents="${CHECKPOINT_LOAD_CONTENTS}"
)

ROLLOUT=(
    actor_rollout_ref.rollout.name="${INFER_BACKEND}"
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.rollout.agent.default_agent_loop=code_search_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}"
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}"
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

REWARD=(
    reward.custom_reward_function.path="${REWARD_PATH}"
    reward.custom_reward_function.name=compute_score
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger="${LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.default_local_dir="${SAVE_PATH}"
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.val_before_train="${VAL_BEFORE_TRAIN}"
    trainer.log_val_generations="${LOG_VAL_GENERATIONS}"
    trainer.resume_mode="${RESUME_MODE}"
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}"
)

if [ -n "${RESUME_FROM_PATH}" ]; then
    TRAINER+=(
        trainer.resume_from_path="${RESUME_FROM_PATH}"
    )
fi

cd "${VERL_ROOT}"

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
