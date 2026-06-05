#!/usr/bin/env bash
set -euo pipefail

# Offline batch inference on the processed verl test set with vLLM.
#
# Run after exporting the RL checkpoint:
#   conda activate verl
#   bash /root/rl-workplace/verl/code-agent/scripts/merge_rl_lora_checkpoint.sh
#   bash /root/rl-workplace/verl/code-agent/scripts/infer_testset_vllm.sh

PROJECT_ROOT="${PROJECT_ROOT:-/root/rl-workplace/verl}"
CODE_AGENT_ROOT="${CODE_AGENT_ROOT:-${PROJECT_ROOT}/code-agent}"

TEST_FILE="${TEST_FILE:-/root/autodl-tmp/datasets/processed/verl_eval/test.parquet}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-Coder-3B-Instruct}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-/root/autodl-tmp/outputs/code-agent-rl/qwen2_5_coder_grpo_lora_agent/merged_step_31/lora_adapter}"
OUTPUT_FILE="${OUTPUT_FILE:-/root/autodl-tmp/outputs/code-agent-rl/qwen2_5_coder_grpo_lora_agent/infer_test_step31.jsonl}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-code-agent-rl}"
USE_LORA="${USE_LORA:-true}"

LIMIT="${LIMIT:--1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:--1}"

DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${TEST_FILE}" ]]; then
  echo "[infer] ERROR: TEST_FILE does not exist: ${TEST_FILE}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[infer] ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ "${USE_LORA}" == "true" ]]; then
  if [[ ! -f "${LORA_ADAPTER_PATH}/adapter_config.json" ]]; then
    echo "[infer] ERROR: LoRA adapter not found: ${LORA_ADAPTER_PATH}" >&2
    echo "[infer] Run scripts/merge_rl_lora_checkpoint.sh first, or override LORA_ADAPTER_PATH." >&2
    exit 1
  fi
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${CODE_AGENT_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [[ "${ENFORCE_EAGER}" == "true" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ "${USE_LORA}" == "true" ]]; then
  EXTRA_ARGS+=(--lora-adapter-path "${LORA_ADAPTER_PATH}" --served-model-name "${SERVED_MODEL_NAME}")
fi

"${PYTHON_BIN}" -m code_agent.serve.offline_infer_testset \
  --test-file "${TEST_FILE}" \
  --output-file "${OUTPUT_FILE}" \
  --model-path "${MODEL_PATH}" \
  --limit "${LIMIT}" \
  --batch-size "${BATCH_SIZE}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  "${EXTRA_ARGS[@]}"

echo "[infer] output: ${OUTPUT_FILE}"
