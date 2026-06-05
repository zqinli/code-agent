#!/usr/bin/env bash
set -euo pipefail

# Start vLLM with the RL LoRA adapter, then expose a FastAPI wrapper.
#
# Usage:
#   conda activate verl
#   bash /root/rl-workplace/verl/code-agent/scripts/serve_vllm_fastapi.sh
#
# Common overrides:
#   MODEL_PATH=/path/to/base \
#   LORA_ADAPTER_PATH=/path/to/lora_adapter \
#   MAX_MODEL_LEN=8192 \
#   bash scripts/serve_vllm_fastapi.sh

PROJECT_ROOT="${PROJECT_ROOT:-/root/rl-workplace/verl}"
CODE_AGENT_ROOT="${CODE_AGENT_ROOT:-${PROJECT_ROOT}/code-agent}"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-Coder-3B-Instruct}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-/root/autodl-tmp/outputs/code-agent-rl/qwen2_5_coder_grpo_lora_agent/merged_step_31/lora_adapter}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-code-agent-rl}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
FASTAPI_HOST="${FASTAPI_HOST:-0.0.0.0}"
FASTAPI_PORT="${FASTAPI_PORT:-8080}"

DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[serve] ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${LORA_ADAPTER_PATH}/adapter_config.json" ]]; then
  echo "[serve] ERROR: LoRA adapter not found: ${LORA_ADAPTER_PATH}" >&2
  echo "[serve] Run scripts/merge_rl_lora_checkpoint.sh first." >&2
  exit 1
fi

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

cd "${PROJECT_ROOT}"
export PYTHONPATH="${CODE_AGENT_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"
export SERVED_MODEL_NAME

echo "[serve] starting vLLM on ${VLLM_BASE_URL}"
echo "[serve] base model: ${MODEL_PATH}"
echo "[serve] lora:       ${SERVED_MODEL_NAME}=${LORA_ADAPTER_PATH}"

"${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank 32 \
  --lora-modules "${SERVED_MODEL_NAME}=${LORA_ADAPTER_PATH}" \
  --generation-config vllm \
  > "${VLLM_LOG:-/tmp/code_agent_vllm.log}" 2>&1 &
VLLM_PID="$!"

echo "[serve] waiting for vLLM to become healthy..."
for _ in $(seq 1 180); do
  if "${PYTHON_BIN}" - <<'PY'
import os
import urllib.request

url = os.environ["VLLM_BASE_URL"].rstrip("/") + "/v1/models"
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    break
  fi

  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "[serve] ERROR: vLLM exited early. Log follows:" >&2
    tail -n 120 "${VLLM_LOG:-/tmp/code_agent_vllm.log}" >&2 || true
    exit 1
  fi
  sleep 2
done

if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
  echo "[serve] ERROR: vLLM is not running." >&2
  exit 1
fi

echo "[serve] starting FastAPI on http://${FASTAPI_HOST}:${FASTAPI_PORT}"
echo "[serve] health: curl http://127.0.0.1:${FASTAPI_PORT}/health"

exec "${PYTHON_BIN}" -m uvicorn code_agent.serve.vllm_fastapi:app \
  --host "${FASTAPI_HOST}" \
  --port "${FASTAPI_PORT}" \
  --workers "${UVICORN_WORKERS}"
