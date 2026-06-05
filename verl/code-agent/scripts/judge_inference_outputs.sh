#!/usr/bin/env bash
set -euo pipefail

# Judge offline inference outputs with an OpenAI-compatible LLM API.
#
# First create a real env file:
#   cp /root/rl-workplace/verl/code-agent/configs/judge.env.example /root/rl-workplace/verl/code-agent/configs/judge.env
#   vim /root/rl-workplace/verl/code-agent/configs/judge.env
#
# Then run:
#   conda activate verl
#   bash /root/rl-workplace/verl/code-agent/scripts/judge_inference_outputs.sh

PROJECT_ROOT="${PROJECT_ROOT:-/root/rl-workplace/verl}"
CODE_AGENT_ROOT="${CODE_AGENT_ROOT:-${PROJECT_ROOT}/code-agent}"

ENV_FILE="${ENV_FILE:-${CODE_AGENT_ROOT}/configs/judge.env}"
INPUT_FILE="${INPUT_FILE:-/root/autodl-tmp/outputs/code-agent-rl/qwen2_5_coder_grpo_lora_agent/infer_test_step31.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-/root/autodl-tmp/outputs/code-agent-rl/qwen2_5_coder_grpo_lora_agent/judge_test_step31.jsonl}"

LIMIT="${LIMIT:--1}"
PROMPT_FIELD="${PROMPT_FIELD:-prompt}"
RESPONSE_FIELD="${RESPONSE_FIELD:-response}"
REFERENCE_FIELD="${REFERENCE_FIELD:-}"
ID_FIELD="${ID_FIELD:-id}"

JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1200}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-120}"
JUDGE_RETRIES="${JUDGE_RETRIES:-3}"
JUDGE_SLEEP="${JUDGE_SLEEP:-0}"
PASS_THRESHOLD="${PASS_THRESHOLD:-3.5}"

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "[judge] ERROR: INPUT_FILE does not exist: ${INPUT_FILE}" >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[judge] ERROR: ENV_FILE does not exist: ${ENV_FILE}" >&2
  echo "[judge] Create it from ${CODE_AGENT_ROOT}/configs/judge.env.example" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${CODE_AGENT_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [[ -n "${REFERENCE_FIELD}" ]]; then
  EXTRA_ARGS+=(--reference-field "${REFERENCE_FIELD}")
fi

"${PYTHON_BIN}" -m code_agent.judge.llm_code_judge \
  --env-file "${ENV_FILE}" \
  --input-file "${INPUT_FILE}" \
  --output-file "${OUTPUT_FILE}" \
  --prompt-field "${PROMPT_FIELD}" \
  --response-field "${RESPONSE_FIELD}" \
  --id-field "${ID_FIELD}" \
  --limit "${LIMIT}" \
  --temperature "${JUDGE_TEMPERATURE}" \
  --max-tokens "${JUDGE_MAX_TOKENS}" \
  --timeout "${JUDGE_TIMEOUT}" \
  --retries "${JUDGE_RETRIES}" \
  --sleep "${JUDGE_SLEEP}" \
  --pass-threshold "${PASS_THRESHOLD}" \
  "${EXTRA_ARGS[@]}"

echo "[judge] output: ${OUTPUT_FILE}"
