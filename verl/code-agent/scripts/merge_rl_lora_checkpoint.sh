#!/usr/bin/env bash
set -euo pipefail

# Export a verl FSDP actor checkpoint to HuggingFace format and LoRA adapter.
#
# Defaults target the latest finished GRPO run in this workspace. Override any of
# these from the command line if you want to export another step or experiment.

# bash /root/rl-workplace/verl/code-agent/scripts/merge_rl_lora_checkpoint.sh

# STEP=30 bash /root/rl-workplace/verl/code-agent/scripts/merge_rl_lora_checkpoint.sh


PROJECT_ROOT="${PROJECT_ROOT:-/root/rl-workplace/verl}"
RUN_DIR="${RUN_DIR:-/root/autodl-tmp/outputs/code-agent-rl/qwen2_5_coder_grpo_lora_agent}"
STEP="${STEP:-auto}"

if [[ -z "${LOCAL_DIR:-}" ]]; then
  if [[ "${STEP}" == "auto" ]]; then
    FOUND_STEP=""
    for step_dir in "${RUN_DIR}"/global_step_*; do
      [[ -d "${step_dir}/actor" ]] || continue
      step_name="$(basename "${step_dir}")"
      step_num="${step_name#global_step_}"
      [[ "${step_num}" =~ ^[0-9]+$ ]] || continue
      [[ -f "${step_dir}/actor/fsdp_config.json" ]] || continue
      compgen -G "${step_dir}/actor/model_world_size_*_rank_0.pt" >/dev/null || continue
      if [[ -z "${FOUND_STEP}" || "${step_num}" -gt "${FOUND_STEP}" ]]; then
        FOUND_STEP="${step_num}"
      fi
    done
    if [[ -z "${FOUND_STEP}" ]]; then
      echo "[merge] ERROR: no complete checkpoint found under ${RUN_DIR}" >&2
      exit 1
    fi
    STEP="${FOUND_STEP}"
  fi
  LOCAL_DIR="${RUN_DIR}/global_step_${STEP}/actor"
fi

TARGET_DIR="${TARGET_DIR:-${RUN_DIR}/merged_step_${STEP}}"

LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TASK_TYPE="${LORA_TASK_TYPE:-CAUSAL_LM}"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[merge] project root: ${PROJECT_ROOT}"
echo "[merge] local dir:    ${LOCAL_DIR}"
echo "[merge] target dir:   ${TARGET_DIR}"

if [[ ! -f "${LOCAL_DIR}/fsdp_config.json" ]]; then
  echo "[merge] ERROR: missing ${LOCAL_DIR}/fsdp_config.json" >&2
  exit 1
fi

if ! compgen -G "${LOCAL_DIR}/model_world_size_*_rank_0.pt" >/dev/null; then
  echo "[merge] ERROR: missing FSDP model shard under ${LOCAL_DIR}" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"

# verl's merger uses this metadata to write a correct PEFT adapter_config.json.
# RL checkpoints do not always carry it, so write it explicitly.
cat > "${LOCAL_DIR}/lora_train_meta.json" <<EOF
{
  "r": ${LORA_RANK},
  "lora_alpha": ${LORA_ALPHA},
  "task_type": "${LORA_TASK_TYPE}"
}
EOF

cd "${PROJECT_ROOT}"

echo "[merge] starting FSDP merge..."
"${PYTHON_BIN}" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${LOCAL_DIR}" \
  --target_dir "${TARGET_DIR}"

echo "[merge] done."
echo "[merge] output: ${TARGET_DIR}"
if [[ -d "${TARGET_DIR}/lora_adapter" ]]; then
  echo "[merge] lora adapter: ${TARGET_DIR}/lora_adapter"
fi
