#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=0,1
unset LD_LIBRARY_PATH

export PYTHONPATH="${PROJECT_ROOT}/verl:${PROJECT_ROOT}/verl/code-agent:${PYTHONPATH:-}"

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 避免 code-agent dense retriever 抢 GPU
export CODE_AGENT_DISABLE_DENSE_RAG=1
export CODE_AGENT_BGE_DEVICE=cpu
export CODE_AGENT_BGE_USE_FP16=0
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/dataset/data/swegym_full_project}"
export CODE_AGENT_RAG_CORPUS="${CODE_AGENT_RAG_CORPUS:-${DATA_ROOT}/processed/retrieval_bm25_topk.jsonl}"
export CODE_AGENT_RAG_INDEX_DIR="${CODE_AGENT_RAG_INDEX_DIR:-${DATA_ROOT}/processed/rag_index}"
export CODE_AGENT_RAG_MAX_CONTEXT_DOCS=20000
export CODE_AGENT_USE_ACTION_STOPS=1
export CODE_AGENT_MAX_OBS_LENGTH=192
export CODE_AGENT_ACTION_MAX_NEW_TOKENS=96
export CODE_AGENT_FINAL_MAX_NEW_TOKENS=256

# Repository checkout root used by tool actions such as <open_file> and <run_sandbox>.
export CODE_AGENT_WORKSPACE_PATH="${CODE_AGENT_WORKSPACE_PATH:-${DATA_ROOT}/repos}"

MODEL="${MODEL:-${PROJECT_ROOT}/models/Qwen2.5-Coder-3B-Instruct-sdpa}"
SFT_ADAPTER="${SFT_ADAPTER:-${PROJECT_ROOT}/outputs/qwen25_coder_3b_swegym_sft_lora_exported/global_step_1292/lora_adapter}"

GRPO_DATA_DIR="${GRPO_DATA_DIR:-${DATA_ROOT}/verl_grpo}"
TRAIN="${TRAIN:-${GRPO_DATA_DIR}/train.parquet}"
VAL="${VAL:-${GRPO_DATA_DIR}/val.parquet}"

OUT="${OUT:-${PROJECT_ROOT}/outputs/qwen25_coder_3b_swegym_code_agent_grpo_lora}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN" \
  data.val_files="$VAL" \
  data.return_raw_chat=True \
  data.train_batch_size=1 \
  data.max_prompt_length=8192 \
  data.max_response_length=640 \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.lora_adapter_path="$SFT_ADAPTER" \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8832 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.calculate_entropy=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_model_len=8832 \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.agent.default_agent_loop=code_search_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/verl/code-agent/configs/agent_loop_config.yaml" \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  reward.custom_reward_function.path="${PROJECT_ROOT}/verl/code-agent/code_agent/rewards/code_agent_reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.default_local_dir="$OUT" \
  trainer.project_name=swegym-code-agent-grpo \
  trainer.experiment_name=qwen25-coder-3b-lora-grpo \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=400 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.resume_mode=auto \
  trainer.val_before_train=False \
  trainer.rollout_update_weights_freq=8 \
  trainer.total_epochs=1
