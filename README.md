# Code-Agent 训练工作区

本仓库是一个基于修改版 `verl` 的代码智能体训练工作区，面向多轮代码定位、工具调用、Patch 生成，以及 Qwen2.5-Coder 的 LoRA SFT 冷启动和弱奖励 GRPO 训练。

## 功能概览

- 构建多轮代码修改轨迹，支持代码检索、文件读取、沙箱执行、Patch 生成和最终回答等工具调用协议。
- 构建面向代码任务的 RAG 检索模块，支持 BM25、BGE-M3 和 Milvus，用于召回相关函数、类定义和跨文件上下文；当前训练默认使用轻量级 BM25 后端。
- 基于 `verl` 对 Qwen2.5-Coder 进行 LoRA SFT 冷启动和 GRPO 训练，优化代码定位、工具调用和 Patch 生成能力。
- 提供 vLLM 离线推理和 FastAPI 服务工具，支持批量生成、在线请求和结果追踪。
- 提供 LLM-as-a-Judge 评估流程，从需求匹配、正确性、可执行性和代码质量等维度评估生成结果。

## 数据流程

数据流程面向 SWE-Gym 风格的代码修改任务，会对 issue 实例进行规范化处理，在 `base_commit` 上检出目标仓库，检索相关代码上下文，并构造 SFT 和弱奖励 GRPO 训练样本。

主要入口：

```bash
dataset/scripts/run_swegym_full_pipeline.py
scripts/filter_sft_8192.py
scripts/prepare_verl_grpo_weak_data.py
```

## 训练流程

主要 SFT 脚本：

```bash
bash scripts/run_verl_sft_qwen25coder3b_2gpu.sh
```

主要 GRPO 脚本：

```bash
bash scripts/run_verl_grpo_qwen25coder3b_lora_code_agent.sh
```

当前 GRPO 路径使用 `code_search_agent` 交互循环，支持如下动作：

```xml
<search_code>query</search_code>
<open_file>path</open_file>
<run_sandbox>command</run_sandbox>
<generate_patch>unified diff patch</generate_patch>
<final>status</final>
```

## 评估

训练阶段目前使用弱 Patch 奖励，实现在：

```bash
verl/code-agent/code_agent/rewards/code_agent_reward.py
```

离线评估提供 vLLM 推理和 LLM-as-a-Judge 打分工具：

```bash
verl/code-agent/scripts/infer_testset_vllm.sh
verl/code-agent/scripts/judge_inference_outputs.sh
```

本地 judge 配置可以参考 `verl/code-agent/configs/judge.env.example`。

## 当前限制

- 当前 GRPO 奖励仍是弱 Patch 奖励，主要基于动作格式、gold file 命中、unified diff 结构和 Patch 相似度。
- 尚未集成完整的 SWE-Gym 执行评估，例如 `git apply`、`pytest`、`FAIL_TO_PASS` 和 `PASS_TO_PASS`。
- BGE-M3 和 Milvus 稠密检索已实现，但训练时可能会为了降低显存占用而关闭。
- `scripts/` 下仍有部分实验性或历史脚本，主要 SFT 和 GRPO 脚本以上方列出的为准。
