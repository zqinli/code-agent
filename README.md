# Code-Agent 训练工作区

本仓库是一个基于修改版 `verl` 的代码智能体训练工作区，面向多轮代码定位、工具调用、Patch 生成，以及 Qwen2.5-Coder 的 LoRA SFT 冷启动和 GRPO 训练。整体思路参考 Search-R1 的检索增强推理范式和 Code-R1 的代码任务强化学习训练流程。

## 功能概览

- 构建多轮代码修改轨迹，支持代码检索、文件读取、沙箱执行、Patch 生成和最终回答等工具调用协议。
- 构建面向代码任务的 RAG 检索模块，支持 BM25、BGE-M3 和 Milvus，用于召回相关函数、类定义和跨文件上下文；当前训练默认使用轻量级 BM25 后端。
- 参考 Search-R1 / Code-R1 的训练范式，基于 `verl` 对 Qwen2.5-Coder 进行 LoRA SFT 冷启动和 GRPO 训练，优化代码定位、工具调用和 Patch 生成能力。
- 提供 vLLM 离线推理和 FastAPI 服务工具，支持批量生成、在线请求和结果追踪。
- 提供 LLM-as-a-Judge 评估流程，从需求匹配、正确性、可执行性和代码质量等维度评估生成结果。

## 数据流程

数据流程面向 SWE-Gym 风格的代码修改任务，会对 issue 实例进行规范化处理，在 `base_commit` 上检出目标仓库，检索相关代码上下文，并构造 SFT 与 GRPO 训练样本。

相关处理脚本位于 `dataset/scripts/` 和 `scripts/` 目录。

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

离线评估提供 vLLM 推理和 LLM-as-a-Judge 打分工具：

```bash
verl/code-agent/scripts/infer_testset_vllm.sh
verl/code-agent/scripts/judge_inference_outputs.sh
```
