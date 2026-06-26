# Code-Agent: 检索增强的代码修改智能体训练框架

本项目面向复杂代码理解、代码定位与自动 Patch 生成任务，构建了一套基于 Qwen2.5-Coder 的代码智能体训练与评估框架。系统围绕多轮工具调用、代码检索、沙箱反馈、SFT 冷启动、GRPO 强化学习和 LLM-as-a-Judge 评估展开，整体思路参考 Search-R1 的检索增强推理范式与 Code-R1 的代码任务强化学习流程。

## 核心亮点

- 设计多轮代码修改智能体流程，将问题解析、上下文检索、文件定位、命令执行、Patch 生成和反馈修正组织为统一的工具调用轨迹。
- 构建面向代码仓库的混合 RAG 检索模块，融合 BM25、BGE-M3 和 Milvus，用于召回相关函数、类定义、调用关系和跨文件依赖上下文。
- 基于 `verl` 训练 Qwen2.5-Coder，完成 LoRA SFT 冷启动和 GRPO 训练，使模型学习代码定位、工具使用和结构化 Patch 生成。
- 集成 vLLM 离线推理与 FastAPI 服务接口，支持批量测试、在线调用和生成结果追踪。
- 搭建 LLM-as-a-Judge 自动评估流程，从需求匹配、Patch 正确性、可执行性和代码质量等维度衡量生成结果。

## 系统流程

```text
Issue / Task
    |
    v
Repository Checkout + Context Preprocessing
    |
    v
Hybrid Code Retrieval
    |
    v
Multi-turn Tool-use Trajectory
    |
    v
SFT Cold Start -> GRPO Training
    |
    v
vLLM / FastAPI Inference
    |
    v
LLM-as-a-Judge Evaluation
```

## 工具调用协议

智能体通过结构化 action 与代码环境交互，覆盖代码搜索、文件读取、沙箱执行、Patch 生成和最终回答等环节。

```xml
<search_code>query</search_code>
<open_file>path</open_file>
<run_sandbox>command</run_sandbox>
<generate_patch>unified diff patch</generate_patch>
<final>status</final>
```

## 主要模块

```text
dataset/scripts/                 数据预处理与轨迹构造
scripts/                         SFT / GRPO 训练入口脚本
verl/code-agent/code_agent/       代码智能体核心逻辑
verl/code-agent/code_agent/tools/ 工具调用与检索组件
verl/code-agent/code_agent/judge/ LLM-as-a-Judge 评估模块
verl/code-agent/scripts/          推理与评估脚本
```

## 训练入口

SFT 冷启动：

```bash
bash scripts/run_verl_sft_qwen25coder3b_2gpu.sh
```

GRPO 训练：

```bash
bash scripts/run_verl_grpo_qwen25coder3b_lora_code_agent.sh
```

## 推理与评估

离线推理：

```bash
bash verl/code-agent/scripts/infer_testset_vllm.sh
```

自动评估：

```bash
bash verl/code-agent/scripts/judge_inference_outputs.sh
```

## 技术栈

- Backbone: Qwen2.5-Coder
- Training: `verl`, LoRA SFT, GRPO
- Retrieval: BM25, BGE-M3, Milvus
- Inference: vLLM, FastAPI
- Evaluation: LLM-as-a-Judge

## 参考

- Search-R1: Retrieval-augmented reasoning for language models
- Code-R1: Reinforcement learning for code reasoning and generation
