# 代码生成智能体

本项目面向复杂代码理解、生成与修改任务，构建具备代码检索、文件阅读、沙箱执行、Patch 生成与反馈优化能力的智能代码生成系统。项目融合混合检索增强、沙箱验证、SFT 冷启动与 GRPO 强化学习对齐，并通过 vLLM 与 FastAPI 支持推理服务化，形成从数据构建、训练对齐到推理评估的完整闭环。整体思路参考 Search-R1 的检索增强推理范式与 Code-R1 的代码任务强化学习流程。

## 核心亮点

- 面向代码修改任务构建多轮交互轨迹数据，设计代码检索、文件读取、沙箱执行、Patch 生成等工具调用协议，形成从问题解析、上下文检索、文件定位、执行验证到 Patch 生成与反馈修正的完整代码修改流程。
- 构建面向代码生成与修改任务的混合 RAG 检索模块，基于 BM25、BGE-M3 与 Milvus 融合稀疏召回和稠密召回结果，为模型提供相关函数、类定义与跨文件依赖上下文。
- 基于 `verl` 框架对 Qwen2.5-Coder 进行 LoRA SFT 冷启动与 GRPO 强化学习训练，优化模型在代码任务中的工具调用、代码定位与结构化 Patch 生成能力。
- 基于 vLLM 与 FastAPI 搭建代码生成模型推理服务，支持离线测试集批量推理、在线请求调用与生成结果追踪。
- 构建 LLM-as-a-Judge 自动评估流程，从需求匹配、Patch 正确性、可执行性与代码质量等维度评估生成结果。

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
SFT Cold Start -> GRPO Alignment
    |
    v
vLLM / FastAPI Inference
    |
    v
LLM-as-a-Judge Evaluation
```

## 工具调用协议

智能体通过结构化 action 与代码环境交互，覆盖代码搜索、文件读取、沙箱执行、Patch 生成和最终回答等环节，与多轮轨迹数据和 GRPO 训练流程保持一致。

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
