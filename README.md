# Code-Agent Training Workspace

This repository contains a SWE-Gym based code-agent training workspace built on top of a modified `verl` tree. It focuses on multi-turn code localization, tool use, patch generation, and weak-reward GRPO training for Qwen2.5-Coder.

## Features

- Multi-turn code-editing data construction from SWE-Gym, with tool-use protocols for code search, file reading, sandbox execution, patch generation, and finalization.
- Code-oriented RAG retrieval module supporting BM25, BGE-M3, and Milvus for retrieving relevant functions, class definitions, and cross-file contexts. BM25 is the default lightweight backend during current training runs.
- LoRA SFT cold-start and GRPO training for Qwen2.5-Coder under the `verl` framework, targeting code localization, tool use, and patch generation.
- vLLM offline inference and FastAPI serving utilities for batch generation, online requests, and output tracing.
- LLM-as-a-Judge evaluation pipeline for assessing generated patches by intent alignment, correctness, executability, and code quality.

## Data Pipeline

The main data source is `SWE-Gym/SWE-Gym`. The preprocessing pipeline normalizes each issue instance, checks out the target repository at `base_commit`, retrieves relevant code context, and builds SFT and weak-GRPO datasets.

Main entry points:

```bash
dataset/scripts/run_swegym_full_pipeline.py
scripts/filter_sft_8192.py
scripts/prepare_verl_grpo_weak_data.py
```

Current local dataset summary:

- Raw valid SWE-Gym instances: `2438`
- SFT 8192-token filtered data: `2721` total, `2585` train, `136` validation
- Weak GRPO data: `2438` total, `2317` train, `121` validation

## Training Pipeline

The primary SFT script is:

```bash
bash scripts/run_verl_sft_qwen25coder3b_2gpu.sh
```

The primary GRPO script is:

```bash
bash scripts/run_verl_grpo_qwen25coder3b_lora_code_agent.sh
```

The active GRPO path uses the `code_search_agent` loop with actions such as:

```xml
<search_code>query</search_code>
<open_file>path</open_file>
<run_sandbox>command</run_sandbox>
<generate_patch>unified diff patch</generate_patch>
<final>status</final>
```

## Evaluation

Training-time feedback currently uses a weak patch reward implemented in:

```bash
verl/code-agent/code_agent/rewards/code_agent_reward.py
```

Offline evaluation utilities are provided for vLLM inference and LLM-as-a-Judge scoring:

```bash
verl/code-agent/scripts/infer_testset_vllm.sh
verl/code-agent/scripts/judge_inference_outputs.sh
```

Local judge credentials should be placed in `verl/code-agent/configs/judge.env`. This file is ignored by Git. Use `verl/code-agent/configs/judge.env.example` as a template.

## Current Limitations

- The current GRPO reward is a weak patch reward based on action format, gold-file hits, unified-diff structure, and patch similarity.
- Full SWE-Gym execution evaluation with `git apply`, `pytest`, `FAIL_TO_PASS`, and `PASS_TO_PASS` is not yet integrated.
- Dense retrieval with BGE-M3 and Milvus is implemented but may be disabled during training to reduce GPU memory usage.
- Some scripts under `scripts/` are experimental or legacy variants; the primary SFT and GRPO scripts are listed above.
