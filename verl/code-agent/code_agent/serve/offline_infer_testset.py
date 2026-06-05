"""Run offline vLLM inference on a verl parquet test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline vLLM inference for code-agent test sets.")
    parser.add_argument("--test-file", required=True, help="Input parquet file.")
    parser.add_argument("--output-file", required=True, help="Output jsonl file.")
    parser.add_argument("--model-path", required=True, help="Base HuggingFace model path.")
    parser.add_argument("--lora-adapter-path", default=None, help="Optional PEFT LoRA adapter path.")
    parser.add_argument("--served-model-name", default="code-agent-rl", help="Name used for the LoRA request.")
    parser.add_argument("--limit", type=int, default=-1, help="Only run the first N examples.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of prompts per vLLM generate call.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def to_messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    messages = []
    for item in value:
        if hasattr(item, "item"):
            item = item.item()
        messages.append({"role": item["role"], "content": item["content"]})
    return messages


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def build_prompts(tokenizer: AutoTokenizer, rows: pd.DataFrame) -> list[str]:
    prompts = []
    for _, row in rows.iterrows():
        messages = to_messages(row["prompt"])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts


def write_records(
    output_path: Path,
    rows: pd.DataFrame,
    outputs: list[Any],
    start_index: int,
) -> None:
    with output_path.open("a", encoding="utf-8") as f:
        for offset, (_, row) in enumerate(rows.iterrows()):
            output = outputs[offset]
            completion = output.outputs[0]
            record = {
                "index": start_index + offset,
                "id": to_jsonable(row.get("extra_info", {})).get("id"),
                "data_source": row.get("data_source"),
                "ability": row.get("ability"),
                "prompt": to_messages(row["prompt"]),
                "response": completion.text,
                "finish_reason": completion.finish_reason,
                "prompt_token_ids_len": len(output.prompt_token_ids or []),
                "output_token_ids_len": len(completion.token_ids or []),
                "extra_info": to_jsonable(row.get("extra_info", {})),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    df = pd.read_parquet(args.test_file)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=bool(args.lora_adapter_path),
        max_lora_rank=32,
        enforce_eager=args.enforce_eager,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    lora_request = None
    if args.lora_adapter_path:
        lora_request = LoRARequest(args.served_model_name, 1, args.lora_adapter_path)

    total = len(df)
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch = df.iloc[start:end]
        prompts = build_prompts(tokenizer, batch)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        write_records(output_path, batch, outputs, start)
        print(f"[infer] wrote {end}/{total} -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
