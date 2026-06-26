#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def calc_total_tokens(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_total_tokens", type=int, default=8192)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.model_max_length = 10**9

    kept = []
    dropped = []

    with input_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue

            row = json.loads(line)
            messages = row.get("messages", [])

            if len(messages) < 2 or messages[-1].get("role") != "assistant":
                dropped.append({
                    "index": idx,
                    "instance_id": row.get("instance_id", ""),
                    "reason": "bad_messages",
                })
                continue

            total_tokens = calc_total_tokens(tokenizer, messages)

            item = {
                "data_source": "swegym_sft",
                "instance_id": row.get("instance_id", ""),
                "repo": row.get("repo", ""),
                "base_commit": row.get("base_commit", ""),
                "stage": row.get("stage", ""),
                "messages": messages,
                "extra_info": {
                    "index": idx,
                    "instance_id": row.get("instance_id", ""),
                    "repo": row.get("repo", ""),
                    "base_commit": row.get("base_commit", ""),
                    "stage": row.get("stage", ""),
                    "total_tokens": total_tokens,
                },
            }

            if total_tokens > args.max_total_tokens:
                dropped.append({
                    "index": idx,
                    "instance_id": row.get("instance_id", ""),
                    "repo": row.get("repo", ""),
                    "stage": row.get("stage", ""),
                    "reason": "total_too_long",
                    "total_tokens": total_tokens,
                })
            else:
                kept.append(item)

    random.Random(args.seed).shuffle(kept)

    val_size = min(args.val_size, max(1, len(kept) // 20))
    val_rows = kept[:val_size]
    train_rows = kept[val_size:]

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    dropped_path = out_dir / "dropped.jsonl"
    stats_path = out_dir / "stats.json"

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    with dropped_path.open("w", encoding="utf-8") as f:
        for x in dropped:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    stats = {
        "input_jsonl": str(input_path),
        "out_dir": str(out_dir),
        "max_total_tokens": args.max_total_tokens,
        "total_input": len(kept) + len(dropped),
        "kept_total": len(kept),
        "dropped_total": len(dropped),
        "train": len(train_rows),
        "val": len(val_rows),
        "kept_stage_counts": dict(Counter(x["stage"] for x in kept)),
        "dropped_reason_counts": dict(Counter(x["reason"] for x in dropped)),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "dropped_path": str(dropped_path),
    }

    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
