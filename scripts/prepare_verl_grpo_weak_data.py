#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--val_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    with input_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue

            r = json.loads(line)

            messages = r.get("messages", [])
            for message in messages:
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    message["content"] = message["content"].replace("- <direct_answer>answer</direct_answer>\n", "")
                    message["content"] = message["content"].replace("<direct_answer>", "<final>")
                    message["content"] = message["content"].replace("</direct_answer>", "</final>")
            reward_meta = r.get("reward_meta", {}) or {}

            gold_patch = reward_meta.get("gold_patch", "")
            gold_files = reward_meta.get("gold_files", [])
            weak_reward_items = reward_meta.get("weak_reward_items", [])

            ground_truth = {
                "reference_patch": gold_patch,
                "gold_patch": gold_patch,
                "gold_files": gold_files,
                "reward_items": weak_reward_items,
                "execution": {
                    "type": "patch"
                },
                "tests": {
                    "reference_patch": gold_patch,
                    "gold_files": gold_files,
                },
                "reward_spec": {
                    "valid_action_format": 1.0,
                    "search_or_open_hits_gold_file": 1.0,
                    "patch_can_apply": 1.0,
                    "does_not_modify_tests": 1.0,
                    "changed_python_files_compile": 1.0,
                    "patch_similarity_to_gold": 1.0,
                },
            }

            row = {
                "data_source": "swegym_code_agent_weak_grpo",
                "prompt": messages,
                "ability": "code_agent",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "index": idx,
                    "instance_id": r.get("instance_id", ""),
                    "repo": r.get("repo", ""),
                    "base_commit": r.get("base_commit", ""),
                    "reward_meta": reward_meta,
                    "gold_patch": gold_patch,
                    "gold_files": gold_files,
                    "weak_reward_items": weak_reward_items,
                },
            }

            rows.append(row)

    if not rows:
        raise RuntimeError("No rows loaded.")

    random.Random(args.seed).shuffle(rows)

    val_size = min(args.val_size, max(1, len(rows) // 20))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    stats_path = out_dir / "stats.json"

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    stats = {
        "input_jsonl": str(input_path),
        "out_dir": str(out_dir),
        "total": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "columns": list(pd.DataFrame(train_rows).columns),
        "data_source_counts": dict(Counter(x["data_source"] for x in rows)),
        "reward_items_counts": dict(
            Counter(
                item
                for x in rows
                for item in x["extra_info"].get("weak_reward_items", [])
            )
        ),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }

    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("train:", train_path)
    print("val  :", val_path)
    print("stats:", stats_path)


if __name__ == "__main__":
    main()
