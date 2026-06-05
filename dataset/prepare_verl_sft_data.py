#!/usr/bin/env python3
"""Convert final SFT JSONL into verl multiturn SFT parquet files."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("/root/autodl-tmp/datasets/processed/final/sft_final.jsonl")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/datasets/processed/verl_sft")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def validate_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"{row.get('id')}: expected exactly 2 messages")
    cleaned: list[dict[str, str]] = []
    expected_roles = ["user", "assistant"]
    for idx, (msg, role) in enumerate(zip(messages, expected_roles)):
        if not isinstance(msg, dict):
            raise ValueError(f"{row.get('id')}: message {idx} is not a dict")
        if msg.get("role") != role:
            raise ValueError(f"{row.get('id')}: message {idx} role should be {role}")
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{row.get('id')}: message {idx} has empty content")
        cleaned.append({"role": role, "content": content})
    assistant = cleaned[1]["content"]
    if "<think>" not in assistant or "<answer>" not in assistant:
        raise ValueError(f"{row.get('id')}: assistant missing required tags")
    return cleaned


def extract_tag(content: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", content, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def normalize_user_prompt(content: str) -> str:
    replacements = {
        "Return your response using the required protocol. Put executable Python code only inside <code>.": (
            "Return your response using the required protocol.\n"
            "- You must include <think>...</think>.\n"
            "- Put executable Python code inside <code>...</code>.\n"
            "- Put the final executable Python code inside <answer>...</answer>.\n"
            "- Do not put explanations outside the required tags."
        ),
        "Return your response using the required protocol. Put the patch only inside <code>.": (
            "Return your response using the required protocol.\n"
            "- You must include <think>...</think>.\n"
            "- Put the executable unified diff patch inside <code>...</code>.\n"
            "- Put the final unified diff patch inside <answer>...</answer>.\n"
            "- Do not put explanations outside the required tags."
        ),
        "Return your response using the required protocol. Do not include <code> unless a verified executable solution is available.": (
            "Return your response using the required protocol.\n"
            "- You must include <think>...</think>.\n"
            "- Use <search>...</search> when retrieval is useful.\n"
            "- Include <code>...</code> only if you have verified executable Python code to provide.\n"
            "- Put the final solution inside <answer>...</answer>.\n"
            "- If no verified executable code is available, <answer> may contain a concise solving strategy instead.\n"
            "- Do not put explanations outside the required tags."
        ),
    }
    for old, new in replacements.items():
        content = content.replace(old, new)
    return content


def make_messages(row: dict[str, Any], assistant_format: str) -> list[dict[str, str]] | None:
    messages = validate_messages(row)
    if assistant_format == "protocol":
        messages[0]["content"] = normalize_user_prompt(messages[0]["content"])
        return messages

    metadata = row.get("metadata") or {}
    assistant = messages[1]["content"]
    code = extract_tag(assistant, "code")
    if metadata.get("code_kind") != "python_code" or not code:
        return None

    if assistant_format == "code_only":
        return [{"role": "user", "content": messages[0]["content"]}, {"role": "assistant", "content": code}]
    if assistant_format == "answer_only":
        content = f"<answer>\n{code}\n</answer>"
        return [{"role": "user", "content": normalize_user_prompt(messages[0]["content"])}, {"role": "assistant", "content": content}]
    if assistant_format == "answer_code":
        think = extract_tag(assistant, "think") or "Generate executable Python code for the requested task."
        search = extract_tag(assistant, "search")
        parts = [f"<think>\n{think}\n</think>"]
        if search:
            parts.append(f"<search>\n{search}\n</search>")
        parts.append(f"<code>\n{code}\n</code>")
        parts.append(f"<answer>\n{code}\n</answer>")
        content = "\n".join(parts)
        return [{"role": "user", "content": normalize_user_prompt(messages[0]["content"])}, {"role": "assistant", "content": content}]
    raise ValueError(f"Unsupported assistant_format: {assistant_format}")


def convert_rows(rows: list[dict[str, Any]], assistant_format: str) -> tuple[list[dict[str, Any]], Counter]:
    output: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for idx, row in enumerate(rows):
        messages = make_messages(row, assistant_format)
        if messages is None:
            skipped["total"] += 1
            skipped[f"dataset:{row.get('dataset')}"] += 1
            skipped[f"code_kind:{(row.get('metadata') or {}).get('code_kind')}"] += 1
            continue
        metadata = row.get("metadata") or {}
        output.append(
            {
                "messages": messages,
                "data_source": row.get("dataset"),
                "extra_info": {
                    "index": idx,
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "task_type": row.get("task_type"),
                    "source_id": metadata.get("source_id"),
                    "raw_id": metadata.get("raw_id"),
                    "has_search": metadata.get("has_search"),
                    "has_code": metadata.get("has_code"),
                    "code_kind": metadata.get("code_kind"),
                },
            }
        )
    return output, skipped


def split_rows(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    val_size = max(1, round(len(shuffled) * val_ratio)) if val_ratio > 0 else 0
    val_rows = shuffled[:val_size]
    train_rows = shuffled[val_size:]
    return train_rows, val_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = Counter(row.get("data_source") for row in rows)
    by_task = Counter((row.get("extra_info") or {}).get("task_type") for row in rows)
    return {
        "total": len(rows),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_task_type": dict(sorted(by_task.items())),
    }


def save_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        dataframe = pd.DataFrame(rows)
    else:
        dataframe = pd.DataFrame(columns=["messages", "data_source", "extra_info"])
    dataframe.to_parquet(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument(
        "--assistant-format",
        choices=["protocol", "code_only", "answer_only", "answer_code"],
        default="protocol",
        help=(
            "protocol keeps original tagged assistant output; code_only keeps only executable Python code "
            "and filters patch/strategy samples; answer_only wraps executable Python in <answer> only; "
            "answer_code duplicates executable Python in <code> and <answer>."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows = read_jsonl(args.input)
    converted, skipped = convert_rows(raw_rows, args.assistant_format)
    if not converted:
        raise RuntimeError("No rows left after conversion. Check --assistant-format and input data.")
    train_rows, val_rows = split_rows(converted, args.val_ratio, args.seed)

    train_path = args.output_dir / "train.parquet"
    val_path = args.output_dir / "val.parquet"
    save_parquet(train_rows, train_path)
    save_parquet(val_rows, val_path)

    stats = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "train_file": str(train_path),
        "val_file": str(val_path),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "assistant_format": args.assistant_format,
        "skipped": dict(sorted(skipped.items())),
        "source_summary": {
            "total": len(raw_rows),
            "by_dataset": dict(sorted(Counter(row.get("dataset") for row in raw_rows).items())),
        },
        "train_summary": summarize(train_rows),
        "val_summary": summarize(val_rows),
        "verl_config_hint": {
            "data.train_files": str(train_path),
            "data.val_files": str(val_path),
            "data.messages_key": "messages",
        },
    }
    write_json(args.output_dir / "sft_data_stats.json", stats)
    print(f"Wrote verl SFT train parquet: {train_path} ({len(train_rows)} rows)")
    print(f"Wrote verl SFT val parquet: {val_path} ({len(val_rows)} rows)")


if __name__ == "__main__":
    main()
