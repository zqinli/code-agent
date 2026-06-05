#!/usr/bin/env python3
"""Replace the old code-agent output protocol text in JSONL/Parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


OLD_PROTOCOL = """Output protocol:
- You must include <think>...</think>.
- You must include <answer>...</answer>.
- Use <search>...</search> only when retrieval is useful.
- Use <code>...</code> only when there is executable code, a patch, or test code for the sandbox.
- The sandbox will execute only the content inside <code>.
- The final user-facing response must be inside <answer>."""


NEW_PROTOCOL = """Output protocol:
- You must include <think>...</think>.
- You must include <answer>...</answer>.
- Use <search>...</search> only when retrieval is useful.
- Use <code>...</code> only as an intermediate sandbox action for code, tests, or patches you want to execute/check.
- The sandbox executes only the content inside <code>; <code> is not the final answer.
- Put the final solution inside <answer>...</answer>.
- For code-generation tasks, <answer> must contain the final executable Python code.
- For patch-generation tasks, <answer> must contain the final unified diff patch.
- Do not put <code>, <search>, <information>, or <observation> tags inside <answer>."""


def update_text(text: Any) -> tuple[Any, int]:
    if not isinstance(text, str):
        return text, 0
    count = text.count(OLD_PROTOCOL)
    if count == 0:
        return text, 0
    return text.replace(OLD_PROTOCOL, NEW_PROTOCOL), count


def update_prompt_value(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        return update_text(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        total = 0
        new_items = []
        for item in value:
            if isinstance(item, dict):
                new_item = dict(item)
                new_content, changed = update_text(new_item.get("content"))
                if changed:
                    new_item["content"] = new_content
                total += changed
                new_items.append(new_item)
            else:
                new_item, changed = update_text(item)
                total += changed
                new_items.append(new_item)
        return new_items, total
    return value, 0


def update_jsonl(path: Path) -> int:
    rows: list[dict[str, Any]] = []
    replacements = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "prompt" in row:
                row["prompt"], changed = update_prompt_value(row["prompt"])
                replacements += changed
            rows.append(row)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return replacements


def update_parquet(path: Path) -> int:
    df = pd.read_parquet(path)
    replacements = 0
    if "prompt" not in df.columns:
        return 0
    new_prompts = []
    for value in df["prompt"].tolist():
        new_value, changed = update_prompt_value(value)
        replacements += changed
        new_prompts.append(new_value)
    df["prompt"] = new_prompts
    df.to_parquet(path, index=False)
    return replacements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = 0
    for path in args.paths:
        if not path.exists():
            print(f"skip missing: {path}")
            continue
        if path.suffix == ".jsonl":
            changed = update_jsonl(path)
        elif path.suffix == ".parquet":
            changed = update_parquet(path)
        else:
            print(f"skip unsupported: {path}")
            continue
        total += changed
        print(f"{path}: {changed} replacements")
    print(f"total replacements: {total}")


if __name__ == "__main__":
    main()
