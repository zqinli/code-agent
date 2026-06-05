#!/usr/bin/env python3
"""Restore RL/Test protocol to the previous think+answer-required version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


NEW_PROTOCOL = """Output protocol:
- You must include <answer>...</answer>.
- Use <search>...</search> only when retrieval is useful.
- Use <code>...</code> only as an intermediate sandbox action for code, tests, or patches you want to execute/check.
- The sandbox executes only the content inside <code>; <code> is not the final answer.
- Put the final solution inside <answer>...</answer>.
- For code-generation tasks, <answer> must contain the final executable Python code.
- For patch-generation tasks, <answer> must contain the final unified diff patch.
- Do not include explanations outside <answer>.
- Do not put <code>, <search>, <information>, or <observation> tags inside <answer>."""


OLD_PROTOCOL = """Output protocol:
- You must include <think>...</think>.
- You must include <answer>...</answer>.
- Use <search>...</search> only when retrieval is useful.
- Use <code>...</code> only as an intermediate sandbox action for code, tests, or patches you want to execute/check.
- The sandbox executes only the content inside <code>; <code> is not the final answer.
- Put the final solution inside <answer>...</answer>.
- For code-generation tasks, <answer> must contain the final executable Python code.
- For patch-generation tasks, <answer> must contain the final unified diff patch.
- Do not put <code>, <search>, <information>, or <observation> tags inside <answer>."""


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def update_prompt_text(text: Any) -> tuple[Any, int]:
    if not isinstance(text, str):
        return text, 0
    if NEW_PROTOCOL in text:
        return text.replace(NEW_PROTOCOL, OLD_PROTOCOL), 1
    return text, 0


def update_prompt(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        return update_prompt_text(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        changed = 0
        items = []
        for item in value:
            if isinstance(item, dict):
                new_item = dict(item)
                new_content, delta = update_prompt_text(new_item.get("content"))
                if delta:
                    new_item["content"] = new_content
                changed += delta
                items.append(new_item)
            else:
                new_item, delta = update_prompt_text(item)
                changed += delta
                items.append(new_item)
        return items, changed
    return value, 0


def update_record(row: dict[str, Any]) -> tuple[dict[str, Any], int]:
    changed = 0
    if "prompt" in row:
        row["prompt"], delta = update_prompt(row["prompt"])
        changed += delta
    if row.get("required_tags") != ["think", "answer"]:
        row["required_tags"] = ["think", "answer"]
        changed += 1
    return row, changed


def update_reward_model(value: Any) -> tuple[Any, int]:
    if not isinstance(value, dict):
        return value, 0
    output = dict(value)
    gt = _loads(output.get("ground_truth"))
    if isinstance(gt, dict):
        gt["required_tags"] = ["think", "answer"]
        output["ground_truth"] = _dumps(gt)
        return output, 1
    return output, 0


def update_extra_info(value: Any) -> tuple[Any, int]:
    if not isinstance(value, dict):
        return value, 0
    output = dict(value)
    output["required_tags_json"] = _dumps(["think", "answer"])
    return output, 1


def update_jsonl(path: Path) -> int:
    rows = []
    changed = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row, delta = update_record(json.loads(line))
            rows.append(row)
            changed += delta
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return changed


def update_parquet(path: Path) -> int:
    df = pd.read_parquet(path)
    changed = 0
    if "prompt" in df.columns:
        prompts = []
        for value in df["prompt"].tolist():
            new_value, delta = update_prompt(value)
            prompts.append(new_value)
            changed += delta
        df["prompt"] = prompts
    if "reward_model" in df.columns:
        reward_models = []
        for value in df["reward_model"].tolist():
            new_value, delta = update_reward_model(value)
            reward_models.append(new_value)
            changed += delta
        df["reward_model"] = reward_models
    if "extra_info" in df.columns:
        extra_infos = []
        for value in df["extra_info"].tolist():
            new_value, delta = update_extra_info(value)
            extra_infos.append(new_value)
            changed += delta
        df["extra_info"] = extra_infos
    df.to_parquet(path, index=False)
    return changed


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
        print(f"{path}: {changed} changes")
    print(f"total changes: {total}")


if __name__ == "__main__":
    main()
