#!/usr/bin/env python3
"""Convert final test JSONL into verl evaluation parquet data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("/root/autodl-tmp/datasets/processed/final/test_final.jsonl")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/datasets/processed/verl_eval")


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


def to_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def ability_for(row: dict[str, Any]) -> str:
    dataset = row.get("dataset")
    task_type = row.get("task_type")
    if dataset == "commitpackft" or task_type == "patch_generation":
        return "code_patch"
    if dataset == "livecodebench" or task_type == "stdin_code_generation":
        return "competitive_programming"
    return "code_generation"


def validate_row(row: dict[str, Any]) -> None:
    required = ["id", "dataset", "task_type", "prompt", "execution", "tests", "evaluation_spec"]
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{row.get('id')}: missing keys {missing}")
    if not isinstance(row["prompt"], str) or not row["prompt"].strip():
        raise ValueError(f"{row.get('id')}: empty prompt")
    if row.get("required_tags") != ["think", "answer"]:
        raise ValueError(f"{row.get('id')}: bad required_tags")


def ground_truth_for(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution": row.get("execution") or {},
        "tests": row.get("tests") or {},
        "evaluation_spec": row.get("evaluation_spec") or {},
        "expected_behavior": row.get("expected_behavior") or {},
        "required_tags": row.get("required_tags") or ["think", "answer"],
        "optional_tags": row.get("optional_tags") or ["search", "code"],
    }


def convert_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        validate_row(row)
        metadata = row.get("metadata") or {}
        ground_truth = ground_truth_for(row)
        output.append(
            {
                "data_source": row.get("dataset"),
                "prompt": [{"role": "user", "content": row["prompt"]}],
                "ability": ability_for(row),
                "reward_model": {
                    "style": "rule",
                    # JSON text keeps the parquet schema stable across all task types.
                    "ground_truth": to_json_text(ground_truth),
                },
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "task_type": row.get("task_type"),
                    "source_id": metadata.get("source_id"),
                    "raw_id": metadata.get("raw_id"),
                    "language": metadata.get("language"),
                    "execution_json": to_json_text(row.get("execution")),
                    "tests_json": to_json_text(row.get("tests")),
                    "evaluation_spec_json": to_json_text(row.get("evaluation_spec")),
                    "expected_behavior_json": to_json_text(row.get("expected_behavior")),
                    "required_tags_json": to_json_text(row.get("required_tags")),
                    "optional_tags_json": to_json_text(row.get("optional_tags")),
                    "cleaning_json": to_json_text(row.get("cleaning")),
                },
            }
        )
    return output


def save_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    dataframe = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    dataframe.to_parquet(path, index=False)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = Counter(row.get("data_source") for row in rows)
    by_ability = Counter(row.get("ability") for row in rows)
    by_exec: Counter[str | None] = Counter()
    by_metric: Counter[str | None] = Counter()
    for row in rows:
        extra = row.get("extra_info") or {}
        execution_type = None
        primary_metric = None
        try:
            execution_type = (json.loads(extra.get("execution_json") or "{}") or {}).get("type")
        except json.JSONDecodeError:
            pass
        try:
            primary_metric = (json.loads(extra.get("evaluation_spec_json") or "{}") or {}).get("primary_metric")
        except json.JSONDecodeError:
            pass
        by_exec[execution_type] += 1
        by_metric[primary_metric] += 1
    return {
        "total": len(rows),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_ability": dict(sorted(by_ability.items())),
        "by_execution_type": dict(sorted(by_exec.items())),
        "by_primary_metric": dict(sorted(by_metric.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="test.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows = read_jsonl(args.input)
    converted = convert_rows(raw_rows)

    output_path = args.output_dir / args.output_name
    save_parquet(converted, output_path)

    stats = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "test_file": str(output_path),
        "source_total": len(raw_rows),
        "test_summary": summarize(converted),
        "verl_config_hint": {
            "data.val_files": str(output_path),
            "data.prompt_key": "prompt",
        },
        "reward_note": "Use the same custom reward function as RL; parse reward_model.ground_truth JSON and execute only <code>.",
    }
    write_json(args.output_dir / "test_data_stats.json", stats)
    print(f"Wrote verl eval parquet: {output_path} ({len(converted)} rows)")


if __name__ == "__main__":
    main()
