#!/usr/bin/env python3
"""Select final SFT/RL/Test datasets from cleaned candidates."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SFT_INPUT = Path("/root/autodl-tmp/datasets/processed/sft_clean_v5/sft_cleaned_candidates.jsonl")
DEFAULT_RL_INPUT = Path("/root/autodl-tmp/datasets/processed/rl_clean_v2/rl_cleaned_candidates.jsonl")
DEFAULT_TEST_INPUT = Path("/root/autodl-tmp/datasets/processed/test_clean_v1/test_cleaned_candidates.jsonl")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/datasets/processed/final")

DEFAULT_PLAN = {
    "sft": {"mbpp": 60, "bigcodebench": 90, "livecodebench": 10, "commitpackft": 140},
    "rl": {"mbpp": 60, "bigcodebench": 130, "livecodebench": 180, "commitpackft": 130},
    "test": {"mbpp": 55, "bigcodebench": 55, "livecodebench": 70, "commitpackft": 20},
}


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def group_by_dataset(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("dataset"))].append(row)
    return grouped


def select_by_plan(
    rows: list[dict[str, Any]],
    plan: dict[str, int],
    rng: random.Random,
    name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = group_by_dataset(rows)
    selected: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"requested": plan, "available": {}, "selected": {}}

    for dataset, requested in plan.items():
        candidates = list(grouped.get(dataset, []))
        stats["available"][dataset] = len(candidates)
        if len(candidates) < requested:
            raise ValueError(
                f"{name}: requested {requested} rows for {dataset}, "
                f"but only {len(candidates)} cleaned rows are available"
            )
        rng.shuffle(candidates)
        chosen = candidates[:requested]
        selected.extend(chosen)
        stats["selected"][dataset] = len(chosen)

    rng.shuffle(selected)
    stats["total_selected"] = len(selected)
    return selected, stats


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = Counter(str(row.get("dataset")) for row in rows)
    by_task = Counter(str(row.get("task_type")) for row in rows)
    return {
        "total": len(rows),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_task_type": dict(sorted(by_task.items())),
    }


def select_all(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "sft": args.sft_input,
        "rl": args.rl_input,
        "test": args.test_input,
    }
    output_names = {
        "sft": "sft_final.jsonl",
        "rl": "rl_final.jsonl",
        "test": "test_final.jsonl",
    }

    stats: dict[str, Any] = {
        "seed": args.seed,
        "plan": DEFAULT_PLAN,
        "inputs": {name: str(path) for name, path in sources.items()},
        "outputs": {},
        "splits": {},
    }

    for name, path in sources.items():
        rows = read_jsonl(path)
        selected, split_stats = select_by_plan(rows, DEFAULT_PLAN[name], rng, name)
        output_path = args.output_dir / output_names[name]
        write_jsonl(output_path, selected)
        stats["outputs"][name] = str(output_path)
        stats["splits"][name] = {
            **split_stats,
            "source_summary": summarize(rows),
            "final_summary": summarize(selected),
        }

    write_json(args.output_dir / "final_stats.json", stats)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-input", type=Path, default=DEFAULT_SFT_INPUT)
    parser.add_argument("--rl-input", type=Path, default=DEFAULT_RL_INPUT)
    parser.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = select_all(args)
    print(f"Wrote final data to {args.output_dir}")
    for name, split_stats in stats["splits"].items():
        summary = split_stats["final_summary"]
        print(f"  {name}: {summary['total']} {summary['by_dataset']}")


if __name__ == "__main__":
    main()
