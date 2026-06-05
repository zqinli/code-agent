#!/usr/bin/env python3
"""Normalize local code-generation datasets into one raw JSONL schema."""

from __future__ import annotations

import argparse
import difflib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/datasets")
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "processed" / "raw" / "raw_all.jsonl"
DEFAULT_CANDIDATE_DIR = DEFAULT_DATA_ROOT / "processed" / "raw_candidates"

DATASETS = ["mbpp", "bigcodebench", "livecodebench", "commitpackft"]
DEFAULT_CANDIDATE_PLAN = {
    "sft": {"mbpp": 400, "bigcodebench": 500, "livecodebench": 200, "commitpackft": 900},
    "rl": {"mbpp": 250, "bigcodebench": 450, "livecodebench": 700, "commitpackft": 600},
    "test": {"mbpp": 100, "bigcodebench": 100, "livecodebench": 100, "commitpackft": 100},
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Counter:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            counts[row["dataset"]] += 1
            counts["total"] += 1
    return counts


def limited(rows: Iterable[dict[str, Any]], max_rows: int | None) -> Iterable[dict[str, Any]]:
    for idx, row in enumerate(rows):
        if max_rows is not None and idx >= max_rows:
            break
        yield row


def safe_json_loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def make_unified_diff(old_path: str, new_path: str, old_text: str, new_text: str) -> str:
    return "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{old_path or 'old_file'}",
            tofile=f"b/{new_path or 'new_file'}",
        )
    )


def mbpp_rows(root: Path, max_rows: int | None) -> Iterable[dict[str, Any]]:
    path = root / "huggingface" / "mbpp" / "data" / "mbpp.jsonl"
    for row in limited(read_jsonl(path), max_rows):
        source_id = str(row.get("task_id"))
        public_tests = row.get("test_list") or []
        hidden_tests = row.get("challenge_test_list") or []
        yield {
            "id": f"mbpp_{source_id}",
            "dataset": "mbpp",
            "source_id": source_id,
            "task_type": "function_generation",
            "language": "python",
            "title": None,
            "instruction": row.get("text", ""),
            "prompt": (
                "Write a Python function for the following task.\n\n"
                f"{row.get('text', '')}\n\n"
                "Public tests:\n"
                + "\n".join(public_tests)
            ).strip(),
            "context": {"test_setup_code": row.get("test_setup_code", "")},
            "code_target": row.get("code", ""),
            "patch_target": None,
            "tests": {
                "public": public_tests,
                "hidden": hidden_tests,
                "setup": row.get("test_setup_code", ""),
            },
            "execution": {"type": "python_function", "entry_point": None},
            "metadata": {"raw_path": str(path)},
        }


def bigcodebench_rows(root: Path, max_rows: int | None, parquet_name: str) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "BigCodeBench requires pyarrow. Run with the conda env that has it, "
            "for example: conda run -n iworkplace python dataset/normalize_raw_datasets.py"
        ) from exc

    path = root / "modelscope" / "bigcodebench" / "bigcode__bigcodebench" / "data" / parquet_name
    table = pq.read_table(path)
    for row in limited(table.to_pylist(), max_rows):
        source_id = str(row.get("task_id"))
        code_prompt = row.get("code_prompt") or ""
        solution = row.get("canonical_solution") or ""
        code_target = code_prompt + solution
        yield {
            "id": source_id.replace("/", "_").lower(),
            "dataset": "bigcodebench",
            "source_id": source_id,
            "task_type": "function_generation",
            "language": "python",
            "title": source_id,
            "instruction": row.get("instruct_prompt", ""),
            "prompt": row.get("complete_prompt") or row.get("instruct_prompt", ""),
            "context": {
                "code_prompt": code_prompt,
                "doc_struct": safe_json_loads(row.get("doc_struct"), {}),
                "libs": safe_json_loads(row.get("libs"), []),
            },
            "code_target": code_target,
            "patch_target": None,
            "tests": {"public": [row.get("test", "")], "hidden": [], "setup": ""},
            "execution": {"type": "python_unittest", "entry_point": row.get("entry_point")},
            "metadata": {"raw_path": str(path)},
        }


def livecodebench_rows(
    root: Path,
    max_rows: int | None,
    hidden_mode: str = "ref",
) -> Iterable[dict[str, Any]]:
    base = root / "modelscope" / "livecodebench" / "livecodebench__code_generation_lite"
    files = sorted(base.glob("test*.jsonl"))
    emitted = 0
    for path in files:
        for row in read_jsonl(path):
            if max_rows is not None and emitted >= max_rows:
                return
            emitted += 1
            source_id = str(row.get("question_id") or f"{path.stem}_{emitted}")
            public_tests = safe_json_loads(row.get("public_test_cases"), [])
            hidden_tests = row.get("private_test_cases")
            hidden_ref = None
            if hidden_tests and hidden_mode == "ref":
                hidden_ref = {
                    "raw_path": str(path),
                    "source_id": source_id,
                    "field": "private_test_cases",
                }
                hidden_tests = None
            elif hidden_mode == "drop":
                hidden_tests = None
            title = row.get("question_title")
            prompt_parts = [p for p in [title, row.get("question_content"), row.get("starter_code")] if p]
            yield {
                "id": f"livecodebench_{source_id}_{path.stem}",
                "dataset": "livecodebench",
                "source_id": source_id,
                "task_type": "stdin_code_generation",
                "language": "python",
                "title": title,
                "instruction": row.get("question_content", ""),
                "prompt": "\n\n".join(prompt_parts),
                "context": {"starter_code": row.get("starter_code", "")},
                "code_target": None,
                "patch_target": None,
                "tests": {"public": public_tests, "hidden": hidden_tests, "setup": ""},
                "execution": {"type": "stdin", "entry_point": None},
                "metadata": {
                    "raw_path": str(path),
                    "platform": row.get("platform"),
                    "contest_id": row.get("contest_id"),
                    "contest_date": row.get("contest_date"),
                    "difficulty": row.get("difficulty"),
                    "hidden_ref": hidden_ref,
                    "metadata": safe_json_loads(row.get("metadata"), {}),
                },
            }


def commitpackft_rows(root: Path, max_rows: int | None, langs: list[str]) -> Iterable[dict[str, Any]]:
    base = root / "modelscope" / "commitpackft" / "bigcode__commitpackft" / "data"
    emitted = 0
    for lang in langs:
        path = base / lang / "data.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"CommitPackFT language file not found: {path}")
        for row in read_jsonl(path):
            if max_rows is not None and emitted >= max_rows:
                return
            emitted += 1
            commit = str(row.get("commit", ""))
            old_file = row.get("old_file", "")
            new_file = row.get("new_file", "")
            old_contents = row.get("old_contents", "")
            new_contents = row.get("new_contents", "")
            patch = make_unified_diff(old_file, new_file, old_contents, new_contents)
            subject = row.get("subject", "")
            message = row.get("message", "")
            instruction = (message or subject).strip()
            yield {
                "id": f"commitpackft_{lang}_{commit[:12]}_{emitted}",
                "dataset": "commitpackft",
                "source_id": commit,
                "task_type": "patch_generation",
                "language": (row.get("lang") or lang).lower(),
                "title": subject,
                "instruction": instruction,
                "prompt": (
                    "Modify the file according to the commit message.\n\n"
                    f"Commit message:\n{instruction}\n\n"
                    f"File: {old_file}\n\n"
                    f"Before:\n{old_contents}"
                ).strip(),
                "context": {
                    "old_file": old_file,
                    "new_file": new_file,
                    "old_contents": old_contents,
                    "new_contents": new_contents,
                },
                "code_target": new_contents,
                "patch_target": patch,
                "tests": {"public": [], "hidden": [], "setup": ""},
                "execution": {"type": "patch", "entry_point": None},
                "metadata": {
                    "raw_path": str(path),
                    "license": row.get("license"),
                    "repos": row.get("repos"),
                },
            }


def normalized_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    root = args.data_root
    per_dataset_max = args.max_per_dataset
    if "mbpp" in args.datasets:
        yield from mbpp_rows(root, per_dataset_max)
    if "bigcodebench" in args.datasets:
        yield from bigcodebench_rows(root, per_dataset_max, args.bigcodebench_parquet)
    if "livecodebench" in args.datasets:
        yield from livecodebench_rows(root, per_dataset_max, args.livecodebench_hidden_mode)
    if "commitpackft" in args.datasets:
        yield from commitpackft_rows(root, per_dataset_max, args.commitpack_langs)


def rows_for_dataset(args: argparse.Namespace, dataset: str) -> Iterable[dict[str, Any]]:
    if dataset == "mbpp":
        yield from mbpp_rows(args.data_root, args.max_per_dataset)
    elif dataset == "bigcodebench":
        yield from bigcodebench_rows(args.data_root, args.max_per_dataset, args.bigcodebench_parquet)
    elif dataset == "livecodebench":
        yield from livecodebench_rows(
            args.data_root,
            args.max_per_dataset,
            args.livecodebench_hidden_mode,
        )
    elif dataset == "commitpackft":
        yield from commitpackft_rows(args.data_root, args.max_per_dataset, args.commitpack_langs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def sample_without_replacement(
    rows: Iterable[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    reservoir: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if idx < count:
            reservoir.append(row)
            continue
        replacement_idx = rng.randint(0, idx)
        if replacement_idx < count:
            reservoir[replacement_idx] = row
    if len(reservoir) < count:
        raise ValueError(f"Requested {count} rows but only found {len(reservoir)} rows")
    rng.shuffle(reservoir)
    return reservoir


def write_candidate_splits(args: argparse.Namespace) -> Counter:
    rng = random.Random(args.seed)
    output_dir = args.candidate_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_by_purpose: dict[str, list[dict[str, Any]]] = {"sft": [], "rl": [], "test": []}
    total_needed = Counter()

    for purpose_plan in DEFAULT_CANDIDATE_PLAN.values():
        total_needed.update(purpose_plan)

    for dataset in DATASETS:
        rows = sample_without_replacement(rows_for_dataset(args, dataset), total_needed[dataset], rng)
        cursor = 0
        for purpose in ["sft", "rl", "test"]:
            need = DEFAULT_CANDIDATE_PLAN[purpose][dataset]
            chunk = rows[cursor : cursor + need]
            cursor += need
            for row in chunk:
                row = dict(row)
                row["candidate_purpose"] = purpose
                selected_by_purpose[purpose].append(row)

    counts: Counter = Counter()
    all_path = output_dir / "raw_candidates_all.jsonl"
    with all_path.open("w", encoding="utf-8") as all_f:
        for purpose in ["sft", "rl", "test"]:
            purpose_rows = selected_by_purpose[purpose]
            rng.shuffle(purpose_rows)
            purpose_path = output_dir / f"{purpose}_candidates.jsonl"
            with purpose_path.open("w", encoding="utf-8") as f:
                for row in purpose_rows:
                    line = json.dumps(row, ensure_ascii=False, sort_keys=True)
                    f.write(line + "\n")
                    all_f.write(line + "\n")
                    counts[f"{purpose}:{row['dataset']}"] += 1
                    counts[purpose] += 1
                    counts["total"] += 1

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mbpp", "bigcodebench", "livecodebench", "commitpackft"],
        choices=DATASETS,
    )
    parser.add_argument(
        "--bigcodebench-parquet",
        default="v0.1.4-00000-of-00001.parquet",
        help="Parquet file under BigCodeBench data/ to normalize.",
    )
    parser.add_argument(
        "--commitpack-langs",
        nargs="+",
        default=["python"],
        help="CommitPackFT language subdirectories to include. Defaults to python.",
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--livecodebench-hidden-mode",
        choices=["ref", "inline", "drop"],
        default="ref",
        help=(
            "How to store LiveCodeBench private_test_cases. 'ref' keeps a pointer "
            "to the raw file and avoids multi-GB normalized outputs."
        ),
    )
    parser.add_argument(
        "--write-candidates",
        action="store_true",
        help="Write sampled raw candidate files for SFT/RL/test instead of one full raw_all file.",
    )
    parser.add_argument("--candidate-output-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--seed", type=int, default=20260511)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_candidates:
        counts = write_candidate_splits(args)
        print(f"Wrote {counts['total']} candidate rows to {args.candidate_output_dir}")
        for key, count in sorted((k, v) for k, v in counts.items() if k != "total"):
            print(f"  {key}: {count}")
    else:
        counts = write_jsonl(args.output, normalized_rows(args))
        print(f"Wrote {counts['total']} rows to {args.output}")
        for dataset, count in sorted((k, v) for k, v in counts.items() if k != "total"):
            print(f"  {dataset}: {count}")


if __name__ == "__main__":
    main()
