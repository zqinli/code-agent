#!/usr/bin/env python3
"""Build a RAG corpus JSONL from sampled raw SFT/RL candidates.

This script only creates text documents for retrieval. It does not build an
embedding index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT_DIR = Path("/root/autodl-tmp/datasets/processed/raw_candidates")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/datasets/processed/rag")
DEFAULT_INPUT_FILES = ["sft_candidates.jsonl", "rl_candidates.jsonl"]


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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def stable_doc_id(*parts: str) -> str:
    raw = "::".join(parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    safe_parts = [p.replace("/", "_").replace(" ", "_") for p in parts if p]
    return "_".join(safe_parts[:4] + [digest]).lower()


def compact_text(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head].rstrip() + "\n...[truncated]...\n" + text[-tail:].lstrip()


def as_json_block(value: Any, max_chars: int) -> str:
    if value in (None, "", [], {}):
        return ""
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return compact_text(text, max_chars)


def first_non_empty(*values: Any) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def make_doc(
    row: dict[str, Any],
    doc_type: str,
    text: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "source_dataset": row["dataset"],
        "source_id": row.get("source_id"),
        "raw_id": row.get("id"),
        "task_type": row.get("task_type"),
        "language": row.get("language"),
        "candidate_purpose": row.get("candidate_purpose"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "doc_id": stable_doc_id(row["dataset"], row.get("id", ""), doc_type),
        "source_dataset": row["dataset"],
        "source_id": row.get("source_id"),
        "doc_type": doc_type,
        "text": text.strip(),
        "metadata": metadata,
    }


def mbpp_doc(row: dict[str, Any], max_code_chars: int, max_tests_chars: int) -> dict[str, Any]:
    tests = row.get("tests") or {}
    text = f"""
Dataset: MBPP
Document type: Python problem and solution pattern

Task:
{compact_text(row.get("instruction"), 2000)}

Public tests:
{as_json_block(tests.get("public"), max_tests_chars)}

Reference implementation:
{compact_text(row.get("code_target"), max_code_chars)}
"""
    return make_doc(row, "problem_solution_pattern", text)


def bigcodebench_doc(row: dict[str, Any], max_code_chars: int, max_tests_chars: int) -> dict[str, Any]:
    context = row.get("context") or {}
    tests = row.get("tests") or {}
    text = f"""
Dataset: BigCodeBench
Document type: complex function/API usage pattern

Instruction:
{compact_text(row.get("instruction"), 3000)}

Code prompt:
{compact_text(context.get("code_prompt"), 2000)}

Libraries:
{as_json_block(context.get("libs"), 1000)}

Doc structure:
{as_json_block(context.get("doc_struct"), 3000)}

Reference implementation:
{compact_text(row.get("code_target"), max_code_chars)}

Test focus:
{as_json_block(tests.get("public"), max_tests_chars)}
"""
    return make_doc(row, "api_function_pattern", text)


def livecodebench_doc(row: dict[str, Any], max_tests_chars: int) -> dict[str, Any]:
    tests = row.get("tests") or {}
    metadata = row.get("metadata") or {}
    text = f"""
Dataset: LiveCodeBench
Document type: competitive programming problem pattern

Title:
{first_non_empty(row.get("title"), row.get("source_id"))}

Problem:
{compact_text(row.get("instruction"), 6000)}

Starter code:
{compact_text((row.get("context") or {}).get("starter_code"), 2000)}

Public examples/tests:
{as_json_block(tests.get("public"), max_tests_chars)}

Platform metadata:
platform={metadata.get("platform")}
contest_id={metadata.get("contest_id")}
contest_date={metadata.get("contest_date")}
difficulty={metadata.get("difficulty")}
"""
    return make_doc(
        row,
        "algorithm_problem_pattern",
        text,
        extra_metadata={
            "platform": metadata.get("platform"),
            "contest_id": metadata.get("contest_id"),
            "contest_date": metadata.get("contest_date"),
            "difficulty": metadata.get("difficulty"),
        },
    )


def commitpackft_doc(row: dict[str, Any], max_code_chars: int, max_patch_chars: int) -> dict[str, Any]:
    context = row.get("context") or {}
    metadata = row.get("metadata") or {}
    text = f"""
Dataset: CommitPackFT
Document type: commit-driven patch pattern

Commit message:
{compact_text(row.get("instruction"), 2000)}

File:
old_file={context.get("old_file")}
new_file={context.get("new_file")}

Before code:
{compact_text(context.get("old_contents"), max_code_chars)}

After code:
{compact_text(context.get("new_contents"), max_code_chars)}

Patch:
{compact_text(row.get("patch_target"), max_patch_chars)}
"""
    return make_doc(
        row,
        "patch_pattern",
        text,
        extra_metadata={
            "old_file": context.get("old_file"),
            "new_file": context.get("new_file"),
            "license": metadata.get("license"),
        },
    )


def row_to_doc(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    dataset = row.get("dataset")
    if dataset == "mbpp":
        return mbpp_doc(row, args.max_code_chars, args.max_tests_chars)
    if dataset == "bigcodebench":
        return bigcodebench_doc(row, args.max_code_chars, args.max_tests_chars)
    if dataset == "livecodebench":
        return livecodebench_doc(row, args.max_tests_chars)
    if dataset == "commitpackft":
        return commitpackft_doc(row, args.max_code_chars, args.max_patch_chars)
    raise ValueError(f"Unsupported dataset: {dataset}")


def input_paths(input_dir: Path, input_files: list[str]) -> list[Path]:
    paths = [input_dir / name for name in input_files]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input candidate files: " + ", ".join(missing))
    return paths


def build_corpus(args: argparse.Namespace) -> Counter:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.output_dir / "corpus.jsonl"
    counts: Counter = Counter()
    seen_doc_ids: set[str] = set()

    with corpus_path.open("w", encoding="utf-8") as out:
        for path in input_paths(args.input_dir, args.input_files):
            for row in read_jsonl(path):
                if row.get("candidate_purpose") == "test":
                    continue
                doc = row_to_doc(args, row)
                if not doc["text"]:
                    counts["skipped_empty_text"] += 1
                    continue
                if doc["doc_id"] in seen_doc_ids:
                    counts["skipped_duplicate_doc_id"] += 1
                    continue
                seen_doc_ids.add(doc["doc_id"])
                out.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")
                counts["total_docs"] += 1
                counts[f"dataset:{doc['source_dataset']}"] += 1
                counts[f"doc_type:{doc['doc_type']}"] += 1
                purpose = doc["metadata"].get("candidate_purpose") or "unknown"
                counts[f"purpose:{purpose}"] += 1

    stats = {
        "corpus_path": str(corpus_path),
        "input_files": [str(path) for path in input_paths(args.input_dir, args.input_files)],
        "counts": dict(sorted(counts.items())),
        "notes": [
            "This script does not build an embedding/vector index.",
            "Test candidates are not used as RAG corpus input.",
            "LiveCodeBench private tests remain referenced by metadata in raw candidates and are not expanded here.",
        ],
    }
    write_json(args.output_dir / "corpus_stats.json", stats)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-files", nargs="+", default=DEFAULT_INPUT_FILES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-code-chars", type=int, default=6000)
    parser.add_argument("--max-patch-chars", type=int, default=8000)
    parser.add_argument("--max-tests-chars", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_corpus(args)
    print(f"Wrote {counts['total_docs']} RAG docs to {args.output_dir / 'corpus.jsonl'}")
    for key, count in sorted((k, v) for k, v in counts.items() if k != "total_docs"):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
