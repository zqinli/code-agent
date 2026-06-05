#!/usr/bin/env python3
"""Build evaluation/test candidate data from sampled raw test candidates.

Test records contain prompts, protocol constraints, execution/test metadata, and
evaluation specs. They do not contain assistant reference outputs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("/root/autodl-tmp/datasets/processed/raw_candidates/test_candidates.jsonl")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/datasets/processed/test")

PROTOCOL_TEXT = """Output protocol:
- You must include <think>...</think>.
- You must include <answer>...</answer>.
- Use <search>...</search> only when retrieval is useful.
- Use <code>...</code> only as an intermediate sandbox action for code, tests, or patches you want to execute/check.
- The sandbox executes only the content inside <code>; <code> is not the final answer.
- Put the final solution inside <answer>...</answer>.
- For code-generation tasks, <answer> must contain the final executable Python code.
- For patch-generation tasks, <answer> must contain the final unified diff patch.
- Do not put <code>, <search>, <information>, or <observation> tags inside <answer>."""


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


def compact_text(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head].rstrip() + "\n...[truncated]...\n" + text[-tail:].lstrip()


def json_text(value: Any, max_chars: int = 4000) -> str:
    if value in (None, "", [], {}):
        return "None"
    return compact_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), max_chars)


def base_record(
    row: dict[str, Any],
    prompt: str,
    execution: dict[str, Any],
    tests: dict[str, Any],
    evaluation_spec: dict[str, Any],
    needs_code: bool,
    search_expected: str,
) -> dict[str, Any]:
    return {
        "id": f"test_{row['id']}",
        "dataset": row["dataset"],
        "task_type": row.get("task_type"),
        "prompt": prompt.strip(),
        "required_tags": ["think", "answer"],
        "optional_tags": ["search", "code"],
        "expected_behavior": {
            "needs_code": needs_code,
            "search_expected": search_expected,
            "execute_code_tag_only": True,
        },
        "execution": execution,
        "tests": tests,
        "evaluation_spec": evaluation_spec,
        "metadata": {
            "source_id": row.get("source_id"),
            "raw_id": row.get("id"),
            "candidate_purpose": row.get("candidate_purpose"),
            "language": row.get("language"),
            "original_execution": row.get("execution") or {},
        },
    }


def mbpp_record(row: dict[str, Any]) -> dict[str, Any]:
    tests = row.get("tests") or {}
    prompt = f"""
You are a Python code generation agent.

Task:
{compact_text(row.get("instruction"), 3000)}

Public tests:
{json_text(tests.get("public"))}

{PROTOCOL_TEXT}
"""
    return base_record(
        row=row,
        prompt=prompt,
        execution={
            "enabled": True,
            "type": "python_function",
            "language": "python",
            "timeout_sec": 5,
            "entry_point": None,
        },
        tests={
            "public": tests.get("public") or [],
            "hidden": tests.get("hidden") or [],
            "setup": tests.get("setup") or "",
        },
        evaluation_spec={
            "primary_metric": "pass_rate",
            "metrics": ["format_valid", "compile_success", "public_pass_rate", "hidden_pass_rate"],
        },
        needs_code=True,
        search_expected="optional",
    )


def bigcodebench_record(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("context") or {}
    tests = row.get("tests") or {}
    prompt = f"""
You are a Python code generation agent. Implement the requested function.

Instruction:
{compact_text(row.get("instruction"), 3500)}

Code prompt:
{compact_text(context.get("code_prompt"), 2500)}

Libraries:
{json_text(context.get("libs"), 1000)}

Public tests:
{json_text(tests.get("public"), 4000)}

{PROTOCOL_TEXT}
"""
    return base_record(
        row=row,
        prompt=prompt,
        execution={
            "enabled": True,
            "type": "python_unittest",
            "language": "python",
            "timeout_sec": 10,
            "entry_point": (row.get("execution") or {}).get("entry_point"),
        },
        tests={
            "public": tests.get("public") or [],
            "hidden": tests.get("hidden") or [],
            "setup": tests.get("setup") or "",
        },
        evaluation_spec={
            "primary_metric": "pass_rate",
            "metrics": ["format_valid", "compile_success", "public_pass_rate", "hidden_pass_rate"],
        },
        needs_code=True,
        search_expected="recommended_if_api_uncertain",
    )


def livecodebench_record(row: dict[str, Any]) -> dict[str, Any]:
    tests = row.get("tests") or {}
    metadata = row.get("metadata") or {}
    prompt = f"""
You are a competitive-programming code generation agent. Solve the problem with executable Python code.

Title:
{row.get("title") or row.get("source_id")}

Problem:
{compact_text(row.get("instruction"), 7000)}

Starter code:
{compact_text((row.get("context") or {}).get("starter_code"), 2000)}

Public examples/tests:
{json_text(tests.get("public"), 4000)}

{PROTOCOL_TEXT}
"""
    return base_record(
        row=row,
        prompt=prompt,
        execution={
            "enabled": True,
            "type": "stdin",
            "language": "python",
            "timeout_sec": 10,
            "entry_point": None,
        },
        tests={
            "public": tests.get("public") or [],
            "hidden": tests.get("hidden"),
            "hidden_ref": metadata.get("hidden_ref"),
            "setup": tests.get("setup") or "",
        },
        evaluation_spec={
            "primary_metric": "pass_rate",
            "metrics": ["format_valid", "compile_success", "public_pass_rate", "hidden_pass_rate"],
        },
        needs_code=True,
        search_expected="optional",
    )


def commitpackft_record(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("context") or {}
    old_file = context.get("old_file") or "unknown"
    prompt = f"""
You are a code modification agent. Generate an applicable unified diff patch for the requested change.

Commit message:
{compact_text(row.get("instruction"), 2000)}

File:
{old_file}

Before code:
{compact_text(context.get("old_contents"), 8000)}

{PROTOCOL_TEXT}
"""
    return base_record(
        row=row,
        prompt=prompt,
        execution={
            "enabled": True,
            "type": "patch",
            "language": row.get("language") or "unknown",
            "timeout_sec": 10,
            "entry_point": None,
            "target_file": old_file,
        },
        tests={
            "public": [],
            "hidden": [],
            "setup": "",
            "reference_patch": row.get("patch_target"),
            "reference_after": context.get("new_contents"),
        },
        evaluation_spec={
            "primary_metric": "patch_score",
            "metrics": ["format_valid", "patch_apply", "syntax_or_static_check", "diff_similarity", "intent_match"],
        },
        needs_code=True,
        search_expected="recommended",
    )


def row_to_test(row: dict[str, Any]) -> dict[str, Any]:
    dataset = row.get("dataset")
    if dataset == "mbpp":
        return mbpp_record(row)
    if dataset == "bigcodebench":
        return bigcodebench_record(row)
    if dataset == "livecodebench":
        return livecodebench_record(row)
    if dataset == "commitpackft":
        return commitpackft_record(row)
    raise ValueError(f"Unsupported dataset: {dataset}")


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if "messages" in record:
        errors.append("test_record_must_not_include_messages")
    if not record.get("prompt"):
        errors.append("missing_prompt")
    if record.get("required_tags") != ["think", "answer"]:
        errors.append("bad_required_tags")
    if sorted(record.get("optional_tags") or []) != ["code", "search"]:
        errors.append("bad_optional_tags")
    for key in ["execution", "tests", "evaluation_spec"]:
        if key not in record:
            errors.append(f"missing_{key}")
    if (record.get("metadata") or {}).get("candidate_purpose") != "test":
        errors.append("candidate_purpose_not_test")
    return errors


def build_test(args: argparse.Namespace) -> Counter:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "test_candidates.jsonl"
    rejected_path = args.output_dir / "test_rejected.jsonl"
    counts: Counter = Counter()

    with output_path.open("w", encoding="utf-8") as out, rejected_path.open("w", encoding="utf-8") as rejected:
        for row in read_jsonl(args.input):
            try:
                record = row_to_test(row)
                errors = validate_record(record)
            except Exception as exc:
                rejected.write(
                    json.dumps(
                        {"raw_id": row.get("id"), "dataset": row.get("dataset"), "error": repr(exc)},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                counts["rejected"] += 1
                continue

            if errors:
                rejected.write(
                    json.dumps(
                        {
                            "raw_id": row.get("id"),
                            "dataset": row.get("dataset"),
                            "errors": errors,
                            "record": record,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                counts["rejected"] += 1
                continue

            out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            counts["total"] += 1
            counts[f"dataset:{record['dataset']}"] += 1
            counts[f"execution:{record['execution']['type']}"] += 1
            counts[f"metric:{record['evaluation_spec']['primary_metric']}"] += 1
            counts[f"search_expected:{record['expected_behavior']['search_expected']}"] += 1

    stats = {
        "input": str(args.input),
        "output": str(output_path),
        "rejected": str(rejected_path),
        "counts": dict(sorted(counts.items())),
        "notes": [
            "Test records do not include assistant reference outputs.",
            "Required/optional tags are protocol constraints only.",
            "Only model output inside <code> should be executed by the sandbox.",
        ],
    }
    write_json(args.output_dir / "test_stats.json", stats)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_test(args)
    print(f"Wrote {counts['total']} test candidate rows to {args.output_dir / 'test_candidates.jsonl'}")
    for key, count in sorted((k, v) for k, v in counts.items() if k != "total"):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
