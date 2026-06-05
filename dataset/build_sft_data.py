#!/usr/bin/env python3
"""Build SFT conversation data from sampled raw SFT candidates.

The output teaches the model the project protocol:
- <think> is required.
- <answer> is required.
- <search> is optional and used when retrieval is useful.
- <code> is optional and contains only executable code or an applicable patch.

This script does not filter candidates down to the final training size.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("/root/autodl-tmp/datasets/processed/raw_candidates/sft_candidates.jsonl")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/datasets/processed/sft")


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


def json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def xml_block(tag: str, content: str) -> str:
    return f"<{tag}>\n{content.strip()}\n</{tag}>"


def assistant_content(
    think: str,
    answer: str,
    search: dict[str, Any] | None = None,
    code: str | None = None,
) -> str:
    parts = [xml_block("think", think)]
    if search is not None:
        parts.append(xml_block("search", json_block(search)))
    if code is not None:
        parts.append(xml_block("code", code))
    parts.append(xml_block("answer", answer))
    return "\n".join(parts)


def public_tests_text(row: dict[str, Any], max_chars: int = 4000) -> str:
    tests = (row.get("tests") or {}).get("public") or []
    if not tests:
        return "None"
    return compact_text(json.dumps(tests, ensure_ascii=False, indent=2), max_chars)


def make_record(
    row: dict[str, Any],
    user_content: str,
    assistant: str,
    has_search: bool,
    has_code: bool,
    code_kind: str | None,
) -> dict[str, Any]:
    return {
        "id": f"sft_{row['id']}",
        "dataset": row["dataset"],
        "task_type": row.get("task_type"),
        "messages": [
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "metadata": {
            "source_id": row.get("source_id"),
            "raw_id": row.get("id"),
            "candidate_purpose": row.get("candidate_purpose"),
            "language": row.get("language"),
            "has_search": has_search,
            "has_code": has_code,
            "code_kind": code_kind,
            "execution": row.get("execution") or {},
        },
    }


def build_mbpp(row: dict[str, Any]) -> dict[str, Any]:
    user = f"""
You are a code generation agent. Complete the Python function for the task.

Task:
{compact_text(row.get("instruction"), 3000)}

Public tests:
{public_tests_text(row)}

Return your response using the required protocol.
- You must include <think>...</think>.
- Put executable Python code inside <code>...</code>.
- Put the final executable Python code inside <answer>...</answer>.
- Do not put explanations outside the required tags.
"""
    think = "This is a Python function-generation task. I should implement the required function and make sure the public tests pass."
    answer = row.get("code_target") or ""
    assistant = assistant_content(
        think=think,
        code=row.get("code_target") or "",
        answer=answer,
    )
    return make_record(row, user, assistant, has_search=False, has_code=True, code_kind="python_code")


def build_bigcodebench(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("context") or {}
    libs = context.get("libs") or []
    user = f"""
You are a code generation agent. Implement the requested Python function.

Instruction:
{compact_text(row.get("instruction"), 3500)}

Code prompt:
{compact_text(context.get("code_prompt"), 2500)}

Libraries:
{json_block(libs)}

Public tests:
{public_tests_text(row)}

Return your response using the required protocol.
- You must include <think>...</think>.
- Put executable Python code inside <code>...</code>.
- Put the final executable Python code inside <answer>...</answer>.
- Do not put explanations outside the required tags.
"""
    search = None
    think = "This is a complex Python function-generation task. I should follow the provided signature and produce self-contained executable code."
    if libs:
        search = {
            "query": "Python usage examples for " + ", ".join(str(lib) for lib in libs[:5]),
            "top_k": 5,
        }
        think += " The task mentions libraries, so a retrieval query can help recall API usage patterns."
    answer = row.get("code_target") or ""
    assistant = assistant_content(
        think=think,
        search=search,
        code=row.get("code_target") or "",
        answer=answer,
    )
    return make_record(
        row,
        user,
        assistant,
        has_search=search is not None,
        has_code=True,
        code_kind="python_code",
    )


def build_livecodebench(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    user = f"""
You are a competitive-programming assistant. Analyze the problem and prepare a retrieval-oriented solving strategy.

Title:
{row.get("title") or row.get("source_id")}

Problem:
{compact_text(row.get("instruction"), 6000)}

Public examples:
{public_tests_text(row)}

Return your response using the required protocol.
- You must include <think>...</think>.
- Use <search>...</search> when retrieval is useful.
- Include <code>...</code> only if you have verified executable Python code to provide.
- Put the final solution inside <answer>...</answer>.
- If no verified executable code is available, <answer> may contain a concise solving strategy instead.
- Do not put explanations outside the required tags.
"""
    query_terms = [
        str(row.get("title") or row.get("source_id") or "programming problem"),
        str(metadata.get("platform") or ""),
        str(metadata.get("difficulty") or ""),
    ]
    search = {"query": " ".join(t for t in query_terms if t).strip(), "top_k": 5}
    think = (
        "This is an algorithmic programming problem. Since no verified reference solution is provided in the raw candidate, "
        "I should not fabricate executable code; I should identify the problem requirements and issue a useful retrieval query."
    )
    answer = "A retrieval query and solving strategy have been prepared. No executable code is provided because this candidate has no verified reference solution."
    assistant = assistant_content(think=think, search=search, answer=answer)
    return make_record(row, user, assistant, has_search=True, has_code=False, code_kind=None)


def build_commitpackft(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("context") or {}
    old_file = context.get("old_file") or "unknown"
    user = f"""
You are a code modification agent. Generate an applicable patch for the requested change.

Commit message:
{compact_text(row.get("instruction"), 2000)}

File:
{old_file}

Before code:
{compact_text(context.get("old_contents"), 8000)}

Return your response using the required protocol.
- You must include <think>...</think>.
- Put the executable unified diff patch inside <code>...</code>.
- Put the final unified diff patch inside <answer>...</answer>.
- Do not put explanations outside the required tags.
"""
    search = {
        "query": f"similar code patch for {compact_text(row.get('instruction'), 200)} in {old_file}",
        "top_k": 5,
    }
    think = "This is a commit-driven patch-generation task. I should infer the intended edit from the commit message and produce an applicable unified diff."
    answer = row.get("patch_target") or ""
    assistant = assistant_content(
        think=think,
        search=search,
        code=row.get("patch_target") or "",
        answer=answer,
    )
    return make_record(row, user, assistant, has_search=True, has_code=True, code_kind="unified_diff")


def row_to_sft(row: dict[str, Any]) -> dict[str, Any]:
    dataset = row.get("dataset")
    if dataset == "mbpp":
        return build_mbpp(row)
    if dataset == "bigcodebench":
        return build_bigcodebench(row)
    if dataset == "livecodebench":
        return build_livecodebench(row)
    if dataset == "commitpackft":
        return build_commitpackft(row)
    raise ValueError(f"Unsupported dataset: {dataset}")


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    messages = record.get("messages") or []
    if len(messages) != 2:
        errors.append("messages_length")
        return errors
    assistant = messages[1].get("content", "")
    for tag in ["think", "answer"]:
        if f"<{tag}>" not in assistant or f"</{tag}>" not in assistant:
            errors.append(f"missing_{tag}")
    if "<code>" in assistant and not record["metadata"].get("has_code"):
        errors.append("code_tag_metadata_mismatch")
    if "<search>" in assistant and not record["metadata"].get("has_search"):
        errors.append("search_tag_metadata_mismatch")
    return errors


def build_sft(args: argparse.Namespace) -> Counter:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "sft_candidates.jsonl"
    rejected_path = args.output_dir / "sft_rejected.jsonl"
    counts: Counter = Counter()

    with output_path.open("w", encoding="utf-8") as out, rejected_path.open("w", encoding="utf-8") as rejected:
        for row in read_jsonl(args.input):
            try:
                record = row_to_sft(row)
                errors = validate_record(record)
            except Exception as exc:  # Keep bad rows inspectable instead of hiding them.
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
            counts[f"has_code:{record['metadata']['has_code']}"] += 1
            counts[f"has_search:{record['metadata']['has_search']}"] += 1
            code_kind = record["metadata"].get("code_kind") or "none"
            counts[f"code_kind:{code_kind}"] += 1

    stats = {
        "input": str(args.input),
        "output": str(output_path),
        "rejected": str(rejected_path),
        "counts": dict(sorted(counts.items())),
        "notes": [
            "No final-size filtering is applied here.",
            "LiveCodeBench candidates are converted to strategy/search SFT records without <code> because raw candidates do not include verified reference code.",
            "The <code> tag is used only for executable Python code or unified diffs.",
        ],
    }
    write_json(args.output_dir / "sft_stats.json", stats)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_sft(args)
    print(f"Wrote {counts['total']} SFT candidate rows to {args.output_dir / 'sft_candidates.jsonl'}")
    for key, count in sorted((k, v) for k, v in counts.items() if k != "total"):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
