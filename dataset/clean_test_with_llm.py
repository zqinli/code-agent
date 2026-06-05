#!/usr/bin/env python3
"""Clean test/evaluation candidates with an OpenAI-compatible LLM judge.

Test records should contain prompts, protocol constraints, execution/test
metadata, and evaluation specs. They must not contain assistant reference
outputs, and prompts must not leak target code/patches.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("/root/autodl-tmp/datasets/processed/test/test_candidates.jsonl")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/datasets/processed/test_clean")
_THREAD_LOCAL = threading.local()


SYSTEM_PROMPT = """You are a strict data-quality judge for held-out evaluation data for a coding agent.

The test record is not supposed to contain a reference assistant answer. It should contain:
- a clear held-out task prompt
- required_tags and optional_tags protocol constraints
- execution metadata for sandbox evaluation
- tests or references needed to compute evaluation metrics
- an evaluation_spec compatible with the task

Judge whether the record is useful as held-out test/evaluation data. Keep records that are clear, evaluable, and not leaked. Reject records with missing/unclear prompts, missing executable evaluation information, inconsistent execution type, useless tests, impossible metrics, or accidental leakage of a reference answer/code/patch in the prompt.

Primary judging principle:
- Keep records that can fairly evaluate model rollout quality.
- Do not reject merely because the prompt is long or includes full public tests.
- Reject if the prompt contains the target solution, target patch, reference_after code, or otherwise makes the answer trivial.
- Reject if evaluation cannot be computed from tests/references in the record.

Dataset notes:
- MBPP should be a Python function-generation task with assert tests.
- BigCodeBench should be a Python unittest/function-generation task with a code prompt and tests. Long prompts with docstrings, imports, and full unittest text are normal.
- LiveCodeBench should be a stdin competitive-programming task with public examples/tests; hidden tests may be referenced by hidden_ref instead of embedded.
- CommitPackFT should be a patch-generation task. It can store reference_patch/reference_after for scoring, but the prompt must not expose them.
- For CommitPackFT, empty before-code may mean adding a new file; do not reject for that alone.

Return only JSON matching the schema."""


JUDGE_SCHEMA = {
    "name": "test_quality_judgement",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "keep": {"type": "boolean"},
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "reason": {"type": "string"},
        },
        "required": ["keep", "score", "issues", "reason"],
    },
    "strict": True,
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            row["_line_no"] = line_no
            yield row


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def compact_text(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head].rstrip() + "\n...[truncated]...\n" + text[-tail:].lstrip()


def local_protocol_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if "messages" in record:
        errors.append("test_record_must_not_include_messages")
    if not record.get("prompt"):
        errors.append("missing_prompt")
    if record.get("required_tags") != ["think", "answer"]:
        errors.append("bad_required_tags")
    if sorted(record.get("optional_tags") or []) != ["code", "search"]:
        errors.append("bad_optional_tags")

    metadata = record.get("metadata") or {}
    if metadata.get("candidate_purpose") != "test":
        errors.append("candidate_purpose_not_test")

    execution = record.get("execution") or {}
    tests = record.get("tests") or {}
    evaluation_spec = record.get("evaluation_spec") or {}
    if not execution:
        errors.append("missing_execution")
    if not evaluation_spec:
        errors.append("missing_evaluation_spec")

    dataset = record.get("dataset")
    execution_type = execution.get("type")
    if dataset == "mbpp" and execution_type != "python_function":
        errors.append("mbpp_bad_execution_type")
    elif dataset == "bigcodebench" and execution_type != "python_unittest":
        errors.append("bigcodebench_bad_execution_type")
    elif dataset == "livecodebench" and execution_type != "stdin":
        errors.append("livecodebench_bad_execution_type")
    elif dataset == "commitpackft" and execution_type != "patch":
        errors.append("commitpackft_bad_execution_type")

    if dataset in {"mbpp", "bigcodebench", "livecodebench"}:
        if not tests.get("public") and not tests.get("hidden") and not tests.get("hidden_ref"):
            errors.append("missing_code_tests")
    if dataset == "commitpackft":
        prompt = record.get("prompt") or ""
        if not tests.get("reference_patch") and not tests.get("reference_after"):
            errors.append("missing_patch_reference")
        if tests.get("reference_patch") and str(tests["reference_patch"]) in prompt:
            errors.append("reference_patch_leaked_in_prompt")
        if tests.get("reference_after") and str(tests["reference_after"]) in prompt:
            errors.append("reference_after_leaked_in_prompt")

    return errors


def judge_payload(record: dict[str, Any], max_prompt_chars: int, max_tests_chars: int) -> str:
    payload = {
        "id": record.get("id"),
        "dataset": record.get("dataset"),
        "task_type": record.get("task_type"),
        "prompt": compact_text(record.get("prompt"), max_prompt_chars),
        "required_tags": record.get("required_tags"),
        "optional_tags": record.get("optional_tags"),
        "expected_behavior": record.get("expected_behavior"),
        "execution": record.get("execution"),
        "tests_excerpt": compact_text(json.dumps(record.get("tests"), ensure_ascii=False), max_tests_chars),
        "evaluation_spec": record.get("evaluation_spec"),
        "metadata": record.get("metadata"),
        "judge_instructions": {
            "keep_score_threshold": "Keep only if the record is clear, held-out, and evaluable.",
            "score_5": "Excellent test sample: clear prompt, no leakage, correct execution config, useful tests/references.",
            "score_4": "Good test sample with minor imperfections.",
            "score_3": "Borderline; reject unless evaluation data is scarce.",
            "score_1_or_2": "Reject.",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def get_client(args: argparse.Namespace) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing openai package. Install it in the environment before running this script.") from exc
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Set environment variable {args.api_key_env}.")
    if args.base_url:
        return OpenAI(api_key=api_key, base_url=args.base_url)
    return OpenAI(api_key=api_key)


def validate_api_config(args: argparse.Namespace) -> None:
    if args.local_only:
        return
    if not os.environ.get(args.api_key_env):
        raise RuntimeError(
            f"Missing API key. Set environment variable {args.api_key_env}, "
            f"or pass --api-key-env with the name of the variable you use."
        )


def get_thread_client(args: argparse.Namespace) -> Any:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = get_client(args)
        _THREAD_LOCAL.client = client
    return client


def call_judge(client: Any, model: str, record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    user_content = judge_payload(record, args.max_prompt_chars, args.max_tests_chars)
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_schema", "json_schema": JUDGE_SCHEMA},
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Empty judge response")
            return json.loads(content)
        except Exception as exc:
            last_error = exc
            if attempt >= args.max_retries:
                break
            time.sleep(args.retry_sleep * (2**attempt))
    raise RuntimeError(f"Judge call failed after retries: {last_error!r}")


def judge_one_record(record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    record_id = str(record.get("id"))
    local_errors = local_protocol_errors(record)
    if local_errors:
        return record, {
            "id": record_id,
            "dataset": record.get("dataset"),
            "keep": False,
            "score": 1,
            "issues": local_errors,
            "reason": "Rejected by local test validation before LLM judging.",
            "judge": "local",
        }

    if args.local_only:
        return record, {
            "id": record_id,
            "dataset": record.get("dataset"),
            "keep": True,
            "score": 4,
            "issues": [],
            "reason": "Passed local test validation. LLM judging disabled by --local-only.",
            "judge": "local_only",
        }

    client = get_thread_client(args)
    llm_decision = call_judge(client, args.model, record, args)
    if args.sleep > 0:
        time.sleep(args.sleep)
    return record, {
        "id": record_id,
        "dataset": record.get("dataset"),
        "keep": bool(llm_decision["keep"]) and int(llm_decision["score"]) >= args.min_score,
        "score": int(llm_decision["score"]),
        "issues": llm_decision.get("issues", []),
        "reason": llm_decision.get("reason", ""),
        "judge": args.model,
    }


def already_processed(decisions_path: Path) -> set[str]:
    if not decisions_path.exists():
        return set()
    done: set[str] = set()
    for row in read_jsonl(decisions_path):
        if row.get("id"):
            done.add(str(row["id"]))
    return done


def clean_test(args: argparse.Namespace) -> Counter:
    validate_api_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kept_path = args.output_dir / "test_cleaned_candidates.jsonl"
    rejected_path = args.output_dir / "test_rejected_by_llm.jsonl"
    decisions_path = args.output_dir / "test_cleaning_decisions.jsonl"
    counts: Counter = Counter()

    processed = already_processed(decisions_path) if args.resume else set()
    records: list[dict[str, Any]] = []
    for record in read_jsonl(args.input):
        if args.max_records is not None and len(records) >= args.max_records:
            break
        record_id = str(record.get("id"))
        if record_id in processed:
            counts["skipped_resume"] += 1
            continue
        records.append(record)
    counts["seen"] = len(records)

    record_by_id = {str(record.get("id")): record for record in records}
    mode = "a" if args.resume else "w"
    with (
        kept_path.open(mode, encoding="utf-8") as kept_f,
        rejected_path.open(mode, encoding="utf-8") as rejected_f,
        decisions_path.open(mode, encoding="utf-8") as decisions_f,
    ):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_id = {
                executor.submit(judge_one_record, record, args): str(record.get("id"))
                for record in records
            }
            for future in concurrent.futures.as_completed(future_to_id):
                try:
                    record, decision = future.result()
                except Exception as exc:
                    record_id = future_to_id[future]
                    record = record_by_id.get(record_id, {"id": record_id, "dataset": None})
                    decision = {
                        "id": record_id,
                        "dataset": record.get("dataset"),
                        "keep": False,
                        "score": 1,
                        "issues": ["judge_exception"],
                        "reason": repr(exc),
                        "judge": args.model if not args.local_only else "local_only",
                    }

                decisions_f.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")
                if decision["keep"]:
                    clean_record = dict(record)
                    clean_record.pop("_line_no", None)
                    clean_record["cleaning"] = {
                        "judge": decision["judge"],
                        "score": decision["score"],
                        "reason": decision["reason"],
                    }
                    kept_f.write(json.dumps(clean_record, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["kept"] += 1
                    counts[f"kept:{record.get('dataset')}"] += 1
                else:
                    rejected_f.write(
                        json.dumps(
                            {
                                "decision": decision,
                                "record": {k: v for k, v in record.items() if k != "_line_no"},
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    counts["rejected"] += 1
                    counts[f"rejected:{record.get('dataset')}"] += 1

    stats = {
        "input": str(args.input),
        "kept": str(kept_path),
        "rejected": str(rejected_path),
        "decisions": str(decisions_path),
        "counts": dict(sorted(counts.items())),
        "model": args.model,
        "min_score": args.min_score,
        "workers": args.workers,
        "notes": [
            "This is a test/evaluation quality-cleaning step, not final size selection.",
            "Test records do not include assistant reference outputs.",
            "Use --max-records for a small paid smoke test before running all test candidates.",
            "Use --resume to append and skip records already present in test_cleaning_decisions.jsonl.",
        ],
    }
    write_json(args.output_dir / "test_cleaning_stats.json", stats)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="Optional OpenAI-compatible API base URL. Defaults to OPENAI_BASE_URL if set.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the API key.",
    )
    parser.add_argument("--min-score", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-prompt-chars", type=int, default=8000)
    parser.add_argument("--max-tests-chars", type=int, default=5000)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 8),
        help="Concurrent LLM judge workers. Defaults to min(os.cpu_count(), 8).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Run only local test validation without calling the OpenAI API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = clean_test(args)
    print(f"Processed {counts['seen']} test records")
    for key, count in sorted((k, v) for k, v in counts.items() if k != "seen"):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
