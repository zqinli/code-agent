#!/usr/bin/env python3
"""Clean RL candidates with an OpenAI-compatible LLM judge.

RL records should contain prompts, protocol constraints, execution/test metadata,
and reward specs. They must not contain assistant reference outputs.
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


DEFAULT_INPUT = Path("/root/autodl-tmp/datasets/processed/rl/rl_candidates.jsonl")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/datasets/processed/rl_clean")
_THREAD_LOCAL = threading.local()


SYSTEM_PROMPT = """You are a strict data-quality judge for reinforcement-learning data for a coding agent.

The RL record is not supposed to contain a reference assistant answer. It should contain:
- a clear task prompt
- required_tags and optional_tags protocol constraints
- execution metadata for sandbox evaluation
- tests or references needed to compute reward
- a reward_spec whose components match the task

Judge whether the record is useful for RL training. Prefer rejecting records with missing/unclear prompts, missing executable evaluation information, inconsistent execution type, useless tests, malformed reward specs, or accidental leakage of a reference assistant response.
Primary judging principle:
- Keep records that are clear enough for rollout and have enough execution/tests/references to compute reward.
- Do not reject merely because the prompt is long, verbose, or includes full public tests. Coding RL prompts often include task instructions, signatures, and tests.
- Do not reject merely because reward_spec is generic, as long as its components are compatible with the task and sum to 1.
- Reject for verbosity only when it makes the task ambiguous, includes irrelevant/unrelated content, or leaks the reference solution.
- Reject reward_spec only when it is mathematically invalid, mismatched to execution type, missing the main reward signal, or impossible to compute from the record.

Dataset notes:
- MBPP should be a Python function-generation task with public and/or hidden assert tests.
- BigCodeBench should be a Python unittest/function-generation task with a code prompt and tests. Long prompts with docstrings, imports, code prompts, and full unittest text are normal and should usually be kept if executable.
- LiveCodeBench should be a stdin competitive-programming task with public examples/tests; hidden tests may be referenced by hidden_ref instead of embedded.
- CommitPackFT should be a patch-generation task. It can use reference_patch/reference_after for reward computation, but the prompt should not expose the target patch or after-code.
- For CommitPackFT, empty before-code may mean adding a new file; do not reject for that alone.

Return only JSON matching the schema."""


JUDGE_SCHEMA = {
    "name": "rl_quality_judgement",
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
        errors.append("rl_record_must_not_include_messages")
    if not record.get("prompt"):
        errors.append("missing_prompt")
    if record.get("required_tags") != ["think", "answer"]:
        errors.append("bad_required_tags")
    if sorted(record.get("optional_tags") or []) != ["code", "search"]:
        errors.append("bad_optional_tags")

    execution = record.get("execution") or {}
    tests = record.get("tests") or {}
    reward_spec = record.get("reward_spec") or {}
    if not execution:
        errors.append("missing_execution")
    if not reward_spec:
        errors.append("missing_reward_spec")
    else:
        try:
            total_reward = sum(float(v) for v in reward_spec.values())
            if abs(total_reward - 1.0) > 1e-6:
                errors.append(f"reward_sum_{total_reward:.4f}")
        except (TypeError, ValueError):
            errors.append("bad_reward_values")

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
        if not tests.get("reference_patch") and not tests.get("reference_after"):
            errors.append("missing_patch_reference")
        prompt = record.get("prompt") or ""
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
        "reward_spec": record.get("reward_spec"),
        "metadata": record.get("metadata"),
        "judge_instructions": {
            "keep_score_threshold": "Keep only if the record is coherent and rewardable.",
            "score_5": "Excellent RL sample: clear prompt, correct execution config, useful tests/references, aligned reward spec.",
            "score_4": "Good RL sample with minor imperfections.",
            "score_3": "Borderline; reject unless data is scarce.",
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
            "reason": "Rejected by local RL validation before LLM judging.",
            "judge": "local",
        }

    if args.local_only:
        return record, {
            "id": record_id,
            "dataset": record.get("dataset"),
            "keep": True,
            "score": 4,
            "issues": [],
            "reason": "Passed local RL validation. LLM judging disabled by --local-only.",
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


def clean_rl(args: argparse.Namespace) -> Counter:
    validate_api_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kept_path = args.output_dir / "rl_cleaned_candidates.jsonl"
    rejected_path = args.output_dir / "rl_rejected_by_llm.jsonl"
    decisions_path = args.output_dir / "rl_cleaning_decisions.jsonl"
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
            "This is a quality-cleaning step, not final size selection.",
            "RL records do not include assistant reference outputs.",
            "Use --max-records for a small paid smoke test before running all RL candidates.",
            "Use --resume to append and skip records already present in rl_cleaning_decisions.jsonl.",
        ],
    }
    write_json(args.output_dir / "rl_cleaning_stats.json", stats)
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
        help="Run only local RL validation without calling the OpenAI API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = clean_rl(args)
    print(f"Processed {counts['seen']} RL records")
    for key, count in sorted((k, v) for k, v in counts.items() if k != "seen"):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
