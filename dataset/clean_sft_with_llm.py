#!/usr/bin/env python3
"""Clean SFT candidates with an OpenAI LLM judge.

The script reads SFT conversation records, performs cheap local protocol checks,
then asks an OpenAI model such as gpt-4o to judge whether each sample is useful
for SFT. It writes kept records, rejected records, and per-sample decisions.

This script does not modify the input file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("/root/autodl-tmp/datasets/processed/sft/sft_candidates.jsonl")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/datasets/processed/sft_clean")

TAG_PATTERN = re.compile(r"<(think|search|code|answer)>|</(think|search|code|answer)>")
_THREAD_LOCAL = threading.local()


SYSTEM_PROMPT = """You are a strict data-quality judge for supervised fine-tuning data for a coding agent.

The project output protocol is:
- <think> is required.
- <answer> is required.
- <search> is optional and should contain a useful retrieval query when present.
- <code> is optional and must contain only executable code, a unified diff patch, or test code.
- <answer> is user-facing text and should not repeat long code.

Judge the sample as SFT data, not as a solved programming task. Prefer rejecting samples that would teach bad formatting, unsafe protocol use, empty code, unrelated answers, malformed patches, or fabricated code.
Important dataset notes:
- For CommitPackFT patch-generation samples, an empty "Before code" section often means the commit adds a new file. Do not reject a new-file unified diff merely because there is no prior file context.
- For CommitPackFT, judge whether the patch content is plausibly aligned with the commit message and target file. A long test file, license header, or boilerplate can be acceptable when the commit asks to add tests or create a file.
- Reject CommitPackFT samples when the patch is malformed, the target file/change type is clearly unrelated to the commit message, or <code> does not contain an applicable unified diff.
- For LiveCodeBench strategy-only samples without <code>, keep only if <answer> gives a concrete solving strategy, algorithm, complexity, or useful retrieval direction. Generic "query prepared" answers should be rejected.
Return only JSON matching the schema."""


JUDGE_SCHEMA = {
    "name": "sft_quality_judgement",
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


def extract_tag(content: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", content, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def tag_order_valid(content: str) -> bool:
    order = {"think": 0, "search": 1, "code": 2, "answer": 3}
    seen_open_tags: list[str] = []
    for match in TAG_PATTERN.finditer(content):
        tag = match.group(1) or match.group(2)
        is_close = match.group(2) is not None
        if not is_close:
            seen_open_tags.append(tag)
    return seen_open_tags == sorted(seen_open_tags, key=lambda t: order[t])


def local_protocol_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    messages = record.get("messages") or []
    if len(messages) != 2:
        return ["messages_must_have_user_and_assistant"]
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        errors.append("bad_message_roles")

    assistant = messages[1].get("content") or ""
    unknown_tags = re.findall(r"</?([a-zA-Z_][a-zA-Z0-9_]*)>", assistant)
    for tag in unknown_tags:
        if tag not in {"think", "search", "code", "answer"}:
            errors.append(f"unknown_tag:{tag}")

    for tag in ["think", "answer"]:
        body = extract_tag(assistant, tag)
        if body is None:
            errors.append(f"missing_{tag}")
        elif not body.strip():
            errors.append(f"empty_{tag}")

    has_code = bool((record.get("metadata") or {}).get("has_code"))
    has_search = bool((record.get("metadata") or {}).get("has_search"))
    code = extract_tag(assistant, "code")
    search = extract_tag(assistant, "search")
    if has_code and not code:
        errors.append("metadata_has_code_but_code_missing")
    if code is not None and not code.strip():
        errors.append("empty_code")
    if has_search and not search:
        errors.append("metadata_has_search_but_search_missing")
    if search is not None:
        try:
            parsed = json.loads(search)
            if not isinstance(parsed, dict) or not parsed.get("query"):
                errors.append("bad_search_json")
        except json.JSONDecodeError:
            errors.append("bad_search_json")
    if not tag_order_valid(assistant):
        errors.append("bad_tag_order")
    return errors


def judge_payload(record: dict[str, Any], max_prompt_chars: int, max_assistant_chars: int) -> str:
    messages = record.get("messages") or [{}, {}]
    payload = {
        "id": record.get("id"),
        "dataset": record.get("dataset"),
        "task_type": record.get("task_type"),
        "metadata": record.get("metadata"),
        "user": compact_text(messages[0].get("content", ""), max_prompt_chars),
        "assistant": compact_text(messages[1].get("content", ""), max_assistant_chars),
        "judge_instructions": {
            "keep_score_threshold": "Keep only if the sample is coherent, protocol-valid, and useful for SFT.",
            "score_5": "Excellent sample, clear task, correct protocol, useful target.",
            "score_4": "Good sample with minor imperfections.",
            "score_3": "Borderline; usable only if data is scarce.",
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


def call_judge(
    client: Any,
    model: str,
    record: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    user_content = judge_payload(record, args.max_prompt_chars, args.max_assistant_chars)
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
        except Exception as exc:  # Retry API/transient/schema issues.
            last_error = exc
            if attempt >= args.max_retries:
                break
            time.sleep(args.retry_sleep * (2**attempt))
    raise RuntimeError(f"Judge call failed after retries: {last_error!r}")


def judge_one_record(record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    record_id = str(record.get("id"))
    local_errors = local_protocol_errors(record)
    if local_errors:
        decision = {
            "id": record_id,
            "dataset": record.get("dataset"),
            "keep": False,
            "score": 1,
            "issues": local_errors,
            "reason": "Rejected by local protocol validation before LLM judging.",
            "judge": "local",
        }
        return record, decision

    if args.local_only:
        decision = {
            "id": record_id,
            "dataset": record.get("dataset"),
            "keep": True,
            "score": 4,
            "issues": [],
            "reason": "Passed local protocol validation. LLM judging disabled by --local-only.",
            "judge": "local_only",
        }
        return record, decision

    client = get_thread_client(args)
    llm_decision = call_judge(client, args.model, record, args)
    decision = {
        "id": record_id,
        "dataset": record.get("dataset"),
        "keep": bool(llm_decision["keep"]) and int(llm_decision["score"]) >= args.min_score,
        "score": int(llm_decision["score"]),
        "issues": llm_decision.get("issues", []),
        "reason": llm_decision.get("reason", ""),
        "judge": args.model,
    }
    if args.sleep > 0:
        time.sleep(args.sleep)
    return record, decision


def already_processed(decisions_path: Path) -> set[str]:
    if not decisions_path.exists():
        return set()
    done: set[str] = set()
    for row in read_jsonl(decisions_path):
        if row.get("id"):
            done.add(str(row["id"]))
    return done


def clean_sft(args: argparse.Namespace) -> Counter:
    validate_api_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kept_path = args.output_dir / "sft_cleaned_candidates.jsonl"
    rejected_path = args.output_dir / "sft_rejected_by_llm.jsonl"
    decisions_path = args.output_dir / "sft_cleaning_decisions.jsonl"
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
                    dataset = None
                    for original in records:
                        if str(original.get("id")) == record_id:
                            dataset = original.get("dataset")
                            record = original
                            break
                    else:
                        record = {"id": record_id, "dataset": None}
                    decision = {
                        "id": record_id,
                        "dataset": dataset,
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
            "Use --max-records for a small paid smoke test before running all SFT candidates.",
            "Use --resume to append and skip records already present in sft_cleaning_decisions.jsonl.",
        ],
    }
    write_json(args.output_dir / "sft_cleaning_stats.json", stats)
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
    parser.add_argument("--max-prompt-chars", type=int, default=6000)
    parser.add_argument("--max-assistant-chars", type=int, default=9000)
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
        help="Run only local protocol validation without calling the OpenAI API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = clean_sft(args)
    print(f"Processed {counts['seen']} SFT records")
    for key, count in sorted((k, v) for k, v in counts.items() if k != "seen"):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
