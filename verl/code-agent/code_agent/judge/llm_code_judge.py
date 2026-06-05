"""Evaluate generated code with an OpenAI-compatible LLM judge API."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests


DIMENSIONS = [
    "correctness",
    "intent_alignment",
    "completeness",
    "executability",
    "code_quality",
]


SYSTEM_PROMPT = """You are a strict but fair senior software engineer judging generated code.

Evaluate the candidate response against the task. Score each dimension from 0 to 5:
0 = completely failed or not assessable
1 = very poor
2 = poor
3 = partially acceptable
4 = good
5 = excellent

Dimensions:
- correctness: whether the generated code/function/patch is functionally correct.
- intent_alignment: whether it matches the user's real intent and requested change.
- completeness: whether it covers the full request and important edge cases.
- executability: whether it can run or be applied without syntax/import/patch errors.
- code_quality: whether it is clear, simple, maintainable, and consistent.

Return only valid JSON. Do not include markdown fences or extra text.
"""


USER_PROMPT_TEMPLATE = """Judge this generated code answer.

Task / prompt:
{prompt}

Candidate response:
{response}

Reference / ground truth / tests, if available:
{reference}

Return this exact JSON schema:
{{
  "scores": {{
    "correctness": <integer 0-5>,
    "intent_alignment": <integer 0-5>,
    "completeness": <integer 0-5>,
    "executability": <integer 0-5>,
    "code_quality": <integer 0-5>
  }},
  "overall_score": <number 0-5>,
  "pass": <true or false>,
  "rationale": {{
    "correctness": "<short reason>",
    "intent_alignment": "<short reason>",
    "completeness": "<short reason>",
    "executability": "<short reason>",
    "code_quality": "<short reason>",
    "summary": "<short overall reason>"
  }},
  "major_issues": ["<issue 1>", "<issue 2>"]
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-a-judge for generated code outputs.")
    parser.add_argument("--input-file", required=True, help="Input jsonl file with prompt and response fields.")
    parser.add_argument("--output-file", required=True, help="Output jsonl file with judge results.")
    parser.add_argument("--env-file", default=None, help="Optional .env file.")
    parser.add_argument("--api-url", default=None, help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default=None, help="API key. Prefer .env or env var for real keys.")
    parser.add_argument("--judge-model", default=None, help="Judge model name.")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--reference-field", default=None)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    parser.add_argument("--pass-threshold", type=float, default=3.5)
    return parser.parse_args()


def load_env_file(path: str | None) -> None:
    if not path:
        return
    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"env file not found: {env_path}")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_nested(record: dict[str, Any], path: str | None, default: Any = "") -> Any:
    if not path:
        return default
    current: Any = record
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_judgement(raw: dict[str, Any], pass_threshold: float) -> dict[str, Any]:
    scores = raw.get("scores", {})
    normalized_scores = {}
    for dim in DIMENSIONS:
        value = scores.get(dim, 0)
        try:
            value = int(round(float(value)))
        except (TypeError, ValueError):
            value = 0
        normalized_scores[dim] = max(0, min(5, value))

    overall = raw.get("overall_score")
    if overall is None:
        overall = sum(normalized_scores.values()) / len(normalized_scores)
    try:
        overall = float(overall)
    except (TypeError, ValueError):
        overall = sum(normalized_scores.values()) / len(normalized_scores)
    overall = max(0.0, min(5.0, overall))

    return {
        "scores": normalized_scores,
        "overall_score": overall,
        "pass": bool(raw.get("pass", overall >= pass_threshold)),
        "rationale": raw.get("rationale", {}),
        "major_issues": raw.get("major_issues", []),
    }


def build_payload(args: argparse.Namespace, model: str, record: dict[str, Any]) -> dict[str, Any]:
    prompt = stringify(get_nested(record, args.prompt_field))
    response = stringify(get_nested(record, args.response_field))
    reference = stringify(get_nested(record, args.reference_field)) if args.reference_field else ""

    user_prompt = USER_PROMPT_TEMPLATE.format(prompt=prompt, response=response, reference=reference or "N/A")
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }


def call_judge_api(
    api_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    url = api_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return extract_json(content)
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"judge API failed after {retries + 1} attempts: {last_error}") from last_error


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)

    api_url = args.api_url or os.getenv("JUDGE_API_URL") or os.getenv("OPENAI_BASE_URL")
    api_key = args.api_key or os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    model = args.judge_model or os.getenv("JUDGE_MODEL")
    if not api_url:
        raise ValueError("Missing API URL. Set --api-url or JUDGE_API_URL/OPENAI_BASE_URL.")
    if not model:
        raise ValueError("Missing judge model. Set --judge-model or JUDGE_MODEL.")

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if args.limit > 0 and count >= args.limit:
                break
            if not line.strip():
                continue

            record = json.loads(line)
            payload = build_payload(args, model, record)
            result: dict[str, Any]
            try:
                raw = call_judge_api(api_url, api_key, payload, args.timeout, args.retries)
                result = normalize_judgement(raw, args.pass_threshold)
                result["judge_error"] = None
            except Exception as exc:
                result = {
                    "scores": {dim: 0 for dim in DIMENSIONS},
                    "overall_score": 0.0,
                    "pass": False,
                    "rationale": {},
                    "major_issues": [],
                    "judge_error": str(exc),
                }

            output_record = {
                "line_no": line_no,
                "id": get_nested(record, args.id_field, default=None),
                "judge_model": model,
                "judge_result": result,
                "source": record,
            }
            dst.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            dst.flush()
            count += 1
            print(f"[judge] wrote {count}: line {line_no}", flush=True)
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"[judge] done: {count} records -> {output_path}")


if __name__ == "__main__":
    main()
