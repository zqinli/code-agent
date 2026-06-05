#!/usr/bin/env python3
"""Build final RAG corpus from final SFT and final RL data.

Test data is intentionally excluded. SFT examples may contribute supervised
code/patch demonstrations. RL examples contribute task and public-evaluation
patterns without putting hidden tests or reference patches into retrievable text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SFT_INPUT = Path("/root/autodl-tmp/datasets/processed/final/sft_final.jsonl")
DEFAULT_RL_INPUT = Path("/root/autodl-tmp/datasets/processed/final/rl_final.jsonl")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/datasets/processed/rag_final")


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
    safe = [p.replace("/", "_").replace(" ", "_") for p in parts if p]
    return "_".join(safe[:4] + [digest]).lower()


def compact_text(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head].rstrip() + "\n...[truncated]...\n" + text[-tail:].lstrip()


def json_excerpt(value: Any, max_chars: int) -> str:
    if value in (None, "", [], {}):
        return ""
    return compact_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), max_chars)


def extract_tag(content: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", content, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def source_metadata(row: dict[str, Any], source_split: str) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return {
        "source_split": source_split,
        "source_dataset": row.get("dataset"),
        "source_id": metadata.get("source_id"),
        "raw_id": metadata.get("raw_id") or row.get("id"),
        "task_type": row.get("task_type"),
        "language": metadata.get("language"),
    }


def make_doc(
    source_split: str,
    row: dict[str, Any],
    doc_type: str,
    text: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = source_metadata(row, source_split)
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "doc_id": stable_doc_id(source_split, row.get("id", ""), doc_type),
        "source_dataset": row.get("dataset"),
        "source_id": metadata.get("source_id"),
        "doc_type": doc_type,
        "text": text.strip(),
        "metadata": metadata,
    }


def sft_doc_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if row.get("dataset") == "commitpackft" or metadata.get("code_kind") == "unified_diff":
        return "sft_patch_demonstration"
    if row.get("dataset") == "livecodebench":
        return "sft_strategy_demonstration"
    return "sft_code_demonstration"


def sft_to_doc(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    messages = row.get("messages") or []
    user = messages[0].get("content", "") if len(messages) > 0 else ""
    assistant = messages[1].get("content", "") if len(messages) > 1 else ""
    think = extract_tag(assistant, "think")
    search = extract_tag(assistant, "search")
    code = extract_tag(assistant, "code")
    answer = extract_tag(assistant, "answer")
    doc_type = sft_doc_type(row)

    text = f"""
Source: final SFT
Dataset: {row.get("dataset")}
Document type: {doc_type}

User task:
{compact_text(user, args.max_prompt_chars)}

Plan:
{compact_text(think, args.max_think_chars)}

Search action:
{compact_text(search, args.max_search_chars)}

Executable code or patch:
{compact_text(code, args.max_code_chars)}

Final answer:
{compact_text(answer, args.max_answer_chars)}
"""
    metadata = row.get("metadata") or {}
    return make_doc(
        "sft_final",
        row,
        doc_type,
        text,
        {
            "has_search": bool(search),
            "has_code": bool(code),
            "code_kind": metadata.get("code_kind"),
        },
    )


def rl_doc_type(row: dict[str, Any]) -> str:
    dataset = row.get("dataset")
    if dataset == "commitpackft":
        return "rl_patch_task_pattern"
    if dataset == "livecodebench":
        return "rl_algorithm_task_pattern"
    if dataset == "bigcodebench":
        return "rl_function_unittest_pattern"
    return "rl_function_assert_pattern"


def public_tests_view(tests: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {
        "public": tests.get("public") or [],
        "setup": tests.get("setup") or "",
    }
    if tests.get("hidden_ref"):
        view["hidden_ref_available"] = True
    if tests.get("reference_patch") or tests.get("reference_after"):
        view["reference_available_for_reward"] = True
    return view


def rl_to_doc(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    tests = row.get("tests") or {}
    execution = row.get("execution") or {}
    reward_spec = row.get("reward_spec") or {}
    doc_type = rl_doc_type(row)

    text = f"""
Source: final RL
Dataset: {row.get("dataset")}
Document type: {doc_type}

RL prompt:
{compact_text(row.get("prompt"), args.max_prompt_chars)}

Execution configuration:
{json_excerpt(execution, args.max_metadata_chars)}

Reward spec:
{json_excerpt(reward_spec, args.max_metadata_chars)}

Public evaluation view:
{json_excerpt(public_tests_view(tests), args.max_tests_chars)}
"""
    return make_doc(
        "rl_final",
        row,
        doc_type,
        text,
        {
            "execution_type": execution.get("type"),
            "search_expected": (row.get("expected_behavior") or {}).get("search_expected"),
            "reference_policy": "hidden/reference fields are not included in retrievable text",
        },
    )


def build_corpus(args: argparse.Namespace) -> Counter:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.output_dir / "corpus.jsonl"
    counts: Counter = Counter()
    seen_doc_ids: set[str] = set()

    with corpus_path.open("w", encoding="utf-8") as out:
        for row in read_jsonl(args.sft_input):
            doc = sft_to_doc(row, args)
            if doc["doc_id"] in seen_doc_ids:
                counts["skipped_duplicate_doc_id"] += 1
                continue
            seen_doc_ids.add(doc["doc_id"])
            out.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")
            counts["total_docs"] += 1
            counts["source:sft_final"] += 1
            counts[f"dataset:{doc['source_dataset']}"] += 1
            counts[f"doc_type:{doc['doc_type']}"] += 1

        for row in read_jsonl(args.rl_input):
            doc = rl_to_doc(row, args)
            if doc["doc_id"] in seen_doc_ids:
                counts["skipped_duplicate_doc_id"] += 1
                continue
            seen_doc_ids.add(doc["doc_id"])
            out.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")
            counts["total_docs"] += 1
            counts["source:rl_final"] += 1
            counts[f"dataset:{doc['source_dataset']}"] += 1
            counts[f"doc_type:{doc['doc_type']}"] += 1

    write_json(
        args.output_dir / "corpus_stats.json",
        {
            "corpus_path": str(corpus_path),
            "inputs": {"sft": str(args.sft_input), "rl": str(args.rl_input)},
            "counts": dict(sorted(counts.items())),
            "notes": [
                "Test data is excluded from the final RAG corpus.",
                "SFT docs include supervised code/patch demonstrations.",
                "RL docs include prompts and public evaluation views only.",
                "RL hidden tests, reference patches, and reference_after code are not included in retrievable text.",
                "This script does not build a vector index.",
            ],
        },
    )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-input", type=Path, default=DEFAULT_SFT_INPUT)
    parser.add_argument("--rl-input", type=Path, default=DEFAULT_RL_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-prompt-chars", type=int, default=6000)
    parser.add_argument("--max-think-chars", type=int, default=1000)
    parser.add_argument("--max-search-chars", type=int, default=1000)
    parser.add_argument("--max-code-chars", type=int, default=8000)
    parser.add_argument("--max-answer-chars", type=int, default=1000)
    parser.add_argument("--max-tests-chars", type=int, default=3000)
    parser.add_argument("--max-metadata-chars", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_corpus(args)
    print(f"Wrote {counts['total_docs']} final RAG docs to {args.output_dir / 'corpus.jsonl'}")
    for key, count in sorted((k, v) for k, v in counts.items() if k != "total_docs"):
        print(f"  {key}: {count}")


def _line_after(label: str, text: str) -> str:
    marker = label + ":"
    if marker not in text:
        return ""
    rest = text.split(marker, 1)[1].strip()
    return rest.splitlines()[0].strip() if rest else ""


def _section_after(label: str, text: str, max_chars: int) -> str:
    marker = label + ":"
    if marker not in text:
        return ""
    rest = text.split(marker, 1)[1].strip()
    # Stop at the next common section label when possible.
    stops = [
        "\n\nPublic tests:",
        "\n\nPublic examples",
        "\n\nCode prompt:",
        "\n\nLibraries:",
        "\n\nFile:",
        "\n\nBefore code:",
        "\n\nExecution configuration:",
        "\n\nReward spec:",
        "\n\nReturn your response",
        "\n\nOutput protocol:",
    ]
    end = len(rest)
    for stop in stops:
        pos = rest.find(stop)
        if pos != -1:
            end = min(end, pos)
    return compact_text(rest[:end], max_chars)


def _tests_from_record(row: dict[str, Any]) -> Any:
    if "tests" in row:
        return (row.get("tests") or {}).get("public") or []
    messages = row.get("messages") or []
    user = messages[0].get("content", "") if messages else ""
    match = re.search(r"Public (?:tests|examples).*?:\n(.*?)(?:\n\nReturn your response|\Z)", user, re.DOTALL)
    return match.group(1).strip() if match else ""


def _diff_summary(diff_text: str, max_lines: int = 12) -> dict[str, Any]:
    added: list[str] = []
    removed: list[str] = []
    files_added = False
    files_deleted = False
    for line in diff_text.splitlines():
        if line.startswith("--- /dev/null") or line.startswith("--- a/") and "\n+++ /dev/null" in diff_text:
            files_added = True
        if line.startswith("+++ /dev/null"):
            files_deleted = True
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+") and line[1:].strip():
            added.append(line[1:].rstrip())
        elif line.startswith("-") and line[1:].strip():
            removed.append(line[1:].rstrip())
    return {
        "change_type": _infer_patch_change_type(diff_text, added, removed, files_added, files_deleted),
        "added_lines": added[:max_lines],
        "removed_lines": removed[:max_lines],
        "added_count": len(added),
        "removed_count": len(removed),
    }


def _infer_patch_change_type(
    diff_text: str,
    added: list[str],
    removed: list[str],
    files_added: bool,
    files_deleted: bool,
) -> str:
    lower = diff_text.lower()
    if files_added or "@@ -0,0 " in lower:
        return "add_new_file"
    if files_deleted:
        return "delete_file"
    if added and removed and len(added) <= 5 and len(removed) <= 5:
        return "small_replacement_or_reorder"
    if "test" in lower or "assert" in lower or "unittest" in lower:
        return "test_change"
    if "import " in lower:
        return "dependency_or_import_change"
    return "code_modification"


def _keyword_summary(*texts: str, max_keywords: int = 18) -> str:
    stop = {
        "the", "and", "for", "with", "from", "this", "that", "into", "when", "then",
        "code", "file", "patch", "change", "fix", "add", "update", "use", "using",
        "your", "return", "response", "inside", "protocol", "generate", "applicable",
    }
    words: list[str] = []
    for text in texts:
        words.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text))
    seen: set[str] = set()
    kept: list[str] = []
    for word in words:
        key = word.lower()
        if key in stop or key in seen:
            continue
        seen.add(key)
        kept.append(word)
        if len(kept) >= max_keywords:
            break
    return ", ".join(kept)


def _patch_edit_hint(intent: str, file_path: str, diff_text: str = "") -> str:
    lower = f"{intent} {file_path}".lower()
    dependency_match = re.search(r"\badd\s+([A-Za-z0-9_.-]+)\s+to\s+(?:the\s+)?(?:requirements|dependencies|install_requires)", lower)
    if dependency_match or ("requirements" in lower and ("setup.py" in lower or "requirements" in file_path.lower())):
        dependency = dependency_match.group(1) if dependency_match else "the requested dependency"
        return (
            f"When the intent is to add {dependency} to requirements and the affected file is {file_path}, "
            "update the dependency list such as install_requires or requirements entries with the requested package, "
            "using a minimal unified diff."
        )
    if "test" in lower and ("add" in lower or "create" in lower):
        return (
            "When the intent asks to add tests, create or update the relevant test file with focused test cases for "
            "the named behavior or utility, avoiding unrelated production-code changes."
        )
    if "import" in lower or "dependency" in lower:
        return (
            "When the intent concerns imports or dependencies, keep the patch narrow: add, remove, or reorder the "
            "specific import/dependency needed by the affected file."
        )
    if "order" in lower or "reorder" in lower:
        return (
            "When the intent mentions order, prefer a minimal line reordering patch and preserve surrounding behavior."
        )
    if "rename" in lower:
        return (
            "When the intent is a rename, update the named symbol/path consistently while avoiding unrelated edits."
        )
    if "fix" in lower:
        return (
            "When the intent is a fix, identify the smallest code change that addresses the described failure mode "
            "and keep the diff scoped to the affected file."
        )
    return (
        "Infer the concrete edit from the commit intent and affected file, then generate the smallest applicable "
        "unified diff that satisfies the requested change."
    )


def _knowledge_doc(
    source_split: str,
    row: dict[str, Any],
    doc_type: str,
    title: str,
    body: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    title = " ".join(str(title).split())
    title = compact_text(title, 160)
    text = f"""
Title: {title}
Source split: {source_split}
Dataset: {row.get("dataset")}
Knowledge type: {doc_type}

{body.strip()}
"""
    return make_doc(source_split, row, doc_type, text, extra_metadata)


def _sft_knowledge_docs(row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    messages = row.get("messages") or []
    user = messages[0].get("content", "") if len(messages) > 0 else ""
    assistant = messages[1].get("content", "") if len(messages) > 1 else ""
    dataset = row.get("dataset")
    code = extract_tag(assistant, "code")
    search = extract_tag(assistant, "search")
    answer = extract_tag(assistant, "answer")
    docs: list[dict[str, Any]] = []

    if dataset == "commitpackft":
        commit_message = _section_after("Commit message", user, 1200)
        file_path = _line_after("File", user)
        summary = _diff_summary(code)
        added_excerpt = "\n".join(summary["added_lines"]) or "No added lines captured."
        removed_excerpt = "\n".join(summary["removed_lines"]) or "No removed lines captured."
        keywords = _keyword_summary(commit_message, file_path, added_excerpt, removed_excerpt)
        edit_hint = _patch_edit_hint(commit_message, file_path, code)
        body = f"""
Commit intent:
{commit_message}

Affected file:
{file_path}

Change type:
{summary["change_type"]}

Reusable patch pattern:
{edit_hint}

Minimal edit summary:
- Added lines: {summary["added_count"]}
- Removed lines: {summary["removed_count"]}

Representative added lines:
{compact_text(added_excerpt, 1200)}

Representative removed lines:
{compact_text(removed_excerpt, 1200)}

Search keywords:
{keywords}

Retrieval query pattern:
{compact_text(search, args.max_search_chars)}
"""
        docs.append(_knowledge_doc("sft_final", row, "patch_pattern", commit_message[:120] or file_path, body))
        return docs

    task = _section_after("Task", user, 2500) or _section_after("Instruction", user, 2500) or _section_after("Problem", user, 2500)
    tests = _tests_from_record(row)
    if dataset == "bigcodebench":
        code_prompt = _section_after("Code prompt", user, 2500)
        libs = _section_after("Libraries", user, 800)
        body = f"""
Task/API usage:
{task}

Relevant libraries/APIs:
{libs or "No explicit third-party library list."}

Function signature / scaffold:
{code_prompt}

Reusable implementation pattern:
{compact_text(code, args.max_code_chars)}

Boundary conditions from public tests:
{compact_text(tests, args.max_tests_chars)}
"""
        docs.append(_knowledge_doc("sft_final", row, "api_usage_pattern", task[:120], body))
        docs.append(_knowledge_doc("sft_final", row, "test_boundary_pattern", task[:120], f"Public tests and checked behavior:\n{compact_text(tests, args.max_tests_chars)}"))
        return docs

    if dataset == "livecodebench":
        body = f"""
Problem pattern:
{task}

Retrieval query / strategy signal:
{compact_text(search, args.max_search_chars)}

Strategy note:
{compact_text(answer, args.max_answer_chars)}

Public examples:
{compact_text(tests, args.max_tests_chars)}
"""
        docs.append(_knowledge_doc("sft_final", row, "algorithm_pattern", task[:120], body))
        return docs

    body = f"""
Programming task:
{task}

Reusable algorithm / implementation:
{compact_text(code, args.max_code_chars)}

Boundary conditions from public tests:
{compact_text(tests, args.max_tests_chars)}
"""
    docs.append(_knowledge_doc("sft_final", row, "algorithm_pattern", task[:120], body))
    docs.append(_knowledge_doc("sft_final", row, "test_boundary_pattern", task[:120], f"Public tests and checked behavior:\n{compact_text(tests, args.max_tests_chars)}"))
    return docs


def _rl_knowledge_docs(row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset = row.get("dataset")
    prompt = row.get("prompt", "")
    tests = row.get("tests") or {}
    execution = row.get("execution") or {}
    reward_spec = row.get("reward_spec") or {}
    task = _section_after("Task", prompt, 2500) or _section_after("Instruction", prompt, 2500) or _section_after("Problem", prompt, 2500) or _section_after("Commit message", prompt, 1800)
    public_view = public_tests_view(tests)
    docs: list[dict[str, Any]] = []

    if dataset == "commitpackft":
        file_path = _line_after("File", prompt)
        keywords = _keyword_summary(task, file_path)
        edit_hint = _patch_edit_hint(task, file_path)
        body = f"""
Patch intent:
{task}

Affected file:
{file_path}

Reusable task pattern:
{edit_hint}

Evaluation signals:
patch_apply, syntax_or_static_check, diff_similarity, intent_match

Reference policy:
Reference patches or after-code may exist for scoring, but are not included in this retrievable document.

Search keywords:
{keywords}
"""
        docs.append(_knowledge_doc("rl_final", row, "patch_task_pattern", task[:120] or file_path, body, {"execution_type": execution.get("type")}))
        return docs

    if dataset == "livecodebench":
        body = f"""
Algorithmic task pattern:
{task}

Execution style:
stdin/stdout Python program, evaluated with public examples and hidden tests when available.

Public examples and boundary signals:
{json_excerpt(public_view, args.max_tests_chars)}

Reward signals:
{json_excerpt(reward_spec, args.max_metadata_chars)}
"""
        docs.append(_knowledge_doc("rl_final", row, "algorithm_task_pattern", task[:120], body, {"execution_type": execution.get("type")}))
        return docs

    doc_type = "function_unittest_pattern" if dataset == "bigcodebench" else "function_assert_pattern"
    body = f"""
Function task pattern:
{task}

Execution style:
{json_excerpt(execution, args.max_metadata_chars)}

Public tests and boundary signals:
{json_excerpt(public_view, args.max_tests_chars)}

Reward signals:
{json_excerpt(reward_spec, args.max_metadata_chars)}
"""
    docs.append(_knowledge_doc("rl_final", row, doc_type, task[:120], body, {"execution_type": execution.get("type")}))
    docs.append(_knowledge_doc("rl_final", row, "test_boundary_pattern", task[:120], f"Public tests and checked behavior:\n{json_excerpt(public_view, args.max_tests_chars)}"))
    return docs


def build_corpus(args: argparse.Namespace) -> Counter:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.output_dir / "corpus.jsonl"
    counts: Counter = Counter()
    seen_doc_ids: set[str] = set()

    with corpus_path.open("w", encoding="utf-8") as out:
        for row in read_jsonl(args.sft_input):
            for doc in _sft_knowledge_docs(row, args):
                if doc["doc_id"] in seen_doc_ids:
                    counts["skipped_duplicate_doc_id"] += 1
                    continue
                seen_doc_ids.add(doc["doc_id"])
                out.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")
                counts["total_docs"] += 1
                counts["source:sft_final"] += 1
                counts[f"dataset:{doc['source_dataset']}"] += 1
                counts[f"doc_type:{doc['doc_type']}"] += 1

        for row in read_jsonl(args.rl_input):
            for doc in _rl_knowledge_docs(row, args):
                if doc["doc_id"] in seen_doc_ids:
                    counts["skipped_duplicate_doc_id"] += 1
                    continue
                seen_doc_ids.add(doc["doc_id"])
                out.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")
                counts["total_docs"] += 1
                counts["source:rl_final"] += 1
                counts[f"dataset:{doc['source_dataset']}"] += 1
                counts[f"doc_type:{doc['doc_type']}"] += 1

    write_json(
        args.output_dir / "corpus_stats.json",
        {
            "corpus_path": str(corpus_path),
            "inputs": {"sft": str(args.sft_input), "rl": str(args.rl_input)},
            "counts": dict(sorted(counts.items())),
            "notes": [
                "Knowledge-style RAG corpus.",
                "Test data is excluded.",
                "Documents summarize reusable algorithms, API usage, test boundaries, patch intents, and execution/reward patterns.",
                "RL hidden tests, reference patches, and reference_after code are not included in retrievable text.",
                "This script does not build a vector index.",
            ],
        },
    )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-input", type=Path, default=DEFAULT_SFT_INPUT)
    parser.add_argument("--rl-input", type=Path, default=DEFAULT_RL_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-prompt-chars", type=int, default=3500)
    parser.add_argument("--max-think-chars", type=int, default=800)
    parser.add_argument("--max-search-chars", type=int, default=800)
    parser.add_argument("--max-code-chars", type=int, default=5000)
    parser.add_argument("--max-answer-chars", type=int, default=800)
    parser.add_argument("--max-tests-chars", type=int, default=2500)
    parser.add_argument("--max-metadata-chars", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_corpus(args)
    print(f"Wrote {counts['total_docs']} knowledge-style final RAG docs to {args.output_dir / 'corpus.jsonl'}")
    for key, count in sorted((k, v) for k, v in counts.items() if k != "total_docs"):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
