"""Rule-based reward for code-agent RL.

The function exported here follows verl's custom reward signature:

    compute_score(data_source, solution_str, ground_truth, extra_info=None)

It evaluates the protocol first, then executes only extracted code or patch
content according to the task execution type stored in ``ground_truth``.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT = 10


ACTION_TAGS = ("search_code", "open_file", "run_sandbox", "generate_patch", "final")
KNOWN_TAGS = ("think", *ACTION_TAGS)
TAG_RE = re.compile(r"<(?P<tag>think|search_code|open_file|run_sandbox|generate_patch|final)>(?P<body>.*?)</(?P=tag)>", re.DOTALL)


def _json_loads_maybe(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_tags(text: str) -> dict[str, list[str]]:
    tags: dict[str, list[str]] = {tag: [] for tag in KNOWN_TAGS}
    for match in TAG_RE.finditer(text or ""):
        tags[match.group("tag")].append(match.group("body").strip())
    return tags


def _strip_markdown_fence(text: str) -> str:
    text = (text or "").strip()
    fenced = re.fullmatch(r"```(?:python|py|diff|patch)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def _extract_diff_from_text(text: str) -> str:
    text = _strip_markdown_fence(text)
    if not text:
        return ""
    match = re.search(r"^diff --git\s+", text, flags=re.MULTILINE)
    if match:
        return text[match.start() :].strip()
    match = re.search(r"^---\s+", text, flags=re.MULTILINE)
    if match and re.search(r"^\+\+\+\s+", text[match.start() :], flags=re.MULTILINE):
        return text[match.start() :].strip()
    return ""


def _extract_executable(solution: str) -> tuple[str, str]:
    """Return final patch/code content and the protocol source tag."""
    tags = _parse_tags(solution)
    if tags["generate_patch"]:
        return _strip_markdown_fence(tags["generate_patch"][-1]), "generate_patch"
    if tags["final"]:
        final_text = _strip_markdown_fence(tags["final"][-1])
        final_diff = _extract_diff_from_text(final_text)
        return (final_diff or final_text), "final"
    raw_diff = _extract_diff_from_text(solution)
    if raw_diff:
        return raw_diff, "raw_diff"
    return _strip_markdown_fence(solution), "raw"


def _format_score(solution: str, gt: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    tags = _parse_tags(solution)
    required = gt.get("required_tags") or []
    optional = set(gt.get("optional_tags") or ACTION_TAGS)
    allowed_tags = set(required) | optional
    allowed_tags.update({"think"})

    required_hits = sum(1 for tag in required if tags.get(tag))
    _, source_tag = _extract_executable(solution)
    terminal_score = 1.0 if source_tag in {"generate_patch", "final", "raw_diff"} else 0.0
    required_score = required_hits / max(1, len(required)) if required else terminal_score

    env_tag_penalty = 0.0
    if re.search(r"</?(information|observation)>", solution or "", flags=re.IGNORECASE):
        env_tag_penalty = 0.25

    malformed_penalty = 0.0
    for tag in KNOWN_TAGS:
        opens = len(re.findall(fr"<{tag}>", solution or ""))
        closes = len(re.findall(fr"</{tag}>", solution or ""))
        if opens != closes:
            malformed_penalty += 0.15

    needs_code = bool((gt.get("expected_behavior") or {}).get("needs_code", True))
    code_score = 1.0
    if needs_code and source_tag not in {"generate_patch", "final", "raw_diff"}:
        code_score = 0.0

    extra_tags = [
        match.group(2)
        for match in re.finditer(r"<(/?)([a-zA-Z_][a-zA-Z0-9_]*)>", solution or "")
        if match.group(2) not in allowed_tags
    ]
    unknown_penalty = min(0.2, 0.05 * len(extra_tags))

    score = max(0.0, 0.7 * required_score + 0.3 * code_score - env_tag_penalty - malformed_penalty - unknown_penalty)
    return score, {
        "has_think": bool(tags["think"]),
        "has_search_code": bool(tags["search_code"]),
        "has_open_file": bool(tags["open_file"]),
        "has_run_sandbox": bool(tags["run_sandbox"]),
        "has_generate_patch": bool(tags["generate_patch"]),
        "has_final": bool(tags["final"]),
        "executable_source": source_tag,
        "format_penalty": env_tag_penalty + malformed_penalty + unknown_penalty,
    }


def _compile_python(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code or "")
        return True, ""
    except SyntaxError as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run_python_script(code: str, stdin: str = "", timeout: int = DEFAULT_TIMEOUT) -> tuple[int, str, str, bool]:
    with tempfile.TemporaryDirectory(prefix="code_agent_reward_") as tmpdir:
        script_path = Path(tmpdir) / "main.py"
        script_path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["python", str(script_path)],
                input=stdin,
                text=True,
                capture_output=True,
                cwd=tmpdir,
                timeout=timeout,
            )
            return proc.returncode, _to_text(proc.stdout), _to_text(proc.stderr), False
        except subprocess.TimeoutExpired as exc:
            stdout = _to_text(exc.stdout)
            stderr = _to_text(exc.stderr) or "timeout"
            return 124, stdout, stderr, True
        except Exception as exc:
            return 1, "", f"{exc.__class__.__name__}: {exc}", False


def _run_exec_tests(code: str, tests: list[str], setup: str = "", timeout: int = DEFAULT_TIMEOUT) -> tuple[float, int, int, str]:
    if not tests:
        return 0.0, 0, 0, "no tests"

    passed = 0
    details: list[str] = []
    for idx, test in enumerate(tests):
        script = "\n\n".join(part for part in [setup, code, str(test)] if part)
        returncode, stdout, stderr, timed_out = _run_python_script(script, timeout=timeout)
        ok = returncode == 0 and not timed_out
        passed += int(ok)
        if not ok and len(details) < 3:
            details.append(f"test_{idx}: rc={returncode} stderr={stderr[-500:]} stdout={stdout[-300:]}")
    return passed / len(tests), passed, len(tests), "\n".join(details)


def _run_unittest_tests(code: str, tests: list[str], setup: str = "", timeout: int = DEFAULT_TIMEOUT) -> tuple[float, int, int, str]:
    if not tests:
        return 0.0, 0, 0, "no tests"

    passed = 0
    details: list[str] = []
    for idx, test in enumerate(tests):
        script = "\n\n".join(
            part
            for part in [
                setup,
                code,
                str(test),
                "if __name__ == '__main__':\n    unittest.main()",
            ]
            if part
        )
        returncode, stdout, stderr, timed_out = _run_python_script(script, timeout=timeout)
        ok = returncode == 0 and not timed_out
        passed += int(ok)
        if not ok and len(details) < 3:
            details.append(f"unittest_{idx}: rc={returncode} stderr={stderr[-700:]} stdout={stdout[-300:]}")
    return passed / len(tests), passed, len(tests), "\n".join(details)


def _normalize_stdout(text: Any) -> str:
    normalized = _to_text(text).strip()
    return "\n".join(line.rstrip() for line in normalized.splitlines()).strip()


def _run_stdin_tests(code: str, cases: list[dict[str, Any]], timeout: int = DEFAULT_TIMEOUT) -> tuple[float, int, int, str]:
    if not cases:
        return 0.0, 0, 0, "no stdin tests"

    passed = 0
    details: list[str] = []
    for idx, case in enumerate(cases):
        expected = _normalize_stdout(str(case.get("output", "")))
        returncode, stdout, stderr, timed_out = _run_python_script(
            code,
            stdin=str(case.get("input", "")),
            timeout=timeout,
        )
        actual = _normalize_stdout(stdout)
        ok = returncode == 0 and not timed_out and actual == expected
        passed += int(ok)
        if not ok and len(details) < 3:
            details.append(
                f"case_{idx}: rc={returncode} expected={expected[-300:]!r} actual={actual[-300:]!r} stderr={stderr[-500:]}"
            )
    return passed / len(cases), passed, len(cases), "\n".join(details)


def _looks_like_unified_diff(patch: str) -> bool:
    if re.search(r"^diff --git\s+", patch or "", re.MULTILINE):
        return True
    return bool(re.search(r"^---\s+", patch or "", re.MULTILINE)) and bool(
        re.search(r"^\+\+\+\s+", patch or "", re.MULTILINE)
    ) and bool(re.search(r"^@@\s+", patch or "", re.MULTILINE))


def _diff_similarity(pred_patch: str, ref_patch: str) -> float:
    if not pred_patch or not ref_patch:
        return 0.0
    return difflib.SequenceMatcher(None, pred_patch.strip(), ref_patch.strip()).ratio()


def _patch_intent_score(pred_patch: str, ref_patch: str, reference_after: str) -> float:
    if not pred_patch:
        return 0.0
    pred_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}|[0-9]+", pred_patch.lower()))
    ref_text = ref_patch or reference_after or ""
    ref_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}|[0-9]+", ref_text.lower()))
    if not ref_tokens:
        return 0.5 if _looks_like_unified_diff(pred_patch) else 0.0
    return min(1.0, len(pred_tokens & ref_tokens) / max(1, min(len(ref_tokens), 80)))


def _evaluate_patch(patch: str, tests: dict[str, Any]) -> dict[str, Any]:
    ref_patch = tests.get("reference_patch") or ""
    reference_after = tests.get("reference_after") or ""
    unified = _looks_like_unified_diff(patch)
    similarity = _diff_similarity(patch, ref_patch)
    intent = _patch_intent_score(patch, ref_patch, reference_after)
    # Without a before-file workspace, true git-apply cannot be checked here.
    patch_apply_proxy = 1.0 if unified else 0.0
    return {
        "patch_apply": patch_apply_proxy,
        "diff_similarity": similarity,
        "intent_match": intent,
        "syntax_or_static_check": 1.0 if unified else 0.0,
    }


def _modified_files_from_patch(patch: str) -> list[str]:
    files: list[str] = []
    for match in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", patch or "", flags=re.MULTILINE):
        files.append(match.group(2).strip())
    for match in re.finditer(r"^\+\+\+\s+(?:b/)?(.+?)$", patch or "", flags=re.MULTILINE):
        path = match.group(1).strip()
        if path != "/dev/null":
            files.append(path)
    return list(dict.fromkeys(files))


def _does_not_modify_tests(patch: str, gt: dict[str, Any]) -> float:
    if not gt.get("forbidden_modify_tests", False):
        return 1.0
    for path in _modified_files_from_patch(patch):
        lowered = path.lower()
        parts = lowered.split("/")
        if "tests" in parts or "test" in parts or lowered.startswith("test_") or "/test_" in lowered:
            return 0.0
    return 1.0


def _search_or_open_hits_gold_file(solution: str, gt: dict[str, Any]) -> float:
    gold_files = [str(x) for x in _as_list(gt.get("gold_files")) if x]
    if not gold_files:
        return 0.0
    tags = _parse_tags(solution)
    action_text = "\n".join(tags["search_code"] + tags["open_file"]).lower()
    if not action_text.strip():
        return 0.0
    for path in gold_files:
        lowered = path.lower()
        basename = lowered.rsplit("/", 1)[-1]
        if lowered in action_text or basename in action_text:
            return 1.0
    return 0.0


def _reward_weights(parts: dict[str, float], reward_spec: Any, reward_items: Any) -> dict[str, float]:
    if isinstance(reward_spec, dict):
        return reward_spec
    items = reward_spec if isinstance(reward_spec, list) else reward_items
    items = [str(item) for item in _as_list(items) if str(item) in parts]
    if not items:
        return {}
    weight = 1.0 / len(items)
    return {item: weight for item in items}


def _weighted_sum(parts: dict[str, float], weights: dict[str, Any]) -> float:
    if not weights:
        return sum(parts.values()) / max(1, len(parts))
    total = 0.0
    used_weight = 0.0
    for key, value in parts.items():
        weight = float(weights.get(key, 0.0) or 0.0)
        total += weight * max(0.0, min(1.0, value))
        used_weight += weight
    if used_weight < 1.0:
        total += (1.0 - used_weight) * max(0.0, min(1.0, parts.get("answer_quality", 0.0)))
    return max(0.0, min(1.0, total))


def _load_ground_truth(ground_truth: Any, extra_info: dict[str, Any] | None) -> dict[str, Any]:
    gt = _json_loads_maybe(ground_truth, {})
    extra_info = extra_info or {}
    if not isinstance(gt, dict):
        gt = {}
    reward_meta = _json_loads_maybe(extra_info.get("reward_meta"), {})
    if isinstance(reward_meta, dict):
        for key in ["gold_patch", "gold_files", "weak_reward_items", "reward_items", "forbidden_modify_tests"]:
            if key in reward_meta and key not in gt:
                gt[key] = reward_meta[key]
    if "reward_items" not in gt and "weak_reward_items" in gt:
        gt["reward_items"] = gt["weak_reward_items"]
    reference_patch = gt.get("reference_patch") or gt.get("gold_patch")
    if reference_patch:
        tests = gt.setdefault("tests", {})
        if isinstance(tests, dict) and not tests.get("reference_patch"):
            tests["reference_patch"] = reference_patch
        execution = gt.setdefault("execution", {})
        if isinstance(execution, dict) and not execution.get("type"):
            execution["type"] = "patch"
    for key, extra_key in [
        ("execution", "execution_json"),
        ("tests", "tests_json"),
        ("reward_spec", "reward_spec_json"),
        ("evaluation_spec", "evaluation_spec_json"),
        ("expected_behavior", "expected_behavior_json"),
    ]:
        if key not in gt or gt.get(key) in (None, {}, []):
            value = _json_loads_maybe(extra_info.get(extra_key), None)
            if value is not None:
                gt[key] = value
    if "reward_spec" not in gt and "evaluation_spec" in gt:
        gt["reward_spec"] = {}
    return gt


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Compute reward for code-agent outputs."""
    gt = _load_ground_truth(ground_truth, extra_info)
    execution = gt.get("execution") or {}
    tests = gt.get("tests") or {}
    reward_spec = gt.get("reward_spec") or {}
    reward_items = gt.get("reward_items") or gt.get("weak_reward_items") or []
    execution_type = execution.get("type")
    timeout = int(execution.get("timeout_sec") or os.environ.get("CODE_AGENT_REWARD_TIMEOUT", DEFAULT_TIMEOUT))

    fmt_score, fmt_info = _format_score(solution_str, gt)
    executable, executable_source = _extract_executable(solution_str or "")
    has_executable = bool(executable.strip())

    compile_score = 0.0
    compile_error = ""
    if has_executable and execution_type != "patch":
        ok, compile_error = _compile_python(executable)
        compile_score = 1.0 if ok else 0.0
    elif has_executable and execution_type == "patch":
        compile_score = 1.0 if _looks_like_unified_diff(executable) else 0.0

    public_score = 0.0
    hidden_score = 0.0
    public_passed = public_total = hidden_passed = hidden_total = 0
    test_details = ""
    patch_scores: dict[str, float] = {}

    if has_executable and compile_score > 0:
        setup = tests.get("setup") or ""
        public_tests = _as_list(tests.get("public"))
        hidden_tests = _as_list(tests.get("hidden"))

        if execution_type == "python_function":
            public_score, public_passed, public_total, test_details = _run_exec_tests(
                executable, [str(t) for t in public_tests], setup=setup, timeout=timeout
            )
            if hidden_tests:
                hidden_score, hidden_passed, hidden_total, _ = _run_exec_tests(
                    executable, [str(t) for t in hidden_tests], setup=setup, timeout=timeout
                )
        elif execution_type == "python_unittest":
            public_score, public_passed, public_total, test_details = _run_unittest_tests(
                executable, [str(t) for t in public_tests], setup=setup, timeout=timeout
            )
            if hidden_tests:
                hidden_score, hidden_passed, hidden_total, _ = _run_unittest_tests(
                    executable, [str(t) for t in hidden_tests], setup=setup, timeout=timeout
                )
        elif execution_type == "stdin":
            public_cases = [case for case in public_tests if isinstance(case, dict)]
            hidden_cases = [case for case in hidden_tests if isinstance(case, dict)]
            public_score, public_passed, public_total, test_details = _run_stdin_tests(
                executable, public_cases, timeout=timeout
            )
            if hidden_cases:
                hidden_score, hidden_passed, hidden_total, _ = _run_stdin_tests(
                    executable, hidden_cases, timeout=timeout
                )
        elif execution_type == "patch":
            patch_scores = _evaluate_patch(executable, tests)

    answer_quality = 1.0 if has_executable else 0.0
    code_extractable = 1.0 if executable_source in {"generate_patch", "final", "raw_diff"} and has_executable else 0.0

    if execution_type == "patch":
        parts = {
            "format": fmt_score,
            "valid_action_format": fmt_score,
            "code_extractable": code_extractable,
            "patch_apply": patch_scores.get("patch_apply", 0.0),
            "patch_can_apply": patch_scores.get("patch_apply", 0.0),
            "syntax_or_static_check": patch_scores.get("syntax_or_static_check", 0.0),
            "changed_python_files_compile": patch_scores.get("syntax_or_static_check", 0.0),
            "diff_similarity": patch_scores.get("diff_similarity", 0.0),
            "patch_similarity_to_gold": patch_scores.get("diff_similarity", 0.0),
            "intent_match": patch_scores.get("intent_match", 0.0),
            "search_or_open_hits_gold_file": _search_or_open_hits_gold_file(solution_str, gt),
            "does_not_modify_tests": _does_not_modify_tests(executable, gt),
            "answer_quality": answer_quality,
        }
    else:
        parts = {
            "format": fmt_score,
            "valid_action_format": fmt_score,
            "code_extractable": code_extractable,
            "compile": compile_score,
            "public_tests": public_score,
            "hidden_tests": hidden_score if hidden_total else public_score,
            "used_search_code": 1.0 if fmt_info["has_search_code"] else 0.0,
            "used_open_file": 1.0 if fmt_info["has_open_file"] else 0.0,
            "used_run_sandbox": 1.0 if fmt_info["has_run_sandbox"] else 0.0,
            "answer_quality": answer_quality,
        }

    score = _weighted_sum(parts, _reward_weights(parts, reward_spec, reward_items))
    pass_rate = public_score if public_total else 0.0

    return {
        "score": score,
        "format_score": fmt_score,
        "code_extractable": code_extractable,
        "compile_score": compile_score,
        "public_score": public_score,
        "hidden_score": hidden_score,
        "pass_rate": pass_rate,
        "public_passed": public_passed,
        "public_total": public_total,
        "hidden_passed": hidden_passed,
        "hidden_total": hidden_total,
        "execution_type": execution_type,
        "executable_source": executable_source,
        "has_executable": has_executable,
        "compile_error": compile_error,
        "test_details": test_details,
        "patch_apply": patch_scores.get("patch_apply", 0.0),
        "diff_similarity": patch_scores.get("diff_similarity", 0.0),
        "intent_match": patch_scores.get("intent_match", 0.0),
        **fmt_info,
    }
