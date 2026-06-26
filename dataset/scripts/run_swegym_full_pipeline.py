#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


def run_cmd(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
) -> Tuple[int, str, str]:
    p = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return p.returncode, p.stdout, p.stderr


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick(row: Dict[str, Any], names: List[str], default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def normalize_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    repo = pick(row, ["repo", "repository", "repo_name"])
    instance_id = pick(row, ["instance_id", "id", "task_id"], f"instance_{idx}")
    base_commit = pick(row, ["base_commit", "commit", "base_sha"])
    problem_statement = pick(row, ["problem_statement", "prompt", "issue", "description"])
    patch = pick(row, ["patch", "gold_patch", "solution_patch", "final_patch"])

    return {
        **row,
        "repo": repo,
        "instance_id": instance_id,
        "base_commit": base_commit,
        "problem_statement": problem_statement,
        "patch": patch,
        "test_patch": pick(row, ["test_patch"], ""),
        "FAIL_TO_PASS": pick(row, ["FAIL_TO_PASS", "fail_to_pass"], []),
        "PASS_TO_PASS": pick(row, ["PASS_TO_PASS", "pass_to_pass"], []),
    }


def extract_modified_files_from_patch(patch: str) -> List[str]:
    files = []
    for m in re.finditer(r"diff --git a/(.*?) b/(.*?)\n", patch or ""):
        old_path, new_path = m.group(1), m.group(2)
        if new_path != "/dev/null":
            files.append(new_path)
        elif old_path != "/dev/null":
            files.append(old_path)
    return sorted(set(files))


def valid_instance(row: Dict[str, Any]) -> bool:
    if not row.get("repo"):
        return False
    if "/" not in row["repo"]:
        return False
    if not row.get("base_commit"):
        return False
    if not row.get("problem_statement", "").strip():
        return False
    if not row.get("patch", "").strip():
        return False
    if not extract_modified_files_from_patch(row.get("patch", "")):
        return False
    return True


def repo_path(repo_dir: Path, repo: str) -> Path:
    owner, name = repo.split("/", 1)
    return repo_dir / owner / name


def safe_id(x: str) -> str:
    return x.replace("/", "__").replace(":", "__")


def is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def is_git_worktree(path: Path) -> bool:
    return (path / ".git").exists()


def clone_one_repo(repo: str, repo_dir: Path, github_base_url: str, clone_mode: str) -> Dict[str, Any]:
    target = repo_path(repo_dir, repo)
    target.parent.mkdir(parents=True, exist_ok=True)

    if is_git_repo(target):
        return {"repo": repo, "status": "exists", "path": str(target)}

    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)

    url = f"{github_base_url.rstrip('/')}/{repo}.git"

    if clone_mode == "full":
        cmd = ["git", "clone", url, str(tmp)]
    else:
        cmd = ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(tmp)]

    rc, out, err = run_cmd(cmd)

    if rc != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        return {
            "repo": repo,
            "status": "failed",
            "url": url,
            "stderr": err[-5000:],
        }

    tmp.rename(target)

    return {
        "repo": repo,
        "status": "cloned",
        "path": str(target),
        "url": url,
        "clone_mode": clone_mode,
    }


def clone_repos(
    repos: List[str],
    repo_dir: Path,
    logs_dir: Path,
    github_base_url: str,
    clone_mode: str,
    max_workers: int,
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)

    success_log = logs_dir / "clone_success.jsonl"
    failed_log = logs_dir / "clone_failed.jsonl"

    if success_log.exists():
        success_log.unlink()
    if failed_log.exists():
        failed_log.unlink()

    print(f"[Clone] repos: {len(repos)}")
    print(f"[Clone] mode : {clone_mode}")
    print(f"[Clone] dir  : {repo_dir}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(clone_one_repo, repo, repo_dir, github_base_url, clone_mode): repo
            for repo in repos
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="clone repos"):
            res = fut.result()
            if res["status"] in {"exists", "cloned"}:
                append_jsonl(success_log, res)
            else:
                append_jsonl(failed_log, res)

    print("[Clone] success:", success_log)
    print("[Clone] failed :", failed_log)


def ensure_commit(repo_dir: Path, row: Dict[str, Any], timeout: int) -> Tuple[bool, str]:
    rp = repo_path(repo_dir, row["repo"])
    commit = row["base_commit"]

    rc, _, _ = run_cmd(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=rp,
        timeout=timeout,
    )

    if rc == 0:
        return True, "exists"

    rc, out, err = run_cmd(
        ["git", "fetch", "origin", commit, "--filter=blob:none"],
        cwd=rp,
        timeout=timeout,
    )

    if rc != 0:
        return False, err[-3000:]

    rc, _, err = run_cmd(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=rp,
        timeout=timeout,
    )

    return rc == 0, err[-3000:]


def create_worktree(
    repo_dir: Path,
    tmp_dir: Path,
    row: Dict[str, Any],
    timeout: int,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    rp = repo_path(repo_dir, row["repo"]).resolve()
    wt = (tmp_dir / safe_id(row["instance_id"])).resolve()
    commit = row["base_commit"]

    if not is_git_repo(rp):
        return None, {
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "reason": "repo_missing",
            "repo_path": str(rp),
        }

    if wt.exists():
        if is_git_worktree(wt):
            run_cmd(["git", "worktree", "remove", "--force", str(wt)], cwd=rp, timeout=timeout)
        else:
            shutil.rmtree(wt, ignore_errors=True)

    ok, msg = ensure_commit(repo_dir, row, timeout)
    if not ok:
        return None, {
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": commit,
            "reason": "commit_unavailable",
            "message": msg,
        }

    wt.parent.mkdir(parents=True, exist_ok=True)

    rc, out, err = run_cmd(
        ["git", "worktree", "add", "--detach", str(wt), commit],
        cwd=rp,
        timeout=timeout,
    )

    if rc != 0:
        return None, {
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": commit,
            "reason": "worktree_add_failed",
            "stderr": err[-5000:],
        }

    return wt, None


def remove_worktree(repo_dir: Path, row: Dict[str, Any], wt: Path, timeout: int) -> None:
    rp = repo_path(repo_dir, row["repo"]).resolve()

    if wt.exists() and is_git_repo(rp):
        run_cmd(["git", "worktree", "remove", "--force", str(wt)], cwd=rp, timeout=timeout)

    if wt.exists():
        shutil.rmtree(wt, ignore_errors=True)


SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".tox",
    "node_modules", "dist", "build", ".venv", "venv", "site-packages",
}

EXTS = {".py", ".pyi", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json"}


def should_skip(rel: Path, include_tests: bool) -> bool:
    parts = set(rel.parts)

    if any(part.startswith("._") for part in rel.parts):
        return True

    if parts & SKIP_DIRS:
        return True

    if not include_tests:
        lowered = [p.lower() for p in rel.parts]
        name = rel.name.lower()
        if "test" in lowered or "tests" in lowered:
            return True
        if name.startswith("test_") or name.endswith("_test.py") or name.endswith("tests.py"):
            return True

    return False


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def chunk_file(
    root: Path,
    path: Path,
    chunk_lines: int,
    overlap_lines: int,
    max_chunk_chars: int,
) -> List[Dict[str, Any]]:
    rel = str(path.relative_to(root))
    text = read_text(path)

    if not text.strip():
        return []

    lines = text.splitlines()
    step = max(1, chunk_lines - overlap_lines)
    chunks = []

    for start in range(0, len(lines), step):
        end = min(len(lines), start + chunk_lines)
        chunk_text = "\n".join(lines[start:end])

        if not chunk_text.strip():
            continue

        if len(chunk_text) > max_chunk_chars:
            chunk_text = chunk_text[:max_chunk_chars] + "\n# [TRUNCATED]\n"

        chunks.append({
            "path": rel,
            "start_line": start + 1,
            "end_line": end,
            "text": chunk_text,
        })

        if end >= len(lines):
            break

    return chunks


def scan_chunks(
    root: Path,
    include_tests: bool,
    max_file_bytes: int,
    chunk_lines: int,
    overlap_lines: int,
    max_chunk_chars: int,
) -> List[Dict[str, Any]]:
    chunks = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(root)

        if should_skip(rel, include_tests):
            continue

        if path.suffix not in EXTS:
            continue

        try:
            if path.stat().st_size > max_file_bytes:
                continue
        except Exception:
            continue

        chunks.extend(
            chunk_file(
                root=root,
                path=path,
                chunk_lines=chunk_lines,
                overlap_lines=overlap_lines,
                max_chunk_chars=max_chunk_chars,
            )
        )

    return chunks


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize(text: str) -> List[str]:
    return [x.lower() for x in TOKEN_RE.findall(text or "")]


def bm25_rank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    q_terms = tokenize(query)
    docs = [tokenize(c["path"] + "\n" + c["text"]) for c in chunks]
    lens = [len(d) for d in docs]
    avgdl = sum(lens) / max(1, len(lens))

    df = Counter()
    for doc in docs:
        for t in set(doc):
            df[t] += 1

    n = len(docs)
    idf = {
        t: math.log(1 + (n - f + 0.5) / (f + 0.5))
        for t, f in df.items()
    }

    q_count = Counter(q_terms)
    scored = []

    for i, doc in enumerate(docs):
        tf = Counter(doc)
        dl = lens[i] or 1
        score = 0.0

        for t, qf in q_count.items():
            if t not in tf:
                continue

            freq = tf[t]
            denom = freq + k1 * (1 - b + b * dl / avgdl)
            score += idf.get(t, 0.0) * (freq * (k1 + 1) / denom) * qf

        if score > 0:
            item = dict(chunks[i])
            item["score"] = score
            scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def format_context(contexts: List[Dict[str, Any]], max_context_chars: int) -> str:
    parts = []
    used = 0

    for i, c in enumerate(contexts, 1):
        path = c["path"]
        lang = "python" if path.endswith((".py", ".pyi")) else ""
        block = (
            f"\n\n### Context {i}: {path}:{c.get('start_line', '?')}-{c.get('end_line', '?')}\n"
            f"```{lang}\n{c['text']}\n```"
        )

        if used + len(block) > max_context_chars:
            remain = max_context_chars - used
            if remain > 500:
                parts.append(block[:remain] + "\n[CONTEXT TRUNCATED]\n")
            break

        parts.append(block)
        used += len(block)

    return "".join(parts).strip()



def parse_maybe_list_for_query(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            x = json.loads(s)
            if isinstance(x, list):
                return [str(v) for v in x]
        except Exception:
            pass
        return [s]
    return [str(value)]


def normalize_query_text(text: str) -> str:
    text = text or ""
    text = text.replace("/", " ")
    text = text.replace("::", " ")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace(".", " ")
    return text


def build_retrieval_query(row: Dict[str, Any]) -> str:
    """
    Code-aware retrieval query.

    不使用 gold patch。
    使用：
    - repo
    - issue/problem_statement
    - hints_text
    - FAIL_TO_PASS / PASS_TO_PASS 测试名

    这样能把 test_dynamodb_update_expressions / update_item_add_float
    这类信号加入 query，比纯 issue BM25 更容易命中源码文件。
    """
    parts = []

    parts.append(row.get("repo", ""))
    parts.append(row.get("problem_statement", ""))
    parts.append(row.get("hints_text", ""))

    for key in ["FAIL_TO_PASS", "PASS_TO_PASS"]:
        for item in parse_maybe_list_for_query(row.get(key)):
            parts.append(normalize_query_text(item))

    return "\n".join(x for x in parts if x and str(x).strip())


def make_search_query(row: Dict[str, Any], gold_files: List[str]) -> str:
    problem = row.get("problem_statement", "")
    title = ""

    for line in problem.splitlines():
        line = line.strip()
        if line and not line.startswith("<!--"):
            title = line
            break

    file_terms = []
    for p in gold_files:
        file_terms.append(Path(p).stem)
        file_terms.append(Path(p).parent.name)

    raw = title + " " + " ".join(file_terms)

    toks = []
    seen = set()

    for t in TOKEN_RE.findall(raw):
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        toks.append(t)
        if len(toks) >= 24:
            break

    return " ".join(toks) or "repository issue related code"


def build_stage1(row: Dict[str, Any], context_text: str) -> Dict[str, Any]:
    user = f"""You are given a GitHub issue and relevant repository context.

Generate a minimal unified diff patch that fixes the issue.

Requirements:
- Only output a valid unified diff patch.
- Do not explain.
- Do not modify tests.
- Keep the change minimal.

Repository: {row["repo"]}
Base commit: {row["base_commit"]}

# Issue

{row.get("problem_statement", "").strip()}

# Retrieved Repository Context

{context_text}
"""

    return {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "stage": "stage1_rag_patch_sft",
        "messages": [
            {
                "role": "system",
                "content": "You are a software engineering agent that writes minimal unified diff patches.",
            },
            {
                "role": "user",
                "content": user,
            },
            {
                "role": "assistant",
                "content": row.get("patch", "").strip(),
            },
        ],
    }


def build_stage2(row: Dict[str, Any], gold_files: List[str], include_sandbox_action: bool) -> Dict[str, Any]:
    actions = []
    actions.append(f"<search_code>\n{make_search_query(row, gold_files)}\n</search_code>")

    for p in gold_files:
        actions.append(f"<open_file>\n{p}\n</open_file>")

    if include_sandbox_action:
        py_files = [p for p in gold_files if p.endswith((".py", ".pyi"))]
        if py_files:
            actions.append(f"<run_sandbox>\npython -m py_compile {' '.join(py_files)}\n</run_sandbox>")

    actions.append(f"<generate_patch>\n{row.get('patch', '').strip()}\n</generate_patch>")
    actions.append("<final>\nPATCH_READY\n</final>")

    user = f"""You are a software engineering agent.

Available actions:
- <direct_answer>answer</direct_answer>
- <search_code>query</search_code>
- <open_file>path</open_file>
- <run_sandbox>command</run_sandbox>
- <generate_patch>unified diff patch</generate_patch>
- <final>status</final>

Given the issue below, choose useful actions and produce a minimal patch.

Repository: {row["repo"]}
Base commit: {row["base_commit"]}

# Issue

{row.get("problem_statement", "").strip()}
"""

    return {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "stage": "stage2_agent_trajectory_sft",
        "modified_files": gold_files,
        "messages": [
            {
                "role": "system",
                "content": "You are an autonomous software engineering agent that can search code, open files, run lightweight sandbox checks, and generate patches.",
            },
            {
                "role": "user",
                "content": user,
            },
            {
                "role": "assistant",
                "content": "\n\n".join(actions),
            },
        ],
    }


def build_grpo_prompt(row: Dict[str, Any]) -> Dict[str, Any]:
    gold_files = extract_modified_files_from_patch(row.get("patch", ""))

    user = f"""You are a software engineering agent.

You may use the following actions:
- <direct_answer>answer</direct_answer>
- <search_code>query</search_code>
- <open_file>path</open_file>
- <run_sandbox>command</run_sandbox>
- <generate_patch>unified diff patch</generate_patch>
- <final>status</final>

Task:
Given the GitHub issue below, decide whether to search code, open files, run lightweight sandbox checks, and generate a minimal patch.

Repository: {row["repo"]}
Base commit: {row["base_commit"]}

# Issue

{row.get("problem_statement", "").strip()}
"""

    return {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "messages": [
            {
                "role": "system",
                "content": "You are an autonomous software engineering agent. Produce tool actions and a minimal patch when needed.",
            },
            {
                "role": "user",
                "content": user,
            },
        ],
        "reward_meta": {
            "gold_patch": row.get("patch", ""),
            "gold_files": gold_files,
            "forbidden_modify_tests": True,
            "reward_items": [
                "valid_action_format",
                "search_or_open_hits_gold_file",
                "patch_can_apply",
                "does_not_modify_tests",
                "changed_python_files_compile",
                "patch_similarity_to_gold",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="SWE-Gym/SWE-Gym")
    parser.add_argument("--local_parquet", type=Path, default=None)
    parser.add_argument("--hf_endpoint", default=None)
    parser.add_argument("--project_dir", type=Path, default=Path("./data/swegym_full_project"))

    parser.add_argument("--clone_mode", choices=["partial", "full"], default="full")
    parser.add_argument("--github_base_url", default="https://github.com")
    parser.add_argument("--max_workers_clone", type=int, default=4)
    parser.add_argument("--force_redownload", action="store_true")

    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--chunk_lines", type=int, default=120)
    parser.add_argument("--overlap_lines", type=int, default=20)
    parser.add_argument("--max_chunk_chars", type=int, default=16000)
    parser.add_argument("--max_context_chars", type=int, default=60000)
    parser.add_argument("--max_file_bytes", type=int, default=300000)
    parser.add_argument("--include_tests", action="store_true")
    parser.add_argument("--include_sandbox_action", action="store_true")
    parser.add_argument("--git_timeout", type=int, default=900)

    args = parser.parse_args()

    args.project_dir = args.project_dir.resolve()
    meta_dir = args.project_dir / "meta"
    repo_dir = args.project_dir / "repos"
    out_dir = args.project_dir / "processed"
    tmp_dir = args.project_dir / "tmp_worktrees"
    logs_dir = args.project_dir / "logs"

    for d in [meta_dir, repo_dir, out_dir, tmp_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")
        print("[HF] endpoint:", os.environ["HF_ENDPOINT"])

    from datasets import load_dataset

    print("[Load] dataset:", args.dataset_name)

    load_kwargs = {}
    if args.force_redownload:
        load_kwargs["download_mode"] = "force_redownload"

    if args.local_parquet is not None:
        args.local_parquet = args.local_parquet.resolve()
        print("[Load] local parquet:", args.local_parquet)
        ds = load_dataset("parquet", data_files={"train": str(args.local_parquet)})
    else:
        ds = load_dataset(args.dataset_name, **load_kwargs)

    if hasattr(ds, "keys"):
        split_names = list(ds.keys())
    else:
        split_names = ["train"]
        ds = {"train": ds}

    print("[Dataset splits]", split_names)

    all_raw = []
    for split in split_names:
        for r in ds[split]:
            nr = normalize_row(dict(r), idx=len(all_raw))
            nr["_split"] = split
            all_raw.append(nr)

    print("[Dataset rows raw]", len(all_raw))

    rows = [r for r in all_raw if valid_instance(r)]

    print("[Dataset rows valid]", len(rows))

    if not rows:
        print("[ERROR] No valid rows. Dataset columns example:")
        print(list(all_raw[0].keys()) if all_raw else "EMPTY")
        raise SystemExit(1)

    write_jsonl(meta_dir / "all_instances.raw.jsonl", all_raw)
    write_jsonl(meta_dir / "all_instances.valid.jsonl", rows)

    repos = sorted(set(r["repo"] for r in rows))
    (meta_dir / "repos.txt").write_text("\n".join(repos) + "\n", encoding="utf-8")

    write_json(meta_dir / "dataset_stats.json", {
        "dataset_name": args.dataset_name,
        "raw_rows": len(all_raw),
        "valid_rows": len(rows),
        "repos": len(repos),
        "splits": split_names,
        "note": "All valid SWE-Gym rows are used for SFT and GRPO data.",
    })

    print("[Repos]", len(repos))

    clone_repos(
        repos=repos,
        repo_dir=repo_dir,
        logs_dir=logs_dir,
        github_base_url=args.github_base_url,
        clone_mode=args.clone_mode,
        max_workers=args.max_workers_clone,
    )

    retrieval_path = out_dir / "retrieval_bm25_topk.jsonl"
    stage1_path = out_dir / "stage1_rag_patch_sft.jsonl"
    stage2_path = out_dir / "stage2_agent_trajectory_sft.jsonl"
    mixed_path = out_dir / "stage1_stage2_mixed_sft.jsonl"
    grpo_path = out_dir / "grpo_prompts.jsonl"
    fail_path = out_dir / "failures.jsonl"

    for p in [retrieval_path, stage1_path, stage2_path, mixed_path, grpo_path, fail_path]:
        if p.exists():
            p.unlink()

    stats = {
        "input_valid_rows": len(rows),
        "processed_success": 0,
        "processed_failures": 0,
        "hit_gold_file": 0,
        "stage1_sft": 0,
        "stage2_sft": 0,
        "mixed_sft": 0,
        "grpo": 0,
    }

    for row in tqdm(rows, desc="build RAG/SFT/GRPO"):
        wt, err = create_worktree(repo_dir, tmp_dir, row, args.git_timeout)

        if err:
            stats["processed_failures"] += 1
            append_jsonl(fail_path, err)
            continue

        assert wt is not None

        try:
            chunks = scan_chunks(
                root=wt,
                include_tests=args.include_tests,
                max_file_bytes=args.max_file_bytes,
                chunk_lines=args.chunk_lines,
                overlap_lines=args.overlap_lines,
                max_chunk_chars=args.max_chunk_chars,
            )

            query = build_retrieval_query(row)
            contexts = bm25_rank(query, chunks, args.top_k)

            gold_files = extract_modified_files_from_patch(row.get("patch", ""))
            hit = any(c["path"] in set(gold_files) for c in contexts)

            retrieval = {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "query": query,
                "gold_files": gold_files,
                "hit_gold_file": hit,
                "num_chunks": len(chunks),
                "contexts": contexts,
            }
            append_jsonl(retrieval_path, retrieval)

            context_text = format_context(contexts, args.max_context_chars)
            if not context_text.strip():
                raise RuntimeError("empty retrieved context")

            ex1 = build_stage1(row, context_text)
            ex2 = build_stage2(row, gold_files, args.include_sandbox_action)
            ex3 = build_grpo_prompt(row)

            append_jsonl(stage1_path, ex1)
            append_jsonl(stage2_path, ex2)
            append_jsonl(mixed_path, ex1)
            append_jsonl(mixed_path, ex2)
            append_jsonl(grpo_path, ex3)

            stats["processed_success"] += 1
            stats["stage1_sft"] += 1
            stats["stage2_sft"] += 1
            stats["mixed_sft"] += 2
            stats["grpo"] += 1

            if hit:
                stats["hit_gold_file"] += 1

        except Exception as e:
            stats["processed_failures"] += 1
            append_jsonl(fail_path, {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "reason": "exception",
                "error": repr(e),
            })

        finally:
            remove_worktree(repo_dir, row, wt, args.git_timeout)

    write_json(out_dir / "build_stats.json", stats)

    print("\n[ALL DONE]")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("Project dir:", args.project_dir)
    print("Stage1 SFT :", stage1_path)
    print("Stage2 SFT :", stage2_path)
    print("Mixed SFT  :", mixed_path)
    print("Weak GRPO  :", grpo_path)
    print("Retrieval  :", retrieval_path)


if __name__ == "__main__":
    main()
