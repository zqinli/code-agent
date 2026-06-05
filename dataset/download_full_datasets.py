#!/usr/bin/env python3
"""
Download the full source datasets for the coding-agent data recipe.

Priority:
  1. Try ModelScope dataset snapshots.
  2. Fall back to Hugging Face datasets.load_dataset.

All data and caches are rooted at /root/autodl-tmp/datasets by default.
Run with:
  conda run -n iworkplace python /root/rl-workplace/dataset/download_full_datasets.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/datasets")
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    purpose: str
    hf_path: str
    hf_kwargs: dict[str, Any]
    modelscope_ids: tuple[str, ...]


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="repobench",
        purpose="repo-level code completion, cross-file context, RAG",
        hf_path="tianyang/repobench_python_v1.1",
        hf_kwargs={},
        modelscope_ids=(
            "tianyang/repobench_python_v1.1",
            "AI-ModelScope/repobench_python_v1.1",
            "AI-ModelScope/RepoBench",
        ),
    ),
    DatasetSpec(
        name="commitpackft",
        purpose="commit-message-driven code edits, bug fixes, patch generation",
        hf_path="bigcode/commitpackft",
        hf_kwargs={"trust_remote_code": True},
        modelscope_ids=(
            "bigcode/commitpackft",
            "AI-ModelScope/commitpackft",
        ),
    ),
    DatasetSpec(
        name="bigcodebench",
        purpose="complex function-level code generation with executable tests",
        hf_path="bigcode/bigcodebench",
        hf_kwargs={},
        modelscope_ids=(
            "bigcode/bigcodebench",
            "AI-ModelScope/bigcodebench",
        ),
    ),
    DatasetSpec(
        name="livecodebench",
        purpose="recent code-generation problems for dynamic execution evaluation",
        hf_path="livecodebench/code_generation_lite",
        hf_kwargs={},
        modelscope_ids=(
            "livecodebench/code_generation_lite",
            "AI-ModelScope/code_generation_lite",
        ),
    ),
    DatasetSpec(
        name="taco",
        purpose="algorithmic reasoning and RLVR executable reward signal",
        hf_path="BAAI/TACO",
        hf_kwargs={"trust_remote_code": True},
        modelscope_ids=(
            "BAAI/TACO",
            "AI-ModelScope/TACO",
        ),
    ),
)


def setup_env(data_root: Path, hf_endpoint: str | None) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MODELSCOPE_CACHE", str(data_root / "modelscope_cache"))
    os.environ.setdefault("HF_HOME", str(data_root / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(data_root / "hf_datasets_cache"))
    if hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def try_modelscope_download(spec: DatasetSpec, data_root: Path, revision: str | None) -> dict[str, Any]:
    from modelscope.hub.snapshot_download import snapshot_download

    errors: list[dict[str, str]] = []
    target_dir = data_root / "modelscope" / spec.name
    for repo_id in spec.modelscope_ids:
        print(f"[modelscope] {spec.name}: trying {repo_id}", flush=True)
        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                cache_dir=str(data_root / "modelscope_cache"),
                local_dir=str(target_dir / repo_id.replace("/", "__")),
                max_workers=8,
            )
            return {
                "ok": True,
                "backend": "modelscope",
                "repo_id": repo_id,
                "local_path": str(local_path),
            }
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"[modelscope][warn] {spec.name}: {repo_id} failed: {message}", flush=True)
            errors.append({"repo_id": repo_id, "error": message})
    return {"ok": False, "backend": "modelscope", "errors": errors}


def try_hf_download(spec: DatasetSpec, data_root: Path) -> dict[str, Any]:
    from datasets import load_dataset

    print(f"[hf] {spec.name}: loading full dataset {spec.hf_path}", flush=True)
    dataset = load_dataset(spec.hf_path, cache_dir=str(data_root / "hf_datasets_cache"), **spec.hf_kwargs)
    summary = summarize_dataset(dataset)
    marker_dir = data_root / "huggingface" / spec.name
    write_json(
        marker_dir / "download_summary.json",
        {
            "hf_path": spec.hf_path,
            "hf_kwargs": spec.hf_kwargs,
            "cache_dir": os.environ.get("HF_DATASETS_CACHE"),
            "summary": summary,
        },
    )
    return {
        "ok": True,
        "backend": "huggingface",
        "hf_path": spec.hf_path,
        "summary": summary,
        "marker_dir": str(marker_dir),
    }


def summarize_dataset(dataset: Any) -> dict[str, Any]:
    if hasattr(dataset, "keys"):
        return {
            "type": type(dataset).__name__,
            "splits": {
                split: {
                    "num_rows": len(value),
                    "columns": list(getattr(value, "column_names", []) or []),
                }
                for split, value in dataset.items()
            },
        }
    return {
        "type": type(dataset).__name__,
        "num_rows": len(dataset),
        "columns": list(getattr(dataset, "column_names", []) or []),
    }


def download_one(
    spec: DatasetSpec,
    data_root: Path,
    prefer_modelscope: bool,
    allow_hf_fallback: bool,
    revision: str | None,
) -> dict[str, Any]:
    started = time.time()
    result: dict[str, Any] = {
        "name": spec.name,
        "purpose": spec.purpose,
        "hf_path": spec.hf_path,
        "modelscope_ids": list(spec.modelscope_ids),
        "started_at": started,
    }

    if prefer_modelscope:
        ms_result = try_modelscope_download(spec, data_root, revision)
        result["modelscope"] = ms_result
        if ms_result.get("ok"):
            result["status"] = "ok"
            result["selected_backend"] = "modelscope"
            result["elapsed_sec"] = round(time.time() - started, 3)
            return result

    if allow_hf_fallback:
        try:
            hf_result = try_hf_download(spec, data_root)
            result["huggingface"] = hf_result
            result["status"] = "ok"
            result["selected_backend"] = "huggingface"
        except Exception as exc:
            result["huggingface"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            result["status"] = "failed"
    else:
        result["status"] = "failed"

    result["elapsed_sec"] = round(time.time() - started, 3)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--hf_endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT))
    parser.add_argument("--revision", default=None, help="Optional ModelScope revision to download.")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[spec.name for spec in DATASETS],
        default=None,
        help="Download only selected dataset names.",
    )
    parser.add_argument(
        "--no_modelscope",
        action="store_true",
        help="Skip ModelScope and use Hugging Face directly.",
    )
    parser.add_argument(
        "--no_hf_fallback",
        action="store_true",
        help="Do not fall back to Hugging Face when ModelScope download fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    setup_env(data_root, args.hf_endpoint)

    selected = [spec for spec in DATASETS if args.only is None or spec.name in set(args.only)]
    manifest_path = data_root / "download_manifest.json"

    print(f"[config] data_root={data_root}", flush=True)
    print(f"[config] MODELSCOPE_CACHE={os.environ.get('MODELSCOPE_CACHE')}", flush=True)
    print(f"[config] HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')}", flush=True)
    print(f"[config] HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}", flush=True)
    print(f"[config] selected={[spec.name for spec in selected]}", flush=True)

    results: list[dict[str, Any]] = []
    for spec in selected:
        print(f"\n=== {spec.name}: {spec.purpose} ===", flush=True)
        result = download_one(
            spec=spec,
            data_root=data_root,
            prefer_modelscope=not args.no_modelscope,
            allow_hf_fallback=not args.no_hf_fallback,
            revision=args.revision,
        )
        results.append(result)
        write_json(manifest_path, {"datasets": results})
        print(f"[result] {spec.name}: {result['status']} via {result.get('selected_backend')}", flush=True)

    failed = [item["name"] for item in results if item.get("status") != "ok"]
    write_json(manifest_path, {"datasets": results, "failed": failed})
    if failed:
        raise SystemExit(f"Failed datasets: {failed}")
    print(f"\n[ok] downloaded {len(results)} datasets. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
