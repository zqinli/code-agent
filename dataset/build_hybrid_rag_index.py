#!/usr/bin/env python3
"""Build BM25 + BGE-M3 + Milvus hybrid RAG indexes for code-task corpus."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_CORPUS = Path("/root/autodl-tmp/datasets/processed/rag_final/corpus.jsonl")
DEFAULT_INDEX_DIR = Path("/root/autodl-tmp/datasets/processed/rag_final/index")


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


def tokenize(text: str) -> list[str]:
    # Code-aware tokenizer: preserve identifiers, paths, dotted names, and CJK chunks.
    tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|"
        r"[A-Za-z0-9_.+\-/#]{2,}|"
        r"[\u4e00-\u9fff]+",
        text.lower(),
    )
    return tokens


def load_docs(corpus_path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for idx, row in enumerate(read_jsonl(corpus_path)):
        text = row.get("text") or ""
        if not text.strip():
            continue
        docs.append(
            {
                "pk": idx,
                "doc_id": row.get("doc_id") or f"doc_{idx}",
                "text": text,
                "doc_type": row.get("doc_type"),
                "source_dataset": row.get("source_dataset"),
                "source_id": row.get("source_id"),
                "metadata": row.get("metadata") or {},
            }
        )
    return docs


def build_bm25(docs: list[dict[str, Any]], index_dir: Path) -> None:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError("Missing dependency rank_bm25. Install rank-bm25 first.") from exc

    tokenized = [tokenize(doc["text"]) for doc in docs]
    bm25 = BM25Okapi(tokenized)
    with (index_dir / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "tokenized_corpus": tokenized}, f)


def write_docstore(docs: list[dict[str, Any]], index_dir: Path) -> None:
    with (index_dir / "docstore.jsonl").open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")


def load_bge_model(model_name: str, device: str, use_fp16: bool) -> Any:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise RuntimeError("Missing dependency FlagEmbedding. Install FlagEmbedding first.") from exc
    kwargs: dict[str, Any] = {"use_fp16": use_fp16}
    if device:
        kwargs["devices"] = [device]
    return BGEM3FlagModel(model_name, **kwargs)


def encode_dense(model: Any, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        output = model.encode(
            batch,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"]
        vectors.append(np.asarray(dense, dtype="float32"))
        print(f"Encoded {min(start + len(batch), len(texts))}/{len(texts)} docs")
    arr = np.vstack(vectors)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def connect_milvus(uri: str, token: str | None, db_name: str | None) -> Any:
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError("Missing dependency pymilvus. Install pymilvus first.") from exc
    kwargs: dict[str, Any] = {"uri": uri}
    if token:
        kwargs["token"] = token
    if db_name:
        kwargs["db_name"] = db_name
    return MilvusClient(**kwargs)


def recreate_collection(client: Any, collection_name: str, dim: int, metric_type: str, drop_existing: bool) -> None:
    if client.has_collection(collection_name):
        if not drop_existing:
            raise RuntimeError(
                f"Milvus collection {collection_name!r} already exists. "
                "Pass --drop-existing to rebuild it."
            )
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type=metric_type,
        consistency_level="Strong",
    )


def insert_milvus(client: Any, collection_name: str, docs: list[dict[str, Any]], vectors: np.ndarray, batch_size: int) -> None:
    for start in range(0, len(docs), batch_size):
        batch_docs = docs[start : start + batch_size]
        batch_vectors = vectors[start : start + len(batch_docs)]
        rows = []
        for doc, vector in zip(batch_docs, batch_vectors):
            rows.append(
                {
                    "id": int(doc["pk"]),
                    "vector": vector.tolist(),
                    "doc_id": doc["doc_id"],
                    "doc_type": doc.get("doc_type") or "",
                    "source_dataset": doc.get("source_dataset") or "",
                    "source_id": str(doc.get("source_id") or ""),
                    "text": doc["text"][:16000],
                    "metadata": json.dumps(doc.get("metadata") or {}, ensure_ascii=False),
                }
            )
        client.insert(collection_name=collection_name, data=rows)
        print(f"Inserted {min(start + len(batch_docs), len(docs))}/{len(docs)} docs into Milvus")


def build_indexes(args: argparse.Namespace) -> dict[str, Any]:
    args.index_dir.mkdir(parents=True, exist_ok=True)
    docs = load_docs(args.corpus)
    if not docs:
        raise RuntimeError(f"No documents found in {args.corpus}")

    print(f"Loaded {len(docs)} docs from {args.corpus}")
    docstore_path = args.index_dir / "docstore.jsonl"
    bm25_path = args.index_dir / "bm25.pkl"
    dense_path = args.index_dir / "dense_vectors.npy"

    if args.reuse_existing and docstore_path.exists() and bm25_path.exists():
        print(f"Reusing existing BM25 index and docstore from {args.index_dir}")
    else:
        write_docstore(docs, args.index_dir)
        build_bm25(docs, args.index_dir)
        print(f"Wrote BM25 index and docstore to {args.index_dir}")

    if args.reuse_existing and dense_path.exists():
        vectors = np.load(dense_path)
        if vectors.shape[0] != len(docs):
            raise RuntimeError(
                f"Existing dense vectors have {vectors.shape[0]} rows, "
                f"but corpus has {len(docs)} docs. Rebuild without --reuse-existing."
            )
        print(f"Reusing dense vectors from {dense_path}")
    else:
        model = load_bge_model(args.model_name, args.device, args.use_fp16)
        vectors = encode_dense(model, [doc["text"] for doc in docs], args.encode_batch_size, args.max_length)
        np.save(dense_path, vectors)
        print(f"Wrote dense vectors to {dense_path}")

    client = connect_milvus(args.milvus_uri, args.milvus_token, args.milvus_db)
    recreate_collection(client, args.collection, vectors.shape[1], args.metric_type, args.drop_existing)
    insert_milvus(client, args.collection, docs, vectors, args.insert_batch_size)

    stats = {
        "corpus": str(args.corpus),
        "index_dir": str(args.index_dir),
        "num_docs": len(docs),
        "embedding_model": args.model_name,
        "embedding_dim": int(vectors.shape[1]),
        "milvus_uri": args.milvus_uri,
        "milvus_collection": args.collection,
        "metric_type": args.metric_type,
        "files": {
            "docstore": str(args.index_dir / "docstore.jsonl"),
            "bm25": str(args.index_dir / "bm25.pkl"),
            "dense_vectors": str(dense_path),
        },
    }
    write_json(args.index_dir / "index_stats.json", stats)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--device", default="", help="Optional device for FlagEmbedding, e.g. cuda:0 or cpu.")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--milvus-token", default=None)
    parser.add_argument("--milvus-db", default=None)
    parser.add_argument("--collection", default="code_rag_bge_m3")
    parser.add_argument("--metric-type", choices=["COSINE", "IP", "L2"], default="COSINE")
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--insert-batch-size", type=int, default=256)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing docstore/BM25/dense_vectors files and only rebuild missing parts / Milvus collection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_indexes(args)
    print("Hybrid RAG index build complete")
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
