"""Hybrid RAG retrieval tool for code-agent rollout.

Retrieval order:
1. BM25 sparse retrieval from ``bm25.pkl``.
2. Dense retrieval from bge-m3 + Milvus when the index is available.
3. Reciprocal-rank fusion over sparse and dense candidates.

If dense/Milvus is not ready yet, the tool degrades to BM25. If BM25 is also
missing, it falls back to a lightweight corpus keyword scan.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CORPUS = Path("/root/autodl-tmp/datasets/processed/rag_final/corpus.jsonl")
DEFAULT_INDEX_DIR = Path("/root/autodl-tmp/datasets/processed/rag_final/index")
DEFAULT_MODEL = Path("/root/autodl-tmp/models/bge-m3")
DEFAULT_MILVUS_URI = str(DEFAULT_INDEX_DIR / "milvus_lite.db")
DEFAULT_COLLECTION = "code_rag_bge_m3"


def tokenize(text: str) -> list[str]:
    """Code-aware tokenizer kept compatible with build_hybrid_rag_index.py."""
    return re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|"
        r"[A-Za-z0-9_.+\-/#]{2,}|"
        r"[\u4e00-\u9fff]+",
        (text or "").lower(),
    )


def _token_set(text: str) -> set[str]:
    return set(tokenize(text))


def _is_metadata_sidecar(path: str) -> bool:
    return any(part.startswith("._") for part in Path(path).parts)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return rows
    return rows


@dataclass
class RetrievalDoc:
    pk: int
    doc_id: str
    text: str
    doc_type: str = ""
    source_dataset: str = ""
    source_id: str = ""
    metadata: dict[str, Any] | None = None


def _normalize_doc(row: dict[str, Any], fallback_pk: int) -> RetrievalDoc:
    return RetrievalDoc(
        pk=int(row.get("pk", row.get("id", fallback_pk)) or fallback_pk),
        doc_id=str(row.get("doc_id") or f"doc_{fallback_pk}"),
        text=str(row.get("text") or ""),
        doc_type=str(row.get("doc_type") or ""),
        source_dataset=str(row.get("source_dataset") or ""),
        source_id=str(row.get("source_id") or ""),
        metadata=row.get("metadata") or {},
    )


class HybridRAGRetriever:
    def __init__(self) -> None:
        self.index_dir = Path(os.environ.get("CODE_AGENT_RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
        self.corpus_path = Path(os.environ.get("CODE_AGENT_RAG_CORPUS", str(DEFAULT_CORPUS)))
        self.model_name = os.environ.get("CODE_AGENT_BGE_MODEL", str(DEFAULT_MODEL))
        self.device = os.environ.get("CODE_AGENT_BGE_DEVICE", "")
        self.use_fp16 = os.environ.get("CODE_AGENT_BGE_USE_FP16", "0") == "1"
        self.max_length = int(os.environ.get("CODE_AGENT_BGE_MAX_LENGTH", "4096"))
        self.milvus_uri = os.environ.get("CODE_AGENT_MILVUS_URI", DEFAULT_MILVUS_URI)
        self.milvus_token = os.environ.get("CODE_AGENT_MILVUS_TOKEN") or None
        self.milvus_db = os.environ.get("CODE_AGENT_MILVUS_DB") or None
        self.collection = os.environ.get("CODE_AGENT_MILVUS_COLLECTION", DEFAULT_COLLECTION)

        self.docs = self._load_docstore()
        self.docs_by_pk = {doc.pk: doc for doc in self.docs}
        self.bm25 = None
        self.bm25_ready = False
        self.dense_ready = False
        self._dense_model = None
        self._milvus_client = None
        self._load_bm25()

    def _load_docstore(self) -> list[RetrievalDoc]:
        rows = _read_jsonl(self.index_dir / "docstore.jsonl")
        if not rows:
            rows = _read_jsonl(self.corpus_path)
        docs: list[RetrievalDoc] = []
        max_docs = int(os.environ.get("CODE_AGENT_RAG_MAX_CONTEXT_DOCS", "20000"))
        for idx, row in enumerate(rows):
            text = str(row.get("text") or "").strip()
            if text:
                docs.append(_normalize_doc(row, len(docs)))
                continue

            contexts = row.get("contexts")
            if not isinstance(contexts, list):
                continue
            for context in contexts:
                if not isinstance(context, dict):
                    continue
                context_text = str(context.get("text") or "").strip()
                if not context_text:
                    continue
                path = str(context.get("path") or "")
                if _is_metadata_sidecar(path):
                    continue
                start_line = context.get("start_line")
                end_line = context.get("end_line")
                location = path
                if start_line is not None and end_line is not None:
                    location = f"{path}:{start_line}-{end_line}"
                docs.append(
                    RetrievalDoc(
                        pk=len(docs),
                        doc_id=f"{row.get('instance_id', idx)}:{len(docs)}",
                        text=f"{location}\n{context_text}" if location else context_text,
                        doc_type="context",
                        source_dataset=str(row.get("repo") or ""),
                        source_id=str(row.get("instance_id") or ""),
                        metadata={
                            "repo": row.get("repo"),
                            "base_commit": row.get("base_commit"),
                            "path": path,
                            "start_line": start_line,
                            "end_line": end_line,
                        },
                    )
                )
                if len(docs) >= max_docs:
                    return docs
        return docs

    def _load_bm25(self) -> None:
        bm25_path = self.index_dir / "bm25.pkl"
        try:
            if not bm25_path.exists():
                return
            with bm25_path.open("rb") as f:
                payload = pickle.load(f)
            self.bm25 = payload["bm25"] if isinstance(payload, dict) else payload
            self.bm25_ready = True
        except Exception:
            self.bm25 = None
            self.bm25_ready = False

    def _load_dense(self) -> bool:
        if self.dense_ready:
            return True
        if os.environ.get("CODE_AGENT_DISABLE_DENSE_RAG", "0") == "1":
            return False
        try:
            from FlagEmbedding import BGEM3FlagModel
            from pymilvus import MilvusClient
        except Exception:
            return False

        try:
            kwargs: dict[str, Any] = {"use_fp16": self.use_fp16}
            if self.device:
                kwargs["devices"] = [self.device]
            self._dense_model = BGEM3FlagModel(self.model_name, **kwargs)

            client_kwargs: dict[str, Any] = {"uri": self.milvus_uri}
            if self.milvus_token:
                client_kwargs["token"] = self.milvus_token
            if self.milvus_db:
                client_kwargs["db_name"] = self.milvus_db
            self._milvus_client = MilvusClient(**client_kwargs)
            if not self._milvus_client.has_collection(self.collection):
                return False
            self.dense_ready = True
            return True
        except Exception:
            self._dense_model = None
            self._milvus_client = None
            self.dense_ready = False
            return False

    def _encode_query(self, query: str) -> list[float] | None:
        if not self._load_dense() or self._dense_model is None:
            return None
        output = self._dense_model.encode(
            [query],
            batch_size=1,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vec = np.asarray(output["dense_vecs"][0], dtype="float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def search_bm25(self, query: str, limit: int) -> list[tuple[int, float]]:
        if self.bm25_ready and self.bm25 is not None and self.docs:
            scores = self.bm25.get_scores(tokenize(query))
            top_idx = np.argsort(scores)[::-1][:limit]
            return [(self.docs[int(idx)].pk, float(scores[int(idx)])) for idx in top_idx if scores[int(idx)] > 0]

        query_terms = _token_set(query)
        scored: list[tuple[int, float]] = []
        for doc in self.docs:
            overlap = len(query_terms & _token_set(doc.text))
            if overlap > 0:
                scored.append((doc.pk, overlap / max(1, len(query_terms))))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def search_dense(self, query: str, limit: int) -> list[tuple[int, float]]:
        vec = self._encode_query(query)
        if vec is None or self._milvus_client is None:
            return []
        try:
            results = self._milvus_client.search(
                collection_name=self.collection,
                data=[vec],
                limit=limit,
                output_fields=["doc_id", "doc_type", "source_dataset", "source_id", "text", "metadata"],
            )
        except Exception:
            return []

        hits: list[tuple[int, float]] = []
        for hit in results[0] if results else []:
            pk = hit.get("id")
            score = float(hit.get("distance", 0.0) or 0.0)
            try:
                hits.append((int(pk), score))
            except Exception:
                continue
        return hits

    def _filter_hits_by_source(
        self,
        hits: list[tuple[int, float]],
        source_dataset: str | None,
    ) -> list[tuple[int, float]]:
        if not source_dataset:
            return hits
        return [
            (pk, score)
            for pk, score in hits
            if (self.docs_by_pk.get(pk) is not None and self.docs_by_pk[pk].source_dataset == source_dataset)
        ]

    def hybrid_search(self, query: str, top_k: int, source_dataset: str | None = None) -> list[dict[str, Any]]:
        candidate_k = max(top_k * (20 if source_dataset else 4), 200 if source_dataset else 20)
        sparse = self.search_bm25(query, candidate_k)
        dense = self.search_dense(query, candidate_k)
        sparse = self._filter_hits_by_source(sparse, source_dataset)
        dense = self._filter_hits_by_source(dense, source_dataset)

        fused: dict[int, dict[str, float]] = {}
        for rank, (pk, score) in enumerate(sparse, 1):
            item = fused.setdefault(pk, {"score": 0.0, "bm25": 0.0, "dense": 0.0})
            item["score"] += 1.0 / (60.0 + rank)
            item["bm25"] = score
        for rank, (pk, score) in enumerate(dense, 1):
            item = fused.setdefault(pk, {"score": 0.0, "bm25": 0.0, "dense": 0.0})
            item["score"] += 1.0 / (60.0 + rank)
            item["dense"] = score

        ranked = sorted(fused.items(), key=lambda item: item[1]["score"], reverse=True)[:top_k]
        output: list[dict[str, Any]] = []
        for pk, scores in ranked:
            doc = self.docs_by_pk.get(pk)
            if doc is None:
                continue
            output.append(
                {
                    "pk": pk,
                    "doc_id": doc.doc_id,
                    "doc_type": doc.doc_type,
                    "source_dataset": doc.source_dataset,
                    "source_id": doc.source_id,
                    "metadata": doc.metadata or {},
                    "text": doc.text,
                    "score": scores["score"],
                    "bm25_score": scores["bm25"],
                    "dense_score": scores["dense"],
                    "retrieval_mode": "hybrid" if dense else "bm25",
                }
            )
        return output


@lru_cache(maxsize=1)
def _retriever() -> HybridRAGRetriever:
    return HybridRAGRetriever()


def run_search(query: str, top_k: int | None = None, source_dataset: str | None = None) -> str:
    """Return top RAG snippets for a search query."""
    query = "" if query is None else str(query).strip()
    if not query:
        return "No query provided."

    k = top_k or int(os.environ.get("CODE_AGENT_SEARCH_TOP_K", "3"))
    docs = _retriever().hybrid_search(query, k, source_dataset=source_dataset)
    if not docs:
        suffix = f" for repo {source_dataset}" if source_dataset else ""
        return f"No relevant RAG document found{suffix}."

    max_chars = int(os.environ.get("CODE_AGENT_SEARCH_DOC_CHARS", "1200"))
    snippets: list[str] = []
    for rank, doc in enumerate(docs, 1):
        text = str(doc.get("text") or "")
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]..."
        snippets.append(
            f"[{rank}] mode={doc.get('retrieval_mode')} fused={doc.get('score', 0):.4f} "
            f"bm25={doc.get('bm25_score', 0):.4f} dense={doc.get('dense_score', 0):.4f} "
            f"dataset={doc.get('source_dataset')} type={doc.get('doc_type')} id={doc.get('doc_id')}\n{text}"
        )
    return "\n\n".join(snippets)
