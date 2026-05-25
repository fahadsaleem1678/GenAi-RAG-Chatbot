from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from utils.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


def rrf_merge(results_list: list[list[Document]], k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion across multiple result lists, deduplicating by content."""
    scores: dict[str, dict] = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            key = doc.page_content[:150]
            if key not in scores:
                scores[key] = {"doc": doc, "score": 0.0}
            scores[key]["score"] += 1.0 / (rank + k)
    return [item["doc"] for item in sorted(scores.values(), key=lambda x: x["score"], reverse=True)]


class Reranker:
    """Cross-encoder reranker. Lazy-loaded; falls back to identity if unavailable."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._unavailable = False

    def _load(self):
        if self._model is not None or self._unavailable:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        except Exception as exc:
            logger.warning("Cross-encoder unavailable, skipping rerank: %s", exc)
            self._unavailable = True
        return self._model

    def rerank(self, query: str, docs: list[Document], k: int) -> list[tuple[Document, float]]:
        if not docs:
            return []
        model = self._load()
        if model is None:
            return [(d, 0.0) for d in docs[:k]]
        pairs = [(query, d.page_content) for d in docs]
        try:
            scores = model.predict(pairs)
        except Exception as exc:
            logger.warning("Rerank failed, falling back to original order: %s", exc)
            return [(d, 0.0) for d in docs[:k]]
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return [(d, float(s)) for d, s in ranked[:k]]


class HybridRetriever:
    """Vector + BM25 hybrid with RRF merging and optional cross-encoder reranking.

    Supports HyDE (separate vector_query) and multi-query retrieval.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        documents: list[Document],
        k_initial: Optional[int] = None,
        k_final: Optional[int] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self.vector_store = vector_store
        self.k_initial = k_initial if k_initial is not None else _env_int("RAG_K_INITIAL", 10)
        self.k_final = k_final if k_final is not None else _env_int("RAG_K_FINAL", 4)
        self.reranker = reranker

        self._bm25 = BM25Retriever.from_documents(documents)
        self._bm25.k = self.k_initial

    def _vector_search(self, query: str) -> list[Document]:
        return self.vector_store.vector_store.similarity_search(query, k=self.k_initial)

    def _bm25_search(self, query: str) -> list[Document]:
        return self._bm25.invoke(query)

    def retrieve(
        self,
        query: str,
        vector_query: Optional[str] = None,
    ) -> list[tuple[Document, float]]:
        """Single-query retrieval. vector_query overrides what goes to FAISS (used for HyDE)."""
        vq = vector_query or query
        merged = rrf_merge([self._vector_search(vq), self._bm25_search(query)])
        if self.reranker:
            return self.reranker.rerank(query, merged, self.k_final)
        return [(d, 0.0) for d in merged[: self.k_final]]

    def retrieve_multi(
        self,
        queries: list[str],
        vector_queries: Optional[list[str]] = None,
    ) -> list[tuple[Document, float]]:
        """Multi-query retrieval: RRF across all (query, vector_query) pairs, then rerank."""
        all_results: list[list[Document]] = []
        for i, q in enumerate(queries):
            vq = (vector_queries[i] if vector_queries and i < len(vector_queries) else None) or q
            all_results.append(self._vector_search(vq))
            all_results.append(self._bm25_search(q))
        merged = rrf_merge(all_results)
        original = queries[0]
        if self.reranker:
            return self.reranker.rerank(original, merged, self.k_final)
        return [(d, 0.0) for d in merged[: self.k_final]]
