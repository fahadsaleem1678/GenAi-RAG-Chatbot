from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _get_ragas_llm():
    """Return a LangChain LLM wrapped for RAGAS, preferring OpenAI → xAI → Groq."""
    try:
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI
        if os.getenv("OPENAI_API_KEY"):
            return LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0.0))
        if os.getenv("XAI_API_KEY"):
            return LangchainLLMWrapper(ChatOpenAI(
                model="grok-3-mini",
                temperature=0.0,
                base_url="https://api.x.ai/v1",
                api_key=os.getenv("XAI_API_KEY"),
            ))
        if os.getenv("GROQ_API_KEY"):
            from langchain_groq import ChatGroq
            return LangchainLLMWrapper(ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0))
    except Exception as exc:
        logger.warning("Could not build RAGAS LLM wrapper: %s", exc)
    return None


def run_comparative_eval(
    test_cases: list[dict],
    pipeline,
) -> list[dict]:
    """Compare three retrieval strategies side-by-side on the same questions.

    Strategies:
      Baseline  — naive FAISS top-k=4 (no hybrid, no rerank)
      Hybrid    — FAISS + BM25 + rerank (no HyDE / multi-query)
      Optimised — full pipeline (HyDE + multi-query + hybrid + rerank + compression)

    Returns a list of {question, strategies: {name: {answer, num_chunks}}} dicts.
    """
    from utils.vector_store import VectorStoreManager
    from utils.retriever import HybridRetriever, Reranker
    from langchain_core.documents import Document

    results = []

    for case in test_cases:
        q = case["question"]
        entry: dict = {"question": q, "strategies": {}}

        # ── Strategy 1: Baseline (raw FAISS top-k=4) ────────────────────────
        try:
            scored = pipeline.vector_store.similarity_search_with_score(q, k=4)
            docs_b = [d for d, _ in scored]
            answer_b = pipeline._generate_answer(q, docs_b, [])
            entry["strategies"]["Baseline"] = {
                "answer": answer_b,
                "num_chunks": len(docs_b),
            }
        except Exception as exc:
            logger.warning("Baseline strategy failed: %s", exc)
            entry["strategies"]["Baseline"] = {"answer": f"Error: {exc}", "num_chunks": 0}

        # ── Strategy 2: Hybrid + rerank only (no HyDE/multi-query) ──────────
        try:
            scored_h = pipeline.retriever.retrieve(q)
            docs_h = [d for d, _ in scored_h]
            answer_h = pipeline._generate_answer(q, docs_h, [])
            entry["strategies"]["Hybrid + Rerank"] = {
                "answer": answer_h,
                "num_chunks": len(docs_h),
            }
        except Exception as exc:
            logger.warning("Hybrid strategy failed: %s", exc)
            entry["strategies"]["Hybrid + Rerank"] = {"answer": f"Error: {exc}", "num_chunks": 0}

        # ── Strategy 3: Full optimised pipeline ─────────────────────────────
        try:
            full_result = pipeline.ask(q, history=[])
            entry["strategies"]["Optimised (Full)"] = {
                "answer": full_result["answer"],
                "num_chunks": len(full_result["retrieved_chunks"]),
            }
        except Exception as exc:
            logger.warning("Optimised strategy failed: %s", exc)
            entry["strategies"]["Optimised (Full)"] = {"answer": f"Error: {exc}", "num_chunks": 0}

        results.append(entry)

    return results


def run_ragas_eval(
    test_cases: list[dict],
    pipeline,
    history: Optional[list] = None,
) -> Optional[dict]:
    """Run RAGAS evaluation on a list of test questions.

    Each test_case: {"question": str, "ground_truth": str (optional)}
    Requires OPENAI_API_KEY or GROQ_API_KEY. Returns {metric: score} dict or None on failure.
    """
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY")):
        return None

    try:
        from datasets import Dataset
        from ragas import evaluate
        try:
            from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision
        except ImportError:
            from ragas.metrics import faithfulness, answer_relevancy, context_precision
    except ImportError as exc:
        logger.warning("RAGAS or datasets not installed: %s", exc)
        return None

    questions, answers, contexts, ground_truths = [], [], [], []
    has_ground_truth = all("ground_truth" in c for c in test_cases)

    for case in test_cases:
        q = case["question"]
        try:
            result = pipeline.ask(q, history=history or [])
            a = result["answer"]
            ctx = [c["content"] for c in result["retrieved_chunks"]]
        except Exception as exc:
            logger.warning("Pipeline failed for '%s': %s", q, exc)
            continue

        questions.append(q)
        answers.append(a)
        contexts.append(ctx)
        if has_ground_truth:
            ground_truths.append(case["ground_truth"])

    if not questions:
        return None

    data: dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if has_ground_truth and len(ground_truths) == len(questions):
        data["ground_truth"] = ground_truths
        metrics = [faithfulness, answer_relevancy, context_precision]
    else:
        metrics = [faithfulness, answer_relevancy]

    try:
        dataset = Dataset.from_dict(data)
        ragas_llm = _get_ragas_llm()
        kwargs = {"llm": ragas_llm} if ragas_llm and not os.getenv("OPENAI_API_KEY") else {}
        result = evaluate(dataset, metrics=metrics, **kwargs)
        return {k: round(float(v), 4) for k, v in result.items() if isinstance(v, (int, float))}
    except Exception as exc:
        logger.warning("RAGAS evaluate() failed: %s", exc)
        return None
