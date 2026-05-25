from __future__ import annotations

import hashlib
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prompts import (
    ANSWER_SYSTEM_PROMPT,
    COMPRESSION_SYSTEM_PROMPT,
    COMPRESSION_USER_TEMPLATE,
    HYDE_SYSTEM_PROMPT,
    MULTI_QUERY_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    REWRITE_USER_TEMPLATE,
)
from utils.chunking import split_documents
from utils.embeddings import get_embeddings
from utils.pdf_loader import load_pdf
from utils.retriever import HybridRetriever, Reranker
from utils.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".rag_cache")
HISTORY_TURNS = 6


class RAGPipeline:
    """Core RAG pipeline: PDF ingestion → hybrid retrieval → HyDE/multi-query → rerank → compress → answer."""

    def __init__(self) -> None:
        self.embeddings = None
        self.vector_store: Optional[VectorStoreManager] = None
        self.retriever: Optional[HybridRetriever] = None
        self.reranker = Reranker()
        self.source_documents: list[Document] = []
        self.chunks: list[Document] = []
        self.pdf_name = ""

    def reset(self) -> None:
        self.vector_store = None
        self.retriever = None
        self.source_documents = []
        self.chunks = []
        self.pdf_name = ""

    def _llm_available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY"))

    def _get_llm(self, temperature: float = 0.0):
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
        if os.getenv("XAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="grok-3-mini",
                temperature=temperature,
                base_url="https://api.x.ai/v1",
                api_key=os.getenv("XAI_API_KEY"),
            )
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)

    def _cache_path(self, file_bytes: bytes) -> Path:
        digest = hashlib.sha1(file_bytes).hexdigest()
        return CACHE_DIR / digest

    def ingest_pdf(self, uploaded_file) -> dict[str, int]:
        """Read a PDF, chunk it, build (or load) a FAISS index, set up retriever."""
        file_bytes = uploaded_file.getvalue()
        if not file_bytes:
            raise ValueError("Uploaded PDF is empty.")

        if self.embeddings is None:
            self.embeddings = get_embeddings()

        self.pdf_name = uploaded_file.name
        self.source_documents = load_pdf(BytesIO(file_bytes), source_name=self.pdf_name)
        if not self.source_documents:
            raise ValueError("No readable text found in the PDF.")

        self.chunks = split_documents(self.source_documents)
        if not self.chunks:
            raise ValueError("PDF text could not be split into chunks.")

        cache_path = self._cache_path(file_bytes)
        cache_hit = False
        if cache_path.exists():
            try:
                self.vector_store = VectorStoreManager.load(cache_path, self.embeddings)
                cache_hit = True
                logger.info("Loaded FAISS index from cache: %s", cache_path)
            except Exception as exc:
                logger.warning("Cache load failed (%s); rebuilding.", exc)

        if not cache_hit:
            self.vector_store = VectorStoreManager.from_documents(self.chunks, self.embeddings)
            try:
                self.vector_store.save(cache_path)
            except Exception as exc:
                logger.warning("Could not persist FAISS index: %s", exc)

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            documents=self.chunks,
            reranker=self.reranker,
        )

        return {
            "num_chunks": len(self.chunks),
            "num_pages": len(self.source_documents),
            "cache_hit": int(cache_hit),
        }

    # ------------------------------------------------------------------ #
    # Query enhancement                                                    #
    # ------------------------------------------------------------------ #

    def _rewrite_query(self, question: str, history: list[dict]) -> str:
        """Rewrite a follow-up question into a standalone query using chat history."""
        if not history or not self._llm_available():
            return question
        try:
            recent = history[-HISTORY_TURNS:]
            convo = "\n".join(f"{m['role'].title()}: {m['content']}" for m in recent)
            llm = self._get_llm(temperature=0.0)
            messages = [
                SystemMessage(content=REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=REWRITE_USER_TEMPLATE.format(chat_history=convo, question=question)),
            ]
            response = llm.invoke(messages)
            return getattr(response, "content", "").strip() or question
        except Exception as exc:
            logger.warning("Query rewriting failed: %s", exc)
            return question

    def _hyde_query(self, question: str) -> Optional[str]:
        """Generate a hypothetical document for HyDE embedding."""
        if not self._llm_available():
            return None
        try:
            llm = self._get_llm(temperature=0.5)
            messages = [
                SystemMessage(content=HYDE_SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
            response = llm.invoke(messages)
            return getattr(response, "content", "").strip() or None
        except Exception as exc:
            logger.warning("HyDE generation failed: %s", exc)
            return None

    def _multi_queries(self, question: str) -> list[str]:
        """Generate 3 rephrasings of the question for multi-query retrieval."""
        if not self._llm_available():
            return [question]
        try:
            llm = self._get_llm(temperature=0.7)
            messages = [
                SystemMessage(content=MULTI_QUERY_SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
            response = llm.invoke(messages)
            content = getattr(response, "content", "").strip()
            variants = [l.strip() for l in content.split("\n") if l.strip()][:3]
            return [question] + variants if variants else [question]
        except Exception as exc:
            logger.warning("Multi-query generation failed: %s", exc)
            return [question]

    def _compress_docs(self, question: str, docs: list[Document]) -> list[Document]:
        """Extract only the sentences from each chunk that are relevant to the question."""
        if not docs or not self._llm_available():
            return docs
        try:
            llm = self._get_llm(temperature=0.0)
            compressed: list[Document] = []
            for doc in docs:
                messages = [
                    SystemMessage(content=COMPRESSION_SYSTEM_PROMPT),
                    HumanMessage(content=COMPRESSION_USER_TEMPLATE.format(question=question, passage=doc.page_content)),
                ]
                response = llm.invoke(messages)
                extracted = getattr(response, "content", "").strip()
                if extracted and extracted.upper() != "IRRELEVANT":
                    compressed.append(Document(page_content=extracted, metadata=doc.metadata))
            return compressed if compressed else docs
        except Exception as exc:
            logger.warning("Contextual compression failed: %s", exc)
            return docs

    # ------------------------------------------------------------------ #
    # Main ask                                                             #
    # ------------------------------------------------------------------ #

    def ask(self, question: str, history: Optional[list[dict]] = None) -> dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
        if self.retriever is None:
            raise ValueError("Please upload a PDF before asking questions.")

        history = history or []

        # Step 1: rewrite for conversational coherence
        search_query = self._rewrite_query(question, history)

        # Step 2: HyDE (hypothetical doc for vector search) + multi-query
        hyde_doc = self._hyde_query(search_query)
        queries = self._multi_queries(search_query)

        # Step 3: retrieve (multi-query with HyDE for vector branch)
        if len(queries) > 1:
            hyde_docs = [hyde_doc] + [None] * (len(queries) - 1) if hyde_doc else None
            scored = self.retriever.retrieve_multi(queries, vector_queries=hyde_docs)
        else:
            scored = self.retriever.retrieve(search_query, vector_query=hyde_doc)

        raw_docs = [d for d, _ in scored]

        # Step 4: contextual compression
        docs = self._compress_docs(question, raw_docs)

        retrieved_chunks = []
        for (raw_doc, score), compressed_doc in zip(scored, docs):
            retrieved_chunks.append({
                "content": compressed_doc.page_content,
                "original_content": raw_doc.page_content,
                "source": compressed_doc.metadata.get("source", self.pdf_name or "PDF"),
                "page": compressed_doc.metadata.get("page"),
                "metadata": compressed_doc.metadata,
                "score": float(score),
            })

        answer = self._generate_answer(question, docs, history)
        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "search_query": search_query,
            "hyde_used": hyde_doc is not None,
            "multi_query_used": len(queries) > 1,
        }

    # ------------------------------------------------------------------ #
    # Answer generation                                                    #
    # ------------------------------------------------------------------ #

    def _generate_answer(self, question: str, docs: list[Document], history: list[dict]) -> str:
        context_blocks = []
        for idx, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page", "?")
            context_blocks.append(f"[Source {idx} | page {page}]\n{doc.page_content}")
        context = "\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"

        if not self._llm_available():
            if not docs:
                return "I could not find any relevant information in the uploaded PDF."
            first_passage = docs[0].page_content.strip().replace("\n", " ")
            if len(first_passage) > 220:
                first_passage = first_passage[:220].rstrip() + "..."
            return (
                "I found relevant passages from the PDF, but no LLM key is configured for a full generated answer.\n\n"
                f"Most relevant passage preview:\n{first_passage}"
            )

        try:
            llm = self._get_llm(temperature=0.1)
            messages: list = [
                SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            ]
            for m in history[-HISTORY_TURNS:]:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    messages.append(AIMessage(content=m["content"]))
            messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"))
            response = llm.invoke(messages)
            return getattr(response, "content", str(response))
        except Exception as exc:
            logger.warning("LLM generation failed: %s", exc)
            if not docs:
                return "I could not find any relevant information in the uploaded PDF."
            first_passage = docs[0].page_content.strip().replace("\n", " ")
            if len(first_passage) > 220:
                first_passage = first_passage[:220].rstrip() + "..."
            return f"LLM generation failed; showing the most relevant passage instead.\n\n{first_passage}"
