from __future__ import annotations

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into manageable chunks for embedding and retrieval.

    Defaults: 900-char chunks with 150-char overlap (~one paragraph each).
    Override via RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP env vars.
    """
    size = chunk_size if chunk_size is not None else _env_int("RAG_CHUNK_SIZE", 900)
    overlap = chunk_overlap if chunk_overlap is not None else _env_int("RAG_CHUNK_OVERLAP", 150)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
