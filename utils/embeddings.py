from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def get_embeddings():
    """Return an embeddings model.

    Preference order:
    1. OpenAI embeddings if OPENAI_API_KEY is set.
    2. HuggingFace sentence-transformers (local) as fallback.
       Note: Groq has no embeddings API — GROQ_API_KEY alone will use local embeddings.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception as exc:
            logger.warning("OpenAI embeddings failed, falling back to local: %s", exc)

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize embeddings. Install either `langchain-openai` with OPENAI_API_KEY "
            "or `langchain-community` plus `sentence-transformers`."
        ) from exc
