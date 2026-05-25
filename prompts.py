"""
prompts.py — Prompt Engineering Registry
=========================================
All prompts used in the PDF RAG pipeline are defined and documented here.
Each prompt includes:
  - Purpose: what it does
  - Technique: which prompt engineering technique is applied
  - Input variables: what placeholders it expects
  - Design notes: why it is structured this way

This file serves as the Prompt Engineering submission for the GenAI course project.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# 1. ANSWER GENERATION PROMPT
# ---------------------------------------------------------------------------
# Technique: Role prompting + Grounded generation + Citation enforcement
# Purpose:   Make the LLM answer ONLY from provided context and cite pages.
# Design:    - Explicit role ("university assistant") sets expected tone.
#            - "ONLY" and "only using" are hard constraints against hallucination.
#            - Citation format [p.X] is enforced in the instruction, and the
#              context is pre-formatted with [Source N | page X] labels so the
#              model can easily match source to page number.
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful university assistant answering questions about a PDF. "
    "Use ONLY the provided context. If the context does not contain the answer, "
    "say so clearly. Cite supporting pages inline using the format [p.X] "
    "(use the page number from the matching Source block). "
    "If multiple pages support a claim, cite all of them, e.g. [p.2][p.5]."
)

ANSWER_USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"


# ---------------------------------------------------------------------------
# 2. CONVERSATIONAL QUERY REWRITING PROMPT
# ---------------------------------------------------------------------------
# Technique: Chain-of-thought lite + Contextual grounding
# Purpose:   Resolve pronouns and follow-up references so retrieval works
#            on standalone queries without conversation context.
# Design:    - Providing the full recent chat history gives the model context
#              to resolve "it", "that", "the second point", etc.
#            - "ONLY the rewritten query" prevents preamble and explanation,
#              which would break downstream embedding.
# ---------------------------------------------------------------------------

REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's latest question into a single self-contained "
    "search query that captures all referenced context from the chat. "
    "Output ONLY the rewritten query, no preamble."
)

REWRITE_USER_TEMPLATE = "Chat so far:\n{chat_history}\n\nLatest question: {question}"


# ---------------------------------------------------------------------------
# 3. HYDE — HYPOTHETICAL DOCUMENT EMBEDDINGS PROMPT
# ---------------------------------------------------------------------------
# Technique: HyDE (Gao et al., 2022) — generate a hypothetical answer and
#            embed it instead of the question for retrieval.
# Purpose:   Bridge the embedding-space gap between questions and document
#            passages. Questions and answers live in different vector regions;
#            a hypothetical answer is closer to real document text.
# Design:    - "as if it were extracted from a document" steers the model to
#              produce passage-style text rather than conversational text.
#            - temperature=0.5 adds slight variation to avoid overfitting to
#              one phrasing of the hypothetical document.
# ---------------------------------------------------------------------------

HYDE_SYSTEM_PROMPT = (
    "Write a short factual passage (2-4 sentences) that would directly answer "
    "the question, as if extracted from a relevant document. "
    "Output ONLY the passage, no preamble."
)

HYDE_USER_TEMPLATE = "{question}"


# ---------------------------------------------------------------------------
# 4. MULTI-QUERY GENERATION PROMPT
# ---------------------------------------------------------------------------
# Technique: Multi-query retrieval (RAG-Fusion style)
# Purpose:   Generate multiple phrasings of the same question to improve
#            retrieval recall across different vocabulary choices.
# Design:    - "slightly different angle" encourages genuine semantic diversity
#              rather than trivial rephrasing.
#            - Strict output format ("exactly 3 lines") ensures clean parsing.
#            - temperature=0.7 promotes creative rephrasing diversity.
# ---------------------------------------------------------------------------

MULTI_QUERY_SYSTEM_PROMPT = (
    "Generate 3 different phrasings of the user's question to improve document retrieval. "
    "Each phrasing should approach the topic from a slightly different angle. "
    "Output exactly 3 lines, one phrasing per line, no numbering or extra text."
)

MULTI_QUERY_USER_TEMPLATE = "{question}"


# ---------------------------------------------------------------------------
# 5. CONTEXTUAL COMPRESSION PROMPT
# ---------------------------------------------------------------------------
# Technique: Contextual compression (Press et al., 2023)
# Purpose:   Strip irrelevant sentences from retrieved chunks before sending
#            to the LLM, reducing noise and context window usage.
# Design:    - "ONLY the sentences" enforces extraction-only behavior.
#            - "IRRELEVANT" sentinel provides a clean signal for filtering.
#            - temperature=0.0 ensures deterministic extraction.
# ---------------------------------------------------------------------------

COMPRESSION_SYSTEM_PROMPT = (
    "Extract ONLY the sentences from the passage that directly help answer "
    "the question. If nothing is relevant, output exactly 'IRRELEVANT'. "
    "Output only the extracted text, no preamble."
)

COMPRESSION_USER_TEMPLATE = "Question: {question}\n\nPassage:\n{passage}"


# ---------------------------------------------------------------------------
# PROMPT SUMMARY TABLE (for paper / report)
# ---------------------------------------------------------------------------

PROMPT_REGISTRY: list[dict] = [
    {
        "id": 1,
        "name": "Answer Generation",
        "technique": "Role prompting + Grounded generation + Citation enforcement",
        "temperature": 0.1,
        "system_prompt": ANSWER_SYSTEM_PROMPT,
        "user_template": ANSWER_USER_TEMPLATE,
        "variables": ["context", "question"],
    },
    {
        "id": 2,
        "name": "Query Rewriting",
        "technique": "Contextual grounding + Output constraint",
        "temperature": 0.0,
        "system_prompt": REWRITE_SYSTEM_PROMPT,
        "user_template": REWRITE_USER_TEMPLATE,
        "variables": ["chat_history", "question"],
    },
    {
        "id": 3,
        "name": "HyDE Generation",
        "technique": "Hypothetical Document Embeddings (Gao et al., 2022)",
        "temperature": 0.5,
        "system_prompt": HYDE_SYSTEM_PROMPT,
        "user_template": HYDE_USER_TEMPLATE,
        "variables": ["question"],
    },
    {
        "id": 4,
        "name": "Multi-Query Generation",
        "technique": "RAG-Fusion multi-query + diversity sampling",
        "temperature": 0.7,
        "system_prompt": MULTI_QUERY_SYSTEM_PROMPT,
        "user_template": MULTI_QUERY_USER_TEMPLATE,
        "variables": ["question"],
    },
    {
        "id": 5,
        "name": "Contextual Compression",
        "technique": "Sentence extraction + irrelevance filtering",
        "temperature": 0.0,
        "system_prompt": COMPRESSION_SYSTEM_PROMPT,
        "user_template": COMPRESSION_USER_TEMPLATE,
        "variables": ["question", "passage"],
    },
]
