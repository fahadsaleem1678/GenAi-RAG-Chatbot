# RAG Optimization Upgrades

This document summarizes all optimizations applied to the baseline PDF RAG chatbot.

---

## Baseline Weaknesses

| Issue | Impact |
|---|---|
| 120-char chunks | Too granular — chunks lost semantic meaning |
| Pure FAISS top-k=4 | Single-modality retrieval, missed keyword-heavy queries |
| No reranking | Embedding similarity ≠ relevance; top chunks often off-target |
| No query rewriting | Follow-up questions broke retrieval (unresolved pronouns/references) |
| No source citations | Answers unverifiable, no page grounding |
| No conversation history sent to LLM | Every question answered in isolation |
| Silent `except: pass` in embeddings | Real failures hidden, hard to debug |
| No FAISS persistence | Full re-embedding on every app restart |

---

## Optimizations Applied

### 1. Improved Chunking (`utils/chunking.py`)

- **Chunk size**: 120 chars → **900 chars** (~1 paragraph per chunk)
- **Overlap**: 25 chars → **150 chars** (better context continuity across chunk boundaries)
- Tunable at runtime via env vars `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` without code changes

**Why it matters**: Tiny chunks fragment sentences and lose semantic context. Paragraph-sized chunks give the embedder enough signal to distinguish topics and give the LLM enough context to cite accurately.

---

### 2. Hybrid Retrieval — FAISS + BM25 + RRF (`utils/retriever.py`)

- Replaced pure FAISS vector search with a **dual-channel retriever**:
  - **FAISS** (dense): captures semantic/conceptual similarity
  - **BM25** (sparse): captures exact keyword matches, acronyms, proper nouns
- Results from both channels merged using **Reciprocal Rank Fusion (RRF)** — a rank-weighted merging strategy that promotes documents appearing high in multiple lists
- `k_initial = 10` candidates per channel before reranking; tunable via `RAG_K_INITIAL`

**Why it matters**: Dense retrieval alone fails on exact-match queries ("what is the definition of X?"). BM25 alone fails on semantic queries ("explain the main idea"). Hybrid consistently outperforms either alone.

---

### 3. Cross-Encoder Reranking (`utils/retriever.py`)

- After hybrid retrieval, a **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescores all candidates
- Unlike bi-encoders (used for embedding), cross-encoders jointly encode the query and each document — far more accurate relevance scoring
- Top `k_final = 4` returned after reranking; tunable via `RAG_K_FINAL`
- Lazy-loaded and cached; graceful fallback if model unavailable

**Why it matters**: Bi-encoder similarity scores rank by vector proximity, not true relevance. The cross-encoder acts as a precision filter on the broader recall set from hybrid retrieval.

---

### 4. Conversational Query Rewriting (`rag_pipeline.py`)

- Before retrieval, the latest question is rewritten into a **standalone query** using the last 6 turns of chat history
- Resolves pronouns, implicit references, and follow-up shorthand ("what about the second point?")
- Falls back to original question if no LLM key or rewriting fails

**Why it matters**: "What about its limitations?" is unanswerable without knowing what "it" refers to. Rewriting expands it to a self-contained query before hitting the retriever.

---

### 5. HyDE — Hypothetical Document Embeddings (`rag_pipeline.py`)

- Before vector search, the LLM generates a **hypothetical 2–4 sentence passage** that would answer the question
- That hypothetical passage (not the question) is used as the embedding for FAISS search
- BM25 still uses the original query (HyDE makes no sense for lexical search)

**Why it matters**: Questions and answers live in different embedding spaces. "What causes inflation?" is semantically far from "Inflation is caused by...". A hypothetical answer bridges that gap, pulling in chunks that directly match answer-style text rather than question-style text.

---

### 6. Multi-Query Retrieval + RRF (`rag_pipeline.py` + `utils/retriever.py`)

- The LLM generates **3 rephrasings** of the question (different angles, different vocabulary)
- Hybrid retrieval runs for all 3 variants simultaneously
- All results merged with RRF across all 6 retrieval lists (3 queries × 2 channels)
- Original query used as the anchor for reranking

**Why it matters**: A single phrasing may miss relevant chunks due to vocabulary mismatch. Multiple phrasings dramatically increase recall coverage, and RRF ensures the most consistently retrieved chunks rise to the top.

---

### 7. Contextual Compression (`rag_pipeline.py`)

- After retrieval and reranking, the LLM **extracts only the sentences** from each chunk that are directly relevant to the question
- Chunks where nothing is relevant are filtered out entirely
- Original (pre-compression) text visible in the UI for transparency

**Why it matters**: Retrieved chunks often contain relevant sentences buried in surrounding noise. Sending only the relevant sentences reduces context window usage, lowers hallucination risk, and improves citation accuracy.

---

### 8. Citation-Style Answers with Conversation History (`rag_pipeline.py`)

- System prompt instructs the LLM to cite page numbers inline: `[p.X]`
- Context blocks numbered as `[Source N | page X]` so the LLM can ground its citations
- Last 6 turns of conversation history included in the LLM call for coherent follow-ups

**Why it matters**: Uncited answers are unverifiable. Inline citations let the user trace every claim back to a specific page, critical for academic use.

---

### 9. FAISS Index Persistence (`utils/vector_store.py`, `rag_pipeline.py`)

- PDF bytes SHA-1 hashed; index saved to `.rag_cache/<hash>/` using `FAISS.save_local`
- On re-upload of the same PDF, index loaded from cache — skips embedding entirely
- Cache hit surfaced in the UI success banner

**Why it matters**: Embedding a document on every restart wastes API tokens and time. Cache makes re-opening the same PDF instant.

---

### 10. RAGAS Evaluation Tab (`app.py`, `utils/evaluator.py`)

- Dedicated **📊 Evaluation** tab in the Streamlit UI
- Enter 1–10 test questions (ground truth optional)
- Pipeline answers each question, RAGAS evaluates the results

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? (detects hallucination) |
| **Answer Relevancy** | Does the answer actually address the question? |
| **Context Precision** | Are the retrieved chunks relevant? (requires ground truth) |

**Why it matters**: Without evaluation you can't tell if changes help. RAGAS gives measurable scores to compare retrieval strategies.

---

### 11. Robustness Fixes

- `utils/embeddings.py`: removed silent `except: pass` — OpenAI fallback now logs a warning with the reason before switching to local embeddings
- `app.py`: sidebar shows API key status, active pipeline features, and all tunable env vars

---

## Architecture After Optimization

```
User Question
    │
    ▼
Query Rewriting (conversational coherence)
    │
    ├─► HyDE Generation → hypothetical passage for FAISS
    └─► Multi-Query Generation → 3 rephrasings
              │
              ▼
    Hybrid Retrieval (per query variant)
    ├── FAISS vector search (uses HyDE embedding)
    └── BM25 keyword search (uses original query)
              │
              ▼
    Reciprocal Rank Fusion (all results merged)
              │
              ▼
    Cross-Encoder Reranking (top 4 selected)
              │
              ▼
    Contextual Compression (irrelevant sentences removed)
              │
              ▼
    LLM Answer Generation
    (with conversation history + [p.X] citations)
              │
              ▼
    Answer + Retrieved Chunks shown in UI
```

---

## New Dependencies

| Package | Purpose |
|---|---|
| `rank-bm25` | BM25 sparse retrieval |
| `ragas` | Retrieval-Augmented Generation evaluation metrics |
| `datasets` | Required by RAGAS for dataset construction |

---

## Tunable Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RAG_CHUNK_SIZE` | `900` | Character size per chunk |
| `RAG_CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `RAG_K_INITIAL` | `10` | Candidates retrieved per channel before reranking |
| `RAG_K_FINAL` | `4` | Final chunks passed to the LLM after reranking |
| `OPENAI_API_KEY` | — | Enables HyDE, multi-query, compression, citations, RAGAS |
