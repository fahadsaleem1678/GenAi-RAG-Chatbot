# GenAI RAG — Document-QA with Retrieval-Augmented Generation

A modern, easy-to-run system for asking questions over PDF documents. This repository combines document ingestion, embedding-based search, sparse retrieval, reranking, and optional LLM answer generation into a single end-to-end pipeline.

## What it does

- Loads PDF files and converts them into searchable text chunks
- Builds or reuses a FAISS vector index for fast semantic retrieval
- Uses BM25 sparse search and reciprocal rank fusion for better recall
- Optionally generates answers with an LLM using retrieved evidence
- Supports evaluation and comparison of retrieval strategies
- Can run locally or inside Docker

## Why use it

- Works with university lecture notes, reports, manuals, and research papers
- Gives grounded answers with citations to specific pages
- Handles follow-up questions with conversational query rewriting
- Lets you compare baseline retrieval against optimized hybrid search
- Includes a simple Streamlit UI for quick experimentation

## Quick start

### Run locally

1. Clone the repository

```bash
git clone https://github.com/<your-username>/genAI-rag.git
cd genAI-rag
```

2. Install Python dependencies

```bash
pip install -r requirements.txt
```

3. Set an API key for one supported provider

```bash
set OPENAI_API_KEY=sk-...
```

4. Start the app

```bash
streamlit run app.py
```

5. Open the app in your browser at `http://localhost:8501`

### Run with Docker

```bash
docker-compose up --build
```

If you want an API key in the Docker environment, add it before the command:

```bash
set XAI_API_KEY=xai-...
docker-compose up --build
```

## Supported providers

This project is designed to work with one API key from any supported provider.

- `OPENAI_API_KEY` — OpenAI models
- `XAI_API_KEY` — xAI / Grok
- `GROQ_API_KEY` — Groq

If no key is provided, retrieval still works for document search, but LLM answer generation will be disabled.

## Main files

- `app.py` — Streamlit application and user interface
- `rag_pipeline.py` — Orchestrates retrieval, reranking, and answer generation
- `prompts.py` — Prompt templates used for query rewriting and generation
- `requirements.txt` — Python dependencies
- `docker-compose.yml` — Docker setup for local deployment
- `Dockerfile` — Container definition
- `optimization.md` — Notes and ideas for improving the pipeline

## Utility modules

- `utils/pdf_loader.py` — PDF loading and text extraction
- `utils/chunking.py` — Splits documents into searchable chunks
- `utils/embeddings.py` — Embedding creation and model handling
- `utils/vector_store.py` — FAISS index management and persistence
- `utils/retriever.py` — Hybrid retrieval, ranking, and search logic
- `utils/evaluator.py` — Evaluation metrics and comparison tools

## Usage notes

- The first upload of a PDF builds the index and may take time.
- Subsequent loads reuse cached FAISS indexes for faster startup.
- Citations are linked to page metadata so answers remain traceable.
- The pipeline is built to be extensible: you can add new retrieval strategies or models.

## Tips

- Use shorter queries for direct answers and longer queries for deeper context.
- Upload a single document or a set of documents for multi-file search.
- Check `optimization.md` to understand how the system was tuned.

## Requirements

- Python 3.11 or newer
- Docker (optional)
- One supported API key for full LLM capabilities

## License

This repository is provided as a working example for document QA and experimentation.
