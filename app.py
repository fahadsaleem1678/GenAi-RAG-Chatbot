import os
from collections import Counter

import streamlit as st

from rag_pipeline import RAGPipeline


st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")


def initialize_state():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retrieved_chunks" not in st.session_state:
        st.session_state.retrieved_chunks = []
    if "document_ready" not in st.session_state:
        st.session_state.document_ready = False
    if "document_name" not in st.session_state:
        st.session_state.document_name = ""
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None


def render_sidebar():
    st.sidebar.title("PDF RAG Settings")
    st.sidebar.caption("PDF → chunks → FAISS+BM25 → HyDE/multi-query → rerank → compress → answer")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### API Keys")
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_xai = bool(os.getenv("XAI_API_KEY"))
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_key = has_openai or has_xai or has_groq
    if has_openai:
        st.sidebar.success("✅ OpenAI key set — full pipeline + OpenAI embeddings active.")
    elif has_xai:
        st.sidebar.success("✅ xAI (Grok) key set — full pipeline active (local embeddings used).")
        st.sidebar.caption("Model: grok-3-mini · Embeddings: all-MiniLM-L6-v2 (local)")
    elif has_groq:
        st.sidebar.success("✅ Groq key set — full pipeline active (local embeddings used).")
        st.sidebar.caption("Model: llama-3.3-70b-versatile · Embeddings: all-MiniLM-L6-v2 (local)")
    else:
        st.sidebar.warning(
            "⚠️ No API key. Retrieval works; HyDE, multi-query, compression, RAGAS "
            "need `OPENAI_API_KEY`, `XAI_API_KEY`, or `GROQ_API_KEY`."
        )

    st.sidebar.markdown("---")
    with st.sidebar.expander("Retrieval settings", expanded=False):
        st.write(f"Chunk size: `{os.getenv('RAG_CHUNK_SIZE', '900')}`")
        st.write(f"Chunk overlap: `{os.getenv('RAG_CHUNK_OVERLAP', '150')}`")
        st.write(f"k_initial (per source): `{os.getenv('RAG_K_INITIAL', '10')}`")
        st.write(f"k_final (after rerank): `{os.getenv('RAG_K_FINAL', '4')}`")
        st.caption("Override via env vars before launching streamlit.")

    if has_key:
        st.sidebar.markdown("---")
        with st.sidebar.expander("Pipeline features (LLM-powered)", expanded=False):
            st.write("✅ Query rewriting (conversational)")
            st.write("✅ HyDE — hypothetical document embedding")
            st.write("✅ Multi-query retrieval (3 phrasings + RRF)")
            st.write("✅ Contextual compression")
            st.write("✅ Citation-style answers [p.X]")
            st.write("✅ RAGAS evaluation")


# ── Dataset / Visualisation tab ─────────────────────────────────────────────

def render_dataset_tab():
    st.subheader("📈 Dataset Overview & Preprocessing Visualizations")

    pipeline = st.session_state.pipeline
    docs = pipeline.source_documents
    chunks = pipeline.chunks

    if not docs:
        st.info("Upload a PDF to see dataset statistics.")
        return

    # ── Summary metrics ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    total_chars = sum(len(d.page_content) for d in docs)
    avg_chunk = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
    col1.metric("Pages", len(docs))
    col2.metric("Chunks", len(chunks))
    col3.metric("Total characters", f"{total_chars:,}")
    col4.metric("Avg chunk size", f"{avg_chunk:.0f} chars")

    st.markdown("---")

    # ── Page-by-page text length bar chart ──────────────────────────────────
    st.markdown("#### Text length per page")
    page_lengths = {f"p.{d.metadata.get('page', i+1)}": len(d.page_content) for i, d in enumerate(docs)}
    st.bar_chart(page_lengths)

    # ── Chunk size distribution ──────────────────────────────────────────────
    st.markdown("#### Chunk size distribution")
    chunk_sizes = [len(c.page_content) for c in chunks]
    size_buckets: dict[str, int] = {}
    for sz in chunk_sizes:
        bucket = f"{(sz // 100) * 100}–{(sz // 100) * 100 + 99}"
        size_buckets[bucket] = size_buckets.get(bucket, 0) + 1
    st.bar_chart(size_buckets)

    # ── Top 20 word frequencies ──────────────────────────────────────────────
    st.markdown("#### Top 20 most frequent words")
    import re
    STOPWORDS = {
        "the","a","an","and","or","in","of","to","is","are","was","were","it",
        "that","this","for","on","with","as","be","at","by","from","have","has",
        "had","not","but","we","you","he","she","they","their","our","its","i",
        "will","can","do","does","did","so","if","there","been","which","about",
        "into","also","more","than","all","one","would","could","should","may",
    }
    all_text = " ".join(d.page_content for d in docs).lower()
    words = re.findall(r"\b[a-z]{3,}\b", all_text)
    filtered = [w for w in words if w not in STOPWORDS]
    freq = dict(Counter(filtered).most_common(20))
    st.bar_chart(freq)

    # ── Raw page content preview ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Page content preview")
    for doc in docs:
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:400].replace("\n", " ")
        with st.expander(f"Page {page} ({len(doc.page_content)} chars)", expanded=False):
            st.write(preview + ("..." if len(doc.page_content) > 400 else ""))


# ── Chat tab ─────────────────────────────────────────────────────────────────

def _render_chunks(chunks):
    for idx, chunk in enumerate(chunks, start=1):
        preview = chunk["content"].strip().replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:300].rstrip() + "..."
        page = chunk.get("page", "?")
        score = chunk.get("score", 0.0)
        st.markdown(f"**Chunk {idx}** — page {page} | rerank score: `{score:.3f}`")
        st.write(preview)
        if chunk.get("original_content") and chunk["original_content"] != chunk["content"]:
            with st.expander("Show original (pre-compression)", expanded=False):
                st.write(chunk["original_content"])


def render_chat_tab():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about the uploaded PDF")

    if prompt:
        if not prompt.strip():
            st.warning("Please enter a non-empty question.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                try:
                    history_for_llm = st.session_state.messages[:-1]
                    response = st.session_state.pipeline.ask(prompt, history=history_for_llm)
                    answer = response["answer"]
                    chunks = response["retrieved_chunks"]
                    search_query = response.get("search_query", prompt)
                    hyde_used = response.get("hyde_used", False)
                    multi_used = response.get("multi_query_used", False)

                    badges = []
                    if search_query.strip() != prompt.strip():
                        badges.append(f"🔎 Rewritten: _{search_query}_")
                    if hyde_used:
                        badges.append("🧪 HyDE")
                    if multi_used:
                        badges.append("🔀 Multi-query")
                    if badges:
                        st.caption(" · ".join(badges))

                    with st.expander("Retrieved Chunks", expanded=False):
                        _render_chunks(chunks)

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.retrieved_chunks = chunks
                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")


# ── Evaluation tab ────────────────────────────────────────────────────────────

def render_eval_tab():
    st.subheader("RAGAS Evaluation & Model Comparison")

    has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("XAI_API_KEY") or os.getenv("GROQ_API_KEY"))
    if not has_key:
        st.warning(
            "RAGAS evaluation requires `OPENAI_API_KEY`, `XAI_API_KEY`, or `GROQ_API_KEY`. "
            "Set one and restart the app."
        )
        return

    if not st.session_state.document_ready:
        st.info("Upload a PDF in the Chat tab first.")
        return

    # ── Single-model RAGAS eval ──────────────────────────────────────────────
    st.markdown("### Single Pipeline Evaluation")
    st.write(
        "Evaluate the **optimised pipeline** (hybrid BM25+FAISS + HyDE + multi-query + reranker + compression) "
        "with RAGAS metrics: Faithfulness, Answer Relevancy, and optionally Context Precision."
    )

    num_cases = st.number_input("Number of test questions", min_value=1, max_value=10, value=3, key="ragas_n")
    test_cases = []
    for i in range(int(num_cases)):
        st.markdown(f"**Test case {i+1}**")
        col1, col2 = st.columns(2)
        with col1:
            q = st.text_input(f"Question {i+1}", key=f"eval_q_{i}",
                              placeholder="e.g. What is the main topic of this document?")
        with col2:
            gt = st.text_input(f"Ground truth (optional) {i+1}", key=f"eval_gt_{i}",
                               placeholder="Leave blank to skip context precision")
        if q.strip():
            test_cases.append({"question": q.strip(), **({"ground_truth": gt.strip()} if gt.strip() else {})})

    if st.button("▶ Run RAGAS Evaluation", disabled=not test_cases):
        with st.spinner(f"Running {len(test_cases)} test cases through the pipeline and RAGAS..."):
            from utils.evaluator import run_ragas_eval
            results = run_ragas_eval(test_cases, st.session_state.pipeline)
            st.session_state.eval_results = results

    if st.session_state.eval_results is not None:
        results = st.session_state.eval_results
        st.markdown("---")
        if not results:
            st.error("Evaluation failed — check logs for details.")
        else:
            labels = {
                "faithfulness": ("Faithfulness", "Is the answer grounded in the retrieved context?"),
                "answer_relevancy": ("Answer Relevancy", "Does the answer address the question?"),
                "context_precision": ("Context Precision", "Are retrieved chunks actually relevant?"),
            }
            cols = st.columns(len(results))
            for col, (metric, score) in zip(cols, results.items()):
                label, help_text = labels.get(metric, (metric, ""))
                col.metric(label=label, value=f"{score:.2%}", help=help_text)

    # ── Multi-model comparative analysis ────────────────────────────────────
    st.markdown("---")
    st.markdown("### Multi-Model Comparative Analysis")
    st.write(
        "Compare **three retrieval strategies** on the same questions to demonstrate "
        "the improvement at each stage of optimisation:"
    )
    st.markdown(
        "| Model | Strategy |\n"
        "|---|---|\n"
        "| **Baseline** | Naive FAISS top-k=4 (original approach) |\n"
        "| **Hybrid** | FAISS + BM25 + RRF + cross-encoder rerank |\n"
        "| **Optimised** | Hybrid + HyDE + multi-query + contextual compression |"
    )

    num_comp = st.number_input("Number of comparison questions", min_value=1, max_value=5, value=2, key="comp_n")
    comp_cases = []
    for i in range(int(num_comp)):
        q = st.text_input(f"Comparison question {i+1}", key=f"comp_q_{i}",
                          placeholder="e.g. Summarise the key findings.")
        if q.strip():
            comp_cases.append({"question": q.strip()})

    if st.button("▶ Run Comparative Analysis", disabled=not comp_cases):
        with st.spinner("Running all 3 strategies — this takes a moment..."):
            from utils.evaluator import run_comparative_eval
            comparison = run_comparative_eval(comp_cases, st.session_state.pipeline)
            st.session_state.comparison_results = comparison

    if st.session_state.comparison_results:
        st.markdown("---")
        st.markdown("#### Results per question")
        for item in st.session_state.comparison_results:
            with st.expander(f"Q: {item['question']}", expanded=True):
                cols = st.columns(3)
                for col, (strategy, data) in zip(cols, item["strategies"].items()):
                    col.markdown(f"**{strategy}**")
                    col.caption(f"Chunks retrieved: {data['num_chunks']}")
                    col.write(data["answer"][:400] + ("..." if len(data["answer"]) > 400 else ""))

        st.markdown("---")
        st.markdown("#### Prompt Engineering Registry")
        from prompts import PROMPT_REGISTRY
        for p in PROMPT_REGISTRY:
            with st.expander(f"Prompt {p['id']}: {p['name']} — _{p['technique']}_"):
                st.markdown(f"**Temperature:** `{p['temperature']}`")
                st.markdown(f"**Input variables:** `{', '.join(p['variables'])}`")
                st.markdown("**System prompt:**")
                st.code(p["system_prompt"], language="text")
                st.markdown("**User template:**")
                st.code(p["user_template"], language="text")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    initialize_state()
    render_sidebar()

    st.title("📄 University RAG Chatbot")
    st.write(
        "Upload a PDF and ask questions. Uses hybrid search (FAISS + BM25), HyDE, "
        "multi-query retrieval, cross-encoder reranking, and contextual compression."
    )

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.session_state.document_name != uploaded_file.name:
            st.session_state.pipeline.reset()
            st.session_state.messages = []
            st.session_state.retrieved_chunks = []
            st.session_state.document_ready = False
            st.session_state.document_name = uploaded_file.name
            st.session_state.eval_results = None
            st.session_state.comparison_results = None

        if not st.session_state.document_ready:
            with st.spinner("Processing PDF and building vector store..."):
                try:
                    result = st.session_state.pipeline.ingest_pdf(uploaded_file)
                    st.session_state.document_ready = True
                    cache_note = " (loaded from cache)" if result.get("cache_hit") else ""
                    st.success(
                        f"Loaded {uploaded_file.name}: {result['num_pages']} pages, "
                        f"{result['num_chunks']} chunks indexed{cache_note}."
                    )
                except Exception as exc:
                    st.error(f"Failed to process PDF: {exc}")
                    st.stop()
        else:
            st.success(f"Using indexed document: {uploaded_file.name}")

    if not st.session_state.document_ready:
        st.info("Upload a PDF to begin.")
        st.stop()

    st.markdown("---")

    chat_tab, dataset_tab, eval_tab = st.tabs(["💬 Chat", "📈 Dataset", "📊 Evaluation"])
    with chat_tab:
        render_chat_tab()
    with dataset_tab:
        render_dataset_tab()
    with eval_tab:
        render_eval_tab()


if __name__ == "__main__":
    main()
