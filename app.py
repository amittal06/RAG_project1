import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ingest import ingest_file, get_chunk_count
from rag import query

st.set_page_config(page_title="RAG Document Q&A", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("RAG Document Q&A")
    st.markdown("Upload documents, then ask questions about their content.")
    st.divider()
    count = get_chunk_count()
    st.metric("Chunks in vector store", count)
    st.caption("Powered by Chroma + HuggingFace embeddings + GPT-4o-mini")

# --- Tabs ---
tab_upload, tab_ask = st.tabs(["📁 Upload Documents", "💬 Ask Questions"])

# ── Upload tab ──────────────────────────────────────────────────────────────
with tab_upload:
    st.header("Upload Documents")
    st.write("Supported formats: PDF, TXT, CSV, XLSX, XLS, DOCX, MD")

    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=["pdf", "txt", "csv", "xlsx", "xls", "docx", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            results = []
            for f in uploaded_files:
                with st.spinner(f"Processing {f.name}..."):
                    try:
                        n = ingest_file(f)
                        results.append((f.name, n, None))
                    except Exception as e:
                        results.append((f.name, 0, str(e)))

            for name, n, err in results:
                if err:
                    st.error(f"{name}: {err}")
                else:
                    st.success(f"{name} — {n} chunks added")

            st.rerun()

# ── Ask tab ─────────────────────────────────────────────────────────────────
with tab_ask:
    st.header("Ask a Question")

    if "history" not in st.session_state:
        st.session_state.history = []

    # Display chat history
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            if item["sources"]:
                with st.expander(f"Sources ({len(item['sources'])} chunks)"):
                    for doc in item["sources"]:
                        src = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "")
                        label = f"**{src}**" + (f" — page {page + 1}" if page != "" else "")
                        st.markdown(label)
                        st.caption(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                        st.divider()

    # Input
    question = st.chat_input("Ask a question about your documents...")
    if question:
        if get_chunk_count() == 0:
            st.warning("No documents ingested yet. Please upload files in the Upload tab first.")
        else:
            with st.spinner("Thinking..."):
                result = query(question)

            st.session_state.history.append({
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
            })
            st.rerun()
