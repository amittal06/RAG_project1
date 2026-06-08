# RAG Document Q&A — Project Plan

## Overview

A Retrieval-Augmented Generation (RAG) web application that lets users upload documents (PDF, TXT, CSV, XLSX, DOCX, MD), stores them as embeddings in a Chroma vector database, and answers natural-language questions about the uploaded content.

---

## Architecture

```
myRAGProject/
├── app.py              # Streamlit web UI (entry point)
├── ingest.py           # Document loading, chunking, Chroma storage
├── rag.py              # RAG chain (retriever + LLM prompt + OpenAI)
├── requirements.txt    # All dependencies
├── .env.example        # API key template
├── PLAN.md             # This file
└── vectorstore/        # Chroma DB persistence dir (auto-created at runtime)
```

---

## Tech Stack

| Layer       | Choice                                  | Reason                                  |
|-------------|-----------------------------------------|-----------------------------------------|
| UI          | Streamlit                               | Pure Python, file uploader built-in     |
| LLM         | OpenAI GPT-4o-mini                      | Fast, cost-effective, high quality      |
| Embeddings  | HuggingFace `all-MiniLM-L6-v2` (local) | Free, runs locally, no API cost         |
| Vector DB   | Chroma                                  | Persistent by default, easy to use      |
| Orchestration | LangChain                             | Rich document loaders + LCEL chains     |

---

## Supported Document Formats

| Extension       | Loader                    |
|-----------------|---------------------------|
| `.pdf`          | `PyPDFLoader`             |
| `.txt`, `.md`   | `TextLoader`              |
| `.csv`          | `CSVLoader`               |
| `.xlsx`, `.xls` | `UnstructuredExcelLoader` |
| `.docx`         | `Docx2txtLoader`          |

---

## File Responsibilities

### `ingest.py`
- `get_embeddings()` — loads local HuggingFace embedding model
- `get_vectorstore()` — connects to Chroma with the embedding model
- `load_document(path, suffix)` — dispatches to the right LangChain loader
- `split_documents(docs)` — chunks with `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
- `ingest_file(uploaded_file)` — end-to-end: save temp → load → split → store → return chunk count
- `get_chunk_count()` — total chunks currently in Chroma

### `rag.py`
- `get_rag_chain()` — builds LCEL chain: retriever → prompt → GPT-4o-mini → string output
- `query(question)` — returns `{"answer": str, "sources": list[Document]}`

### `app.py`
- **Tab 1 — Upload Documents**: multi-file uploader, "Process" button, per-file status feedback
- **Tab 2 — Ask Questions**: chat-style interface, answer display, source chunks in expander
- Sidebar shows live chunk count from Chroma

---

## UI Layout

```
┌─────────────────────────────────────────────────────┐
│ Sidebar: "RAG Q&A" | Chunks in vector store: 42     │
├─────────────────────────────────────────────────────┤
│  📁 Upload Documents  |  💬 Ask Questions            │
│ ─────────────────────────────────────────────────── │
│  [Upload tab]                                        │
│  Drop files here (PDF, TXT, CSV, XLSX, DOCX)        │
│  [Process Documents]                                 │
│  ✅ report.pdf → 38 chunks added                     │
│  ✅ data.csv   → 12 chunks added                     │
│                                                      │
│  [Ask tab — chat interface]                          │
│  User: What is the main topic?                       │
│  Assistant: The main topic is...                     │
│  ▼ Sources (4 chunks)                               │
│    report.pdf | page 3 | "Revenue increased..."     │
│                                                      │
│  [Ask a question about your documents...]            │
└─────────────────────────────────────────────────────┘
```

---

## Setup & Running

### 1. Install dependencies
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API key
```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Run the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Verification Checklist

- [ ] Upload a PDF → chunk count shown, no errors
- [ ] Upload a CSV → processes without error
- [ ] Switch to Ask tab → question answered using document content
- [ ] Off-topic question → "I don't know based on the provided documents"
- [ ] Restart the app → Chroma persists, questions still work without re-uploading
