import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"local_files_only": True},
    )


def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=get_embeddings(),
    )


def load_document(file_path: str, suffix: str) -> list:
    suffix = suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif suffix in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif suffix == ".csv":
        loader = CSVLoader(file_path, encoding="utf-8")
    elif suffix in (".xlsx", ".xls"):
        loader = UnstructuredExcelLoader(file_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader.load()


def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(docs)


def ingest_file(uploaded_file, vectorstore: Chroma = None) -> int:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        docs = load_document(tmp_path, suffix)
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        # Drop pages/rows with no extractable text (e.g. scanned image PDFs)
        docs = [d for d in docs if d.page_content.strip()]
        if not docs:
            raise ValueError(
                "No text could be extracted from this file. "
                "If it is a scanned PDF, please use a text-based PDF or convert it with OCR first."
            )

        chunks = split_documents(docs)
        chunks = [c for c in chunks if c.page_content.strip()]
        if not chunks:
            raise ValueError("Document was loaded but produced no usable text chunks after splitting.")

        vs = vectorstore or get_vectorstore()
        vs.add_documents(chunks)
        return len(chunks)
    finally:
        os.unlink(tmp_path)


def get_chunk_count(vectorstore: Chroma = None) -> int:
    try:
        vs = vectorstore or get_vectorstore()
        return vs._collection.count()
    except Exception:
        return 0
