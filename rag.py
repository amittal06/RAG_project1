import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_chroma import Chroma
from ingest import get_vectorstore

load_dotenv()

PROMPT_TEMPLATE = """You are a precise assistant. Answer ONLY from the provided context.
If the answer is not present in the context, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}
"""


def _format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain(vectorstore: Chroma = None):
    vs = vectorstore or get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def query(question: str, vectorstore: Chroma = None) -> dict:
    chain, retriever = get_rag_chain(vectorstore)
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return {"answer": answer, "sources": sources}
