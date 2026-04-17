"""
RAG Retriever
- Queries the FAISS vector store
- Returns top-k relevant chunks + their source filenames
"""

from typing import Tuple, List
from langchain_huggingface import HuggingFaceEmbeddings
from rag.ingest import load_vectorstore


def retrieve(query: str, k: int = 4) -> Tuple[List[str], List[str]]:
    """
    Retrieve top-k relevant chunks for a given query.

    Args:
        query: The search query (crop name + user question)
        k: Number of chunks to retrieve

    Returns:
        (chunks, sources) — list of text chunks and their source filenames
    """
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)

    chunks = [doc.page_content for doc in results]
    sources = list({doc.metadata.get("source", "unknown") for doc in results})

    return chunks, sources
