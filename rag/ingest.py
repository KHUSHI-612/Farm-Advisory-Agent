"""
RAG Ingestion Pipeline
- Loads all .txt files from rag/docs/
- Splits them into overlapping chunks
- Embeds with HuggingFace (free, no API key needed)
- Saves FAISS vector store locally to rag/vectorstore/
"""

import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


DOCS_DIR = Path(__file__).parent / "docs"
VECTORSTORE_DIR = Path(__file__).parent / "vectorstore"


def build_vectorstore():
    """Load docs, chunk, embed, and save FAISS index."""
    print("Loading agronomy documents...")

    docs = []
    for txt_file in DOCS_DIR.glob("*.txt"):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        loaded = loader.load()
        # Tag each doc with its source filename
        for doc in loaded:
            doc.metadata["source"] = txt_file.name
        docs.extend(loaded)
        print(f"  Loaded: {txt_file.name}")

    if not docs:
        raise ValueError(f"No .txt files found in {DOCS_DIR}")

    print(f"\nSplitting {len(docs)} documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("\nBuilding embeddings (this may take a minute on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"Vector store saved to {VECTORSTORE_DIR}")

    return vectorstore


def load_vectorstore():
    """Load existing FAISS index, or build it if it doesn't exist."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    if (VECTORSTORE_DIR / "index.faiss").exists():
        return FAISS.load_local(
            str(VECTORSTORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Vector store not found — building now...")
        return build_vectorstore()


if __name__ == "__main__":
    build_vectorstore()
    print("\nDone! Vector store is ready.")
