from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Uses a free local HuggingFace embedding model.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vectorstore(chunks: List[Document]) -> FAISS:
    """
    Creates a FAISS vector database from document chunks.
    """
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_vectorstore(vectorstore: FAISS, path: str = "faiss_index") -> None:
    """
    Saves FAISS vectorstore locally.
    """
    vectorstore.save_local(path)


def load_vectorstore(path: str = "faiss_index") -> FAISS:
    """
    Loads FAISS vectorstore from disk.
    """
    embeddings = get_embedding_model()
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )


def vectorstore_exists(path: str = "faiss_index") -> bool:
    """
    Checks if a saved FAISS index exists.
    """
    index_path = Path(path)
    return index_path.exists() and any(index_path.iterdir())


def get_or_create_vectorstore(chunks: List[Document], path: str = "faiss_index") -> FAISS:
    """
    Loads saved FAISS index if available, otherwise creates and saves a new one.
    """
    if vectorstore_exists(path):
        return load_vectorstore(path)

    vectorstore = create_vectorstore(chunks)
    save_vectorstore(vectorstore, path)
    return vectorstore


def similarity_search(vectorstore: FAISS, query: str, k: int = 5):
    """
    Retrieves the most relevant chunks.
    """
    return vectorstore.similarity_search(query, k=k)