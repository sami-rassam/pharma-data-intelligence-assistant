from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Uses a free local HuggingFace embedding model.
    This avoids needing paid embedding APIs.
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
    vectorstore.save_local(path)


def load_vectorstore(path: str = "faiss_index") -> FAISS:
    embeddings = get_embedding_model()
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )


def similarity_search(vectorstore: FAISS, query: str, k: int = 5):
    """
    Retrieves the most relevant chunks.
    """
    return vectorstore.similarity_search(query, k=k)