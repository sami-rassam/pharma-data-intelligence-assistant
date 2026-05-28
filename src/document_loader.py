from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_single_file(file_path: str) -> List[Document]:
    """
    Loads one PDF, TXT, MD or DOCX file into LangChain Document objects.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in [".txt", ".md"]:
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return loader.load()


def load_documents_from_folder(folder_path: str = "data/sample_docs") -> List[Document]:
    """
    Loads all supported documents from a folder.
    """
    folder = Path(folder_path)
    documents = []

    supported_extensions = ["*.pdf", "*.txt", "*.md", "*.docx"]

    for extension in supported_extensions:
        for file_path in folder.glob(extension):
            try:
                loaded_docs = load_single_file(str(file_path))
                for doc in loaded_docs:
                    doc.metadata["source"] = file_path.name
                documents.extend(loaded_docs)
            except Exception as error:
                print(f"Could not load {file_path}: {error}")

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Splits documents into smaller chunks for vector search.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    return splitter.split_documents(documents)


def load_and_split_documents(folder_path: str = "data/sample_docs") -> List[Document]:
    """
    Full loading pipeline.
    """
    documents = load_documents_from_folder(folder_path)
    chunks = split_documents(documents)
    return chunks