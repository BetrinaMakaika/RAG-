from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
)

def load_documents(data_dir: str = "data/") -> List:
    """
    Load documents from the data directory.
    Supports: text files, PDFs, and web URLs from urls.txt
    """
    path = Path(data_dir)
    documents = []

    # Load local text files
    for file_path in path.glob("*.txt"):
        if file_path.name not in ["urls.txt", "url.txt"]:  # Skip URL files
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

    # Load PDF files
    for pdf_path in path.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {e}")

    # Load web URLs from urls.txt
    urls_file = path / "url.txt"
    if not urls_file.exists():
        urls_file = path / "urls.txt"
    if urls_file.exists():
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
            for url in urls:
                try:
                    web_loader = WebBaseLoader(url)
                    web_docs = web_loader.load()
                    # Add metadata to web documents
                    for doc in web_docs:
                        doc.metadata["source_type"] = "web"
                        doc.metadata["url"] = url
                    documents.extend(web_docs)
                except Exception as e:
                    print(f"Error loading URL {url}: {e}")

    return documents


def chunk_documents(
    documents: List,
    chunk_size: int = 2000,
    chunk_overlap: int = 50,
    chunking_strategy: str = "recursive"
) -> List:
    """
    Split documents into chunks.

    Students MUST modify this function:
    - Experiment with different chunk sizes
    - Try different text splitters (Markdown, Token-based)
    - Add metadata to chunks (source, page number, etc.)
    """
    if chunking_strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    elif chunking_strategy == "markdown":
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("##", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    elif chunking_strategy == "token":
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")

    chunks = splitter.split_documents(documents)

    # Add custom chunk metadata for tracking
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents, created {len(chunks)} chunks")
