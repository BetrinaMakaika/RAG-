from __future__ import annotations

import re
from pathlib import Path
from typing import List, Literal, Optional
from datetime import datetime

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
)
from langchain.schema import Document


CATEGORY_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"registr|enrol|admission", re.I), "registration"),
    (re.compile(r"fee|tuition|payment|finance|burs", re.I), "fees"),
    (re.compile(r"timetable|schedule|lecture|exam|calendar", re.I), "timetable"),
    (re.compile(r"hostel|accommodat|residen|dorm", re.I), "hostel"),
    (re.compile(r"dept|department|faculty|course|unit|module", re.I), "departmental"),
]

DEFAULT_CATEGORY = "general"


def _detect_category(source: str) -> str:
    """Infer the student-services category from a file path or URL."""
    for pattern, category in CATEGORY_RULES:
        if pattern.search(source):
            return category
    return DEFAULT_CATEGORY



def load_text_files(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for fp in data_dir.glob("*.txt"):
        loader = TextLoader(str(fp), encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata.setdefault("source", fp.name)
            doc.metadata["file_type"] = "txt"
        docs.extend(loaded)
    return docs


def load_pdf_files(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for fp in data_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(fp))
        loaded = loader.load()     
        for doc in loaded:
            doc.metadata.setdefault("source", fp.name)
            doc.metadata["file_type"] = "pdf"
        docs.extend(loaded)
    return docs


def load_markdown_files(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for fp in data_dir.glob("*.md"):
        loader = UnstructuredMarkdownLoader(str(fp))
        loaded = loader.load()
        for doc in loaded:
            doc.metadata.setdefault("source", fp.name)
            doc.metadata["file_type"] = "markdown"
        docs.extend(loaded)
    return docs


def load_web_pages(urls: List[str]) -> List[Document]:
    """Load one or more public university web pages."""
    if not urls:
        return []
    loader = WebBaseLoader(urls)
    docs = loader.load()
    for doc in docs:
        doc.metadata["file_type"] = "web"
    return docs


def load_documents(
    data_dir: str = "data/",
    include_pdfs: bool = True,
    include_markdown: bool = True,
    extra_urls: Optional[List[str]] = None,
) -> List[Document]:
    
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    documents: List[Document] = []

    documents.extend(load_text_files(path))

    if include_pdfs:
        documents.extend(load_pdf_files(path))

    if include_markdown:
        documents.extend(load_markdown_files(path))

    if extra_urls:
        documents.extend(load_web_pages(extra_urls))

    
    for doc in documents:
        src = doc.metadata.get("source", "")
        doc.metadata["category"] = _detect_category(src)
        doc.metadata["ingested_at"] = datetime.utcnow().isoformat()

    print(
        f"[loader] Loaded {len(documents)} documents "
        f"from '{data_dir}'"
        + (f" + {len(extra_urls)} URL(s)" if extra_urls else "")
    )
    return documents


ChunkStrategy = Literal["recursive", "markdown", "token"]


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    chunking_strategy: ChunkStrategy = "recursive",
    propagate_headers: bool = True,
) -> List[Document]:
  
    chunks: List[Document] = []

    if chunking_strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_documents(documents)

    elif chunking_strategy == "markdown":
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        # Secondary recursive split to cap chunk size
        secondary = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for doc in documents:
            md_chunks = md_splitter.split_text(doc.page_content)
            for mc in md_chunks:
                # Carry parent metadata forward
                mc.metadata.update({k: v for k, v in doc.metadata.items()
                                     if k not in mc.metadata})
                if propagate_headers:
                    header_prefix = " › ".join(
                        v for k, v in mc.metadata.items()
                        if k in ("h1", "h2", "h3") and v
                    )
                    if header_prefix:
                        mc.page_content = f"[{header_prefix}]\n{mc.page_content}"
            sub_chunks = secondary.split_documents(md_chunks)
            chunks.extend(sub_chunks)

    elif chunking_strategy == "token":
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

    else:
        raise ValueError(
            f"Unknown chunking_strategy '{chunking_strategy}'. "
            "Choose from: recursive | markdown | token"
        )

    # Enrich every chunk with stable metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata.setdefault("category", DEFAULT_CATEGORY)
        # Short preview useful for debugging / logging
        chunk.metadata["preview"] = chunk.page_content[:80].replace("\n", " ")

    print(
        f"[loader] {len(documents)} document(s) → "
        f"{len(chunks)} chunks "
        f"(strategy={chunking_strategy}, size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks



def filter_by_category(
    chunks: List[Document],
    category: str,
) -> List[Document]:
    """Return only chunks matching a specific student-services category."""
    return [c for c in chunks if c.metadata.get("category") == category]


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/"

    docs = load_documents(
        data_dir=data_path,
        include_pdfs=True,
        include_markdown=True,
    )
    chunks = chunk_documents(docs, chunking_strategy="recursive")

    # Category breakdown
    from collections import Counter
    cats = Counter(c.metadata["category"] for c in chunks)
    print("\nCategory breakdown:")
    for cat, count in cats.most_common():
        print(f"  {cat:<20} {count:>4} chunks")

    # Sample chunk
    if chunks:
        sample = chunks[0]
        print(f"\nSample chunk #{sample.metadata['chunk_id']}:")
        print(f"  source   : {sample.metadata.get('source')}")
        print(f"  category : {sample.metadata.get('category')}")
        print(f"  preview  : {sample.metadata.get('preview')}")