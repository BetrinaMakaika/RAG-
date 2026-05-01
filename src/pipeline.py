"""
RAG Pipeline Orchestration
===========================
Main pipeline that ties together all RAG components.
"""

import logging
from typing import List, Optional, Dict
from pathlib import Path

from langchain_community.vectorstores import Chroma
from .loader import load_documents, chunk_documents
from .embedder import get_embedder
from .retriever import create_vectorstore, get_retriever
from .generator import get_llm, create_rag_prompt, create_qa_chain, generate_response


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG Pipeline class.

    Students MUST modify this class:
    - Add initialization options
    - Implement caching
    - Add evaluation hooks
    - Customize preprocessing/postprocessing
    """

    def __init__(
        self,
        data_dir: str = "data/",
        embedder_provider: str = "huggingface",
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "huggingface",
        llm_model: str = "gpt2",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        retrieval_k: int = 2,
        vectorstore_type: str = "chroma",
        persist_dir: Optional[str] = "vectorstore"
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.persist_dir = persist_dir

        # Initialize components - students can modify configurations
        logger.info("Initializing embedder...")
        self.embedder = get_embedder(
            provider=embedder_provider,
            model_name=embedder_model
        )

        logger.info("Initializing LLM...")
        self.llm = get_llm(
            provider=llm_provider,
            model_name=llm_model
        )

        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

    def load_and_index(self, force_rebuild: bool = False):
        """
        Load documents and create vector index.

        Students MUST modify:
        - Add incremental indexing
        - Implement document update logic
        """
        # Check if vectorstore exists and is valid
        persist_path = Path(self.persist_dir) if self.persist_dir else None
        vectorstore_valid = False
        
        if persist_path and persist_path.exists() and not force_rebuild:
            try:
                logger.info("Loading existing vectorstore...")
                self.vectorstore = Chroma(
                    persist_directory=str(persist_path),
                    embedding_function=self.embedder
                )
                # Verify vectorstore is not empty
                if self.vectorstore and len(self.vectorstore.get()['ids']) > 0:
                    vectorstore_valid = True
                    logger.info(f"Loaded vectorstore with {len(self.vectorstore.get()['ids'])} documents")
            except Exception as e:
                logger.warning(f"Failed to load vectorstore: {e}. Will rebuild...")
                vectorstore_valid = False
        
        if not vectorstore_valid:
            logger.info("Loading and chunking documents...")
            documents = load_documents(self.data_dir)
            chunks = chunk_documents(
                documents,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

            logger.info("Creating vector index...")
            self.vectorstore = create_vectorstore(
                chunks,
                self.embedder,
                db_type="chroma",
                persist_dir=self.persist_dir
            )

        logger.info("Setting up retriever...")
        self.retriever = get_retriever(
            self.vectorstore,
            k=self.retrieval_k
        )
