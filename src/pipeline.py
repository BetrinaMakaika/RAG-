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

        logger.info("Creating QA chain...")
        prompt = create_rag_prompt()
        self.qa_chain = create_qa_chain(self.llm, self.retriever, prompt)

    def query(self, question: str, return_sources: bool = True) -> Dict:
        """
        Query the RAG pipeline.
        Includes query preprocessing and response postprocessing for student services.
        """
        if self.qa_chain is None:
            raise RuntimeError("Pipeline not initialized. Call load_and_index() first.")

        # Query preprocessing - normalize student questions
        processed_question = question.strip().lower()
        
        # Common student service keywords mapping
        keywords = {
            'reg': 'registration',
            'enroll': 'enrollment',
            'fee': 'fees',
            'payment': 'fees',
            'time': 'timetable',
            'schedule': 'timetable',
            'hostel': 'accommodation',
            'dorm': 'accommodation',
            'advisor': 'academic advising',
            'department': 'departmental support'
        }
        
        for abbr, full in keywords.items():
            if abbr in processed_question:
                processed_question = processed_question.replace(abbr, full)

        logger.info(f"Querying: {question}")
        response = generate_response(self.qa_chain, processed_question, return_sources)
        
        # Response postprocessing
        if "answer" in response:
            # Add confidence indicator based on response length
            answer_text = response["answer"].strip()
            if not answer_text or answer_text == "No answer generated.":
                response["confidence"] = "low"
            elif len(answer_text) < 20:
                response["confidence"] = "low"
            else:
                response["confidence"] = "medium"
        
        return response

    def evaluate(self, test_queries: List[Dict]) -> Dict:
        """
        Evaluate the pipeline on test queries.
        
        Computes basic metrics:
        - Answer length (as proxy for confidence)
        - Source availability
        - Query success rate
        """
        if self.qa_chain is None:
            raise RuntimeError("Pipeline not initialized. Call load_and_index() first.")
        
        results = []
        total_queries = len(test_queries)
        successful_queries = 0
        sources_found = 0
        total_answer_length = 0
        
        for item in test_queries:
            query = item["question"]
            expected = item.get("expected_answer", "")

            response = self.query(query, return_sources=True)
            
            answer_text = response.get("answer", "")
            has_sources = len(response.get("sources", [])) > 0
            
            results.append({
                "query": query,
                "expected": expected,
                "answer": answer_text,
                "sources": response.get("sources", []),
                "confidence": response.get("confidence", "unknown")
            })
            
            # Calculate metrics
            if answer_text and answer_text != "No answer generated.":
                successful_queries += 1
                total_answer_length += len(answer_text)
            
            if has_sources:
                sources_found += 1

        # Compute evaluation metrics
        metrics = {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
            "avg_answer_length": total_answer_length / successful_queries if successful_queries > 0 else 0,
            "sources_found_rate": sources_found / total_queries if total_queries > 0 else 0,
            "results": results
        }
        
        return metrics


def main():
    """Main entry point for testing - use main.py instead."""
    raise RuntimeError(
        "pipeline.py should not be run directly. "
        "Please use: python main.py"
    )


if __name__ == "__main__":
    print("⚠️  Error: pipeline.py should not be run directly.")
    print("Please use: python main.py")
    print("\nThe RAGPipeline class is imported by main.py and the src package.")
    import sys
    sys.exit(1)
