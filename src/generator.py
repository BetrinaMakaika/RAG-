"""
LLM Generator Module
====================
Students: customize the prompt and LLM configuration.
"""

from typing import List, Optional, Dict
import re
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


def get_llm(
    provider: str = "huggingface",
    model_name: str = "distilgpt2",
    temperature: float = 0.5,
    **kwargs
):
    """
    Get an LLM for generation.

    modify this:
    - Choose appropriate model (gpt2, distilgpt2, etc.)
    - Adjust temperature for creativity vs accuracy
    - Add API key configurations

    Recommended local models (HuggingFace):
    - distilgpt2 (small, fast, cached)
    - facebook/opt-350m (better quality but larger)
    - meta-llama/Llama-2-7b (if you have GPU)
    """
    if provider == "ollama":
        return Ollama(model=model_name, temperature=temperature)
    elif provider == "huggingface":
        # Use local HuggingFace transformers for local inference (no API key needed)
        from langchain_community.llms import HuggingFacePipeline
        try:
            return HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text-generation",
                model_kwargs={},
                pipeline_kwargs={
                    "max_new_tokens": 100,
                    "do_sample": False
                }
            )
        except Exception as e:
            # Fallback with simpler settings
            return HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text-generation",
                model_kwargs={},
                pipeline_kwargs={
                    "max_new_tokens": 80
                }
            )
    elif provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def create_rag_prompt(
    system_message: Optional[str] = None,
    template: Optional[str] = None
) -> PromptTemplate:
    """
    Create a RAG prompt template for Student Services FAQ Assistant.
    Customized for UNIMA student inquiries about registration, fees, timetable, hostel, and departmental support.
    """
    if system_message is None:
        system_message = """You are a helpful Student Services FAQ Assistant for the University of Malawi (UNIMA).
You help students with questions about:
- Registration and enrollment
- Fees and payments
- Timetables and academic schedules
- Hostel and accommodation
- Departmental support and academic advising

Answer based on the provided context. If information is not available, say so clearly.
Be concise and helpful."""

    if template is None:
        template = """Based on this context, answer the question briefly in 1-2 sentences:

Context: {context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    return prompt


def create_qa_chain(llm, retriever, prompt: Optional[PromptTemplate] = None):
    """
    Create a RetrievalQA chain.

    Students MUST modify:
    - Chain type (stuff, map_reduce, refine)
    - Add return_source_documents=True
    - Implement custom output parsing
    """
    if prompt is None:
        prompt = create_rag_prompt()

    # Use chain_type_kwargs to pass the prompt with improved settings
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        },
        return_source_documents=True,
        verbose=False
    )

    return qa_chain


def generate_response(
    qa_chain,
    query: str,
    return_sources: bool = True
) -> Dict:
    """
    Generate a response using the RAG pipeline.

    Returns:
        Dict with 'answer' and optionally 'source_documents'
    """
    try:
        result = qa_chain.invoke({"query": query})

        # Handle case where result might be empty or malformed
        answer = result.get("result", "No answer generated.")
        
        # Ensure answer is a string
        if not isinstance(answer, str):
            answer = str(answer) if answer else "No answer generated."

        # Clean up the answer by removing the prompt template if present
        answer = answer.strip()
        
        # Extract just the answer portion if the full prompt template is included
        # The prompt template ends with "Answer:" so we want everything after that
        if 'Answer:' in answer:
            # Find the last occurrence of "Answer:" and take everything after it
            parts = answer.rsplit('Answer:', 1)
            if len(parts) > 1:
                answer = parts[1].strip()
        
        # Remove period if it's the only character (fragment detection)
        answer_stripped = answer.rstrip('.')
        
        # If answer is too short or appears to be a fragment (single word, very short), use fallback
        if len(answer_stripped) < 15 or (len(answer_stripped.split()) == 1 and len(answer_stripped) < 20):
            # Check if we have sources we can build from
            if return_sources and "source_documents" in result and result["source_documents"]:
                # Extract text from first source document
                answer = result["source_documents"][0].page_content[:200]
            else:
                answer = "Unable to generate a complete response. Please try rephrasing your question."
        
        # Limit to first 3 sentences to prevent rambling, but only if we have multiple sentences
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        if len(sentences) > 3:
            answer = '. '.join(sentences[:3]) + '.'
        elif len(sentences) > 0 and not answer.endswith('.'):
            answer = '. '.join(sentences) + '.'
        
        response = {
            "answer": answer if answer else "Unable to generate a response."
        }

        if return_sources and "source_documents" in result and result["source_documents"]:
            sources = [
                {
                    "content": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
            response["sources"] = sources

        return response
    except IndexError as e:
        # Handle index out of range errors from the model
        return {
            "answer": "Error: The model encountered an issue. Please try a different query.",
            "error": "index_out_of_range"
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "error": "generation_error"
        }


if __name__ == "__main__":
    # Test prompt creation
    prompt = create_rag_prompt()
    print("Default prompt template:")
    print(prompt.template)
