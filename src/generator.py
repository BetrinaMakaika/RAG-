"""
LLM Generator Module
====================
Students: customize the prompt and LLM configuration.
"""

from typing import List, Optional, Dict
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


def get_llm(
    provider: str = "huggingface",
    model_name: str = "gpt2",
    temperature: float = 0.7,
    **kwargs
):
    """
    Get an LLM for generation.

    modify this:
    - Choose appropriate model (gpt2, distilgpt2, etc.)
    - Adjust temperature for creativity vs accuracy
    - Add API key configurations

    Recommended local models (HuggingFace):
    - gpt2 (small, fast, default)
    - distilgpt2 (very small)
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
                    "temperature": temperature,
                    "max_new_tokens": 128,
                    "do_sample": False,
                    "pad_token_id": 50256
                }
            )
        except Exception as e:
            # Fallback with simpler settings
            return HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text-generation",
                model_kwargs={},
                pipeline_kwargs={
                    "max_new_tokens": 64,
                    "do_sample": False,
                    "pad_token_id": 50256
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
        template = """Context: {context}

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

    # Use chain_type_kwargs to pass the prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
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

        response = {
            "answer": answer.strip() if answer else "Unable to generate a response."
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
