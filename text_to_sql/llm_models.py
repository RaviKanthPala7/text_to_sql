"""
LLM and embedding constructors for Text-to-SQL and evaluation.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from .config import get_google_api_key, get_groq_api_key, get_llm_config


def get_query_llm() -> ChatGoogleGenerativeAI:
    """Return the LLM used to generate SQL queries from natural language."""
    cfg = get_llm_config()
    api_key = get_google_api_key()
    return ChatGoogleGenerativeAI(model=cfg.google_model, api_key=api_key)


def get_evaluator_llm() -> ChatGroq:
    """Return the LLM used for RAGAS-style evaluation."""
    cfg = get_llm_config()
    api_key = get_groq_api_key()
    return ChatGroq(model=cfg.groq_model, api_key=api_key)


def get_evaluator_embeddings() -> HuggingFaceEmbeddings:
    """Return the embeddings model used for evaluation."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

