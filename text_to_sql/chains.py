"""
LangChain chains for Text-to-SQL.
"""

import re
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.chat_models import BaseChatModel

from .db import get_schema
from .prompts import get_intent_validation_prompt, get_sql_prompt


def create_intent_validation_chain(llm: BaseChatModel):
    """
    Create a chain that analyzes a user question and returns ALLOWED or NOT_ALLOWED with a reason.
    Used as a guardrail before generating SQL.
    """
    prompt = get_intent_validation_prompt()
    chain = prompt | llm | StrOutputParser()
    return chain


def validate_query_intent(chain, question: str) -> tuple[bool, str | None]:
    """
    Run the intent validation chain. Returns (allowed, message).
    - If allowed: (True, None).
    - If not allowed: (False, reason_string) where reason_string is the natural-language explanation.
    """
    response = chain.invoke({"question": question}).strip()
    if response.upper().startswith("ALLOWED"):
        return True, None
    if response.upper().startswith("NOT_ALLOWED"):
        reason = response.split(":", 1)[-1].strip() if ":" in response else response
        return False, reason or "This type of request is not allowed."
    # Fallback: treat unclear response as not allowed for safety
    return False, "Unable to validate this request. Please ask a read-only question about the database."


def create_sql_chain(db, llm: BaseChatModel):
    """
    Create a chain that takes a question and returns a model-generated SQL query.
    """
    prompt = get_sql_prompt()

    chain = (
        RunnablePassthrough.assign(schema=lambda _: get_schema(db))
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    return chain


def extract_sql_from_response(response: str) -> str:
    """
    Extract a SQL query from a model response.

    Supports responses that optionally wrap the SQL in ```sql ... ``` fences.
    """
    match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()


def generate_sql_query(chain, question: str) -> str:
    """
    Run the SQL chain for a single question and return a clean SQL query string.
    """
    resp = chain.invoke({"question": question})
    return extract_sql_from_response(resp)


def batch_generate_sql_queries(chain, questions: list[str]) -> list[str]:
    """
    Generate SQL queries for a batch of questions.
    """
    results: list[str] = []
    for q in questions:
        resp = chain.invoke({"question": q})
        results.append(extract_sql_from_response(resp))
    return results

