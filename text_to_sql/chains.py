"""
LangChain chains for Text-to-SQL.
"""

import re
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.chat_models import BaseChatModel

from .db import get_schema
from .prompts import get_sql_prompt


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

