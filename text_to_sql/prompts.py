"""
Prompt templates for Text-to-SQL chains.
"""

from langchain_core.prompts import ChatPromptTemplate


SQL_PROMPT_TEMPLATE = """
Based on the table schema below, write a SQL query that would answer the user's question.
Remember: Only provide the SQL query, do not include anything else.
Provide the SQL query in a single line; do not add line breaks.

Use the exact table and column names from the schema, including exact spelling and casing.
Table names must be lowercase (e.g. customers, products, regions, 2017_budgets, sales_order, state_regions).

Table schema:
{schema}

Question:
{question}

SQL Query:
"""


INTENT_VALIDATION_TEMPLATE = """You are a guardrail for a Text-to-SQL application. Analyze the user's question and decide if it is allowed.

Allowed: The user is asking a read-only question about data in the database (e.g. list, show, count, find, what is). Only SELECT-style queries are permitted.

Not allowed:
- The user wants to delete, drop, truncate, update, insert, or alter data or schema (e.g. "delete rows", "drop table", "remove data").
- The question is harmful, violent, offensive, or inappropriate.
- The question is not related to querying the database at all (e.g. general chat, off-topic).

Respond with exactly one of these two formats:
- If allowed: ALLOWED
- If not allowed: NOT_ALLOWED: <one short sentence explaining why this is not allowed, in polite natural language for the user>

User question:
{question}

Your response:"""


def get_sql_prompt() -> ChatPromptTemplate:
    """Return the chat prompt template for SQL generation."""
    return ChatPromptTemplate.from_template(SQL_PROMPT_TEMPLATE.strip())


def get_intent_validation_prompt() -> ChatPromptTemplate:
    """Return the prompt template for query intent validation (guardrail)."""
    return ChatPromptTemplate.from_template(INTENT_VALIDATION_TEMPLATE.strip())

