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


def get_sql_prompt() -> ChatPromptTemplate:
    """Return the chat prompt template for SQL generation."""
    return ChatPromptTemplate.from_template(SQL_PROMPT_TEMPLATE.strip())

