"""
Core package for Text-to-SQL functionality.

This package exposes high-level helpers for:
- Connecting to the database
- Building LLM models and chains
- Running evaluation
"""

from .db import get_database, get_schema
from .chains import create_sql_chain, generate_sql_query

__all__ = [
    "get_database",
    "get_schema",
    "create_sql_chain",
    "generate_sql_query",
]

