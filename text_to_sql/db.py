"""
Database helpers for Text-to-SQL.
"""

from typing import Any

import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, text

from .config import DatabaseConfig, build_mysql_uri, get_database_config


def get_database(config: DatabaseConfig | None = None) -> SQLDatabase:
    """
    Create and return a SQLDatabase instance using the provided configuration.

    If no configuration is given, it is loaded from environment variables.
    """
    if config is None:
        config = get_database_config()

    uri = build_mysql_uri(config)
    return SQLDatabase.from_uri(uri, sample_rows_in_table_info=2)


def get_schema(db: SQLDatabase) -> str:
    """Return the full table schema for the given database."""
    return db.get_table_info()


def run_query(db: SQLDatabase, sql: str) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a pandas DataFrame.
    
    This provides clean, tabular results like SQL Workbench instead of
    raw string output from SQLDatabase.run().
    """
    # Get the underlying SQLAlchemy engine from the SQLDatabase
    # SQLDatabase stores the engine in _engine attribute
    if hasattr(db, '_engine'):
        engine = db._engine
    else:
        # Fallback: create engine from config
        config = get_database_config()
        uri = build_mysql_uri(config)
        engine = create_engine(uri)
    
    # Execute query and return as DataFrame
    with engine.connect() as connection:
        result = connection.execute(text(sql))
        # Get column names
        columns = list(result.keys())
        # Get all rows
        rows = result.fetchall()
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=columns)
        return df

