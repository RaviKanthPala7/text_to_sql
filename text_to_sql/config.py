"""
Configuration utilities for the Text-to-SQL project.

Reads settings from environment variables (optionally via a .env file)
so that secrets are not hard-coded in source code.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    name: str


@dataclass
class LLMConfig:
    google_model: str
    groq_model: str


def get_database_config() -> DatabaseConfig:
    """Return database configuration loaded from environment variables."""
    return DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", "root"),
        name=os.getenv("DB_NAME", "text_to_sql"),
    )


def get_llm_config() -> LLMConfig:
    """Return LLM configuration loaded from environment variables."""
    return LLMConfig(
        google_model=os.getenv("GOOGLE_MODEL_NAME", "gemini-2.5-flash"),
        groq_model=os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
    )


def get_google_api_key() -> str:
    """Return Google API key for Gemini models."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    return api_key


def get_groq_api_key() -> str:
    """Return Groq API key for evaluator models."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return api_key


def build_mysql_uri(config: DatabaseConfig) -> str:
    """Construct the SQLAlchemy MySQL URI from the database configuration.
    When DB_HOST starts with /cloudsql/, builds a URI that uses the Unix socket
    (for Cloud SQL Auth Proxy on Cloud Run).
    """
    if config.host.startswith("/cloudsql/"):
        # Cloud SQL Auth Proxy: connect via Unix socket
        from urllib.parse import quote_plus
        user = quote_plus(config.user)
        password = quote_plus(config.password)
        return (
            f"mysql+pymysql://{user}:{password}@/{config.name}"
            f"?unix_socket={config.host}"
        )
    return (
        f"mysql+pymysql://{config.user}:{config.password}"
        f"@{config.host}:{config.port}/{config.name}"
    )

