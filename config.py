"""
Central configuration — reads from .env file.
All other modules import from here.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


@dataclass
class Settings:
    # Database
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str

    # LLM provider: "lmstudio" (offline) | "claude" (Anthropic API)
    llm_provider: str

    # LM Studio
    lm_studio_url: str
    lm_studio_model: str

    # Claude (Anthropic API)
    anthropic_api_key: str
    claude_model: str

    # Embeddings
    embedding_model: str
    embedding_dim: int

    # Chunking
    chunk_size: int
    chunk_overlap: int

    @property
    def db_dsn(self) -> str:
        return (
            f"host={self.db_host} port={self.db_port} "
            f"dbname={self.db_name} user={self.db_user} "
            f"password={self.db_password}"
        )


def load_settings() -> Settings:
    return Settings(
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=int(os.getenv("DB_PORT", "5432")),
        db_name=os.getenv("DB_NAME", "postgres"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", ""),
        llm_provider=os.getenv("LLM_PROVIDER", "lmstudio"),
        lm_studio_url=os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1"),
        lm_studio_model=os.getenv("LM_STUDIO_MODEL", "local-model"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "384")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
    )


# Module-level singleton
settings = load_settings()
