"""
Pure vector similarity search using pgvector.
"""

from __future__ import annotations
from embeddings.encoder import encode_one
from db.connection import get_conn
from db import queries
from db.models import RetrievalResult


def search(query: str, k: int = 5, min_similarity: float = 0.4) -> list[RetrievalResult]:
    """Returns top-k chunks by cosine similarity to the query."""
    query_vec = encode_one(query)
    with get_conn() as conn:
        return queries.match_chunks(conn, query_vec, match_count=k, min_similarity=min_similarity)
