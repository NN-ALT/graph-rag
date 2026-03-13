"""
Pure vector similarity search using pgvector.
"""

from __future__ import annotations
from embeddings.encoder import encode_one
from db.connection import get_conn
from db import queries


def search(query: str, k: int = 5, min_similarity: float = 0.4) -> list[dict]:
    query_vec = encode_one(query)
    with get_conn() as conn:
        results = queries.match_chunks(conn, query_vec, match_count=k, min_similarity=min_similarity)
    return results
