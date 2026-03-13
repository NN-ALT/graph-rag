"""
Hybrid retrieval: vector search seeds -> graph expansion -> fetch + deduplicate.
"""

from __future__ import annotations
from db.connection import get_conn
from db import queries
from retrieval.vector_search import search as vector_search
from retrieval.graph_search import expand_with_graph


def retrieve(query: str, k: int = 5, graph_hops: int = 1, min_similarity: float = 0.4) -> list[dict]:
    vector_results = vector_search(query, k=k, min_similarity=min_similarity)
    if not vector_results:
        return []
    seed_ids = [str(r["chunk_id"]) for r in vector_results]
    seed_scores = {str(r["chunk_id"]): r["similarity"] for r in vector_results}
    expanded_ids = expand_with_graph(seed_ids, hops=graph_hops)
    new_ids = [cid for cid in expanded_ids if cid not in seed_scores]
    expanded_chunks = []
    if new_ids:
        with get_conn() as conn:
            expanded_chunks = queries.get_chunks_by_ids(conn, new_ids)
    combined = list(vector_results)
    for chunk in expanded_chunks:
        combined.append({"chunk_id": chunk.id, "document_id": chunk.document_id, "content": chunk.content, "similarity": 0.0, "source": "graph"})
    for r in combined:
        if "source" not in r:
            r["source"] = "vector"
    return combined
