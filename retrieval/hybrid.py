"""
Hybrid retrieval: vector search seeds -> graph expansion -> fetch + deduplicate.
"""

from __future__ import annotations
from db.connection import get_conn
from db import queries
from db.models import RetrievalResult
from retrieval.vector_search import search as vector_search
from retrieval.graph_search import expand_with_graph

# Graph-expanded chunks are scored relative to the lowest vector seed score,
# decayed by a fixed penalty to rank them below direct vector matches.
_GRAPH_SCORE_DECAY = 0.1


def retrieve(
    query: str,
    k: int = 5,
    graph_hops: int = 1,
    min_similarity: float = 0.4,
) -> list[RetrievalResult]:
    """
    Full hybrid retrieval pipeline.
    Returns RetrievalResult list sorted by similarity (vector seeds first,
    graph-expanded after).
    """
    # Step 1: vector search for seed chunks
    vector_results = vector_search(query, k=k, min_similarity=min_similarity)
    if not vector_results:
        return []

    seed_ids = [str(r.chunk_id) for r in vector_results]
    seed_scores = {str(r.chunk_id): r.similarity for r in vector_results}

    # Baseline score for graph-expanded results: just below the lowest seed score
    min_seed_score = min(seed_scores.values())
    graph_base_score = max(0.0, min_seed_score - _GRAPH_SCORE_DECAY)

    # Step 2: graph expansion
    expanded_ids = expand_with_graph(seed_ids, hops=graph_hops)
    new_ids = [cid for cid in expanded_ids if cid not in seed_scores]

    # Step 3: fetch expanded chunks from DB and wrap as RetrievalResult
    graph_results: list[RetrievalResult] = []
    if new_ids:
        with get_conn() as conn:
            chunks = queries.get_chunks_by_ids(conn, new_ids)
        graph_results = [
            RetrievalResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                similarity=graph_base_score,
                source="graph",
            )
            for chunk in chunks
        ]

    combined = vector_results + graph_results
    combined.sort(key=lambda r: r.similarity, reverse=True)
    return combined
