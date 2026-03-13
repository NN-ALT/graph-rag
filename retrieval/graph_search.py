"""
Graph-aware retrieval: expand seed chunk IDs via graph traversal.
"""

from __future__ import annotations
from graph.store import load_graph_from_db
from graph.traversal import get_related_chunk_ids


def expand_with_graph(
    seed_chunk_ids: list[str],
    hops: int = 1,
) -> list[str]:
    """
    Load the graph from DB, find all nodes whose source_chunk_id is in
    seed_chunk_ids, traverse hops levels of edges, and return all
    connected chunk IDs (including seeds).
    """
    if not seed_chunk_ids:
        return []

    graph = load_graph_from_db()
    seed_set = set(seed_chunk_ids)

    # Find node IDs whose chunk is in the seed set
    seed_node_ids = [
        nid for nid, data in graph.nodes(data=True)
        if data.get("source_chunk_id") in seed_set
    ]

    expanded_chunks = set(seed_chunk_ids)
    for node_id in seed_node_ids:
        related = get_related_chunk_ids(graph, node_id, hops=hops)
        expanded_chunks.update(related)

    return list(expanded_chunks)
