"""
Syncs graph between Postgres and an in-memory networkx DiGraph.
"""

from __future__ import annotations
import networkx as nx
from db.connection import get_conn
from db import queries


def load_graph_from_db() -> nx.DiGraph:
    with get_conn() as conn:
        nodes = queries.get_all_graph_nodes(conn)
        edges = queries.get_all_graph_edges(conn)

    G = nx.DiGraph()

    for n in nodes:
        G.add_node(
            str(n["id"]),
            label=n["label"],
            node_type=n["node_type"],
            source_chunk_id=str(n["source_chunk_id"]) if n["source_chunk_id"] else None,
            properties=n["properties"] or {},
        )

    for e in edges:
        G.add_edge(
            str(e["source_node_id"]),
            str(e["target_node_id"]),
            relation_type=e["relation_type"],
            weight=e["weight"],
            source_chunk_id=str(e["source_chunk_id"]) if e["source_chunk_id"] else None,
        )

    return G
