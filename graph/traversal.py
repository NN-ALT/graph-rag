"""
Graph traversal utilities.
"""

from __future__ import annotations
from uuid import UUID
import networkx as nx


def get_neighborhood(graph: nx.DiGraph, node_id: str, hops: int = 2) -> list[str]:
    """Return all node IDs within `hops` of the given node."""
    ego = nx.ego_graph(graph, node_id, radius=hops, undirected=True)
    return list(ego.nodes())


def get_node_id_by_label(graph: nx.DiGraph, label: str) -> str | None:
    label_lower = label.lower()
    for node_id, data in graph.nodes(data=True):
        if data.get("label", "").lower() == label_lower:
            return node_id
    return None


def get_related_chunk_ids(graph: nx.DiGraph, node_id: str, hops: int = 1) -> list[str]:
    """Return chunk IDs connected to the node's neighborhood."""
    neighborhood = get_neighborhood(graph, node_id, hops)
    chunk_ids = set()

    for nid in neighborhood:
        data = graph.nodes[nid]
        if data.get("source_chunk_id"):
            chunk_ids.add(data["source_chunk_id"])

        for _, _, edata in graph.edges(nid, data=True):
            if edata.get("source_chunk_id"):
                chunk_ids.add(edata["source_chunk_id"])

    return list(chunk_ids)


def find_path(graph: nx.DiGraph, src_label: str, tgt_label: str) -> list[str] | None:
    """Return node labels along the shortest path between two labeled nodes."""
    src_id = get_node_id_by_label(graph, src_label)
    tgt_id = get_node_id_by_label(graph, tgt_label)
    if not src_id or not tgt_id:
        return None
    try:
        path = nx.shortest_path(graph.to_undirected(), src_id, tgt_id)
        return [graph.nodes[n]["label"] for n in path]
    except nx.NetworkXNoPath:
        return None
