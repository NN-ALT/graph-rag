"""
Builds graph_nodes and graph_edges from a list of Chunks.
"""

from __future__ import annotations
from db.models import Chunk, GraphEdge
from db.connection import get_conn
from db import queries
from graph.extractor import extract_entities_and_relations


def build_graph_from_chunks(chunks: list[Chunk]) -> dict:
    node_count = 0
    edge_count = 0

    with get_conn() as conn:
        for chunk in chunks:
            if not chunk.id:
                continue

            nodes, edges = extract_entities_and_relations(chunk)

            # Upsert nodes and collect label→id mapping
            label_to_id: dict[str, object] = {}
            for node in nodes:
                node_id = queries.upsert_graph_node(conn, node)
                label_to_id[node.label.lower()] = node_id
                node_count += 1

            # Upsert edges — resolve label references to real IDs
            for edge in edges:
                src_label = edge.properties.pop("_src_label", None)
                tgt_label = edge.properties.pop("_tgt_label", None)

                src_id = label_to_id.get(src_label.lower() if src_label else "")
                tgt_id = label_to_id.get(tgt_label.lower() if tgt_label else "")

                if not src_id or not tgt_id:
                    continue

                real_edge = GraphEdge(
                    source_node_id=src_id,
                    target_node_id=tgt_id,
                    relation_type=edge.relation_type,
                    weight=edge.weight,
                    source_chunk_id=edge.source_chunk_id,
                    properties=edge.properties,
                )
                queries.upsert_graph_edge(conn, real_edge)
                edge_count += 1

    print(f"[graph] Upserted {node_count} nodes, {edge_count} edges")
    return {"nodes": node_count, "edges": edge_count}


def build_graph_from_all_documents() -> dict:
    """Rebuild graph from all chunks already in the database."""
    with get_conn() as conn:
        docs = queries.get_all_documents(conn)
        all_chunks = []
        for doc in docs:
            chunks = queries.get_chunks_by_document(conn, doc.id)
            all_chunks.extend(chunks)

    print(f"[graph] Processing {len(all_chunks)} chunks across {len(docs)} documents")
    return build_graph_from_chunks(all_chunks)
