"""
Builds graph_nodes and graph_edges from a list of Chunks.
"""

from __future__ import annotations
import logging
from db.models import Chunk, GraphEdge
from db.connection import get_conn
from db import queries
from graph.extractor import extract_entities_and_relations

log = logging.getLogger(__name__)


def build_graph_from_chunks(chunks: list[Chunk]) -> dict:
    node_count = 0
    edge_count = 0
    processed_ids = []

    with get_conn() as conn:
        for chunk in chunks:
            if not chunk.id:
                continue

            nodes, edges = extract_entities_and_relations(chunk)

            label_to_id: dict[str, object] = {}
            for node in nodes:
                node_id = queries.upsert_graph_node(conn, node)
                label_to_id[node.label.lower()] = node_id
                node_count += 1

            for edge in edges:
                src_label = edge.properties.pop("_src_label", None)
                tgt_label = edge.properties.pop("_tgt_label", None)

                src_id = label_to_id.get(src_label.lower() if src_label else "")
                tgt_id = label_to_id.get(tgt_label.lower() if tgt_label else "")

                if not src_id or not tgt_id:
                    log.warning(
                        "Skipping edge '%s' -> '%s' in chunk %s: node ID not found",
                        src_label, tgt_label, chunk.id,
                    )
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

            processed_ids.append(chunk.id)

        if processed_ids:
            queries.mark_chunks_graph_indexed(conn, processed_ids)

    log.info("Upserted %d nodes, %d edges from %d chunks", node_count, edge_count, len(processed_ids))
    return {"nodes": node_count, "edges": edge_count}


def build_graph_from_all_documents(force: bool = False) -> dict:
    """
    Build graph from chunks not yet graph-indexed.

    Args:
        force: If True, resets all index markers and reprocesses every chunk.
               If False (default), only processes chunks added since the last run.
    """
    with get_conn() as conn:
        queries.ensure_graph_indexed_column(conn)
        if force:
            queries.reset_graph_indexed(conn)
            log.info("Force rebuild: cleared graph_indexed_at on all chunks")
        chunks = queries.get_chunks_needing_graph_index(conn)

    if not chunks:
        log.info("All chunks are already graph-indexed. Use --force to rebuild from scratch.")
        return {"nodes": 0, "edges": 0}

    log.info("Processing %d unindexed chunk(s)", len(chunks))
    return build_graph_from_chunks(chunks)
