"""
All SQL queries as typed functions.
All DB access goes through here — no raw SQL elsewhere.
"""

from __future__ import annotations
from uuid import UUID
import json
import numpy as np
from db.models import Document, Chunk, Embedding, GraphNode, GraphEdge


# ── Documents ───────────────────────────────────────────────────────────────────────────────

def insert_document(conn, doc: Document) -> UUID:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (title, source, content, doc_type, metadata)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (doc.title, doc.source, doc.content, doc.doc_type, json.dumps(doc.metadata)),
        )
        return cur.fetchone()[0]


def get_all_documents(conn) -> list[Document]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, source, content, doc_type, metadata FROM documents ORDER BY created_at")
        rows = cur.fetchall()
    return [Document(id=r[0], title=r[1], source=r[2], content=r[3], doc_type=r[4], metadata=r[5]) for r in rows]


# ── Chunks ────────────────────────────────────────────────────────────────────────────────

def insert_chunk(conn, chunk: Chunk) -> UUID:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks (document_id, chunk_index, content, token_count, char_start, char_end, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                str(chunk.document_id),
                chunk.chunk_index,
                chunk.content,
                chunk.token_count,
                chunk.char_start,
                chunk.char_end,
                json.dumps(chunk.metadata),
            ),
        )
        return cur.fetchone()[0]


def get_chunks_by_ids(conn, chunk_ids: list[UUID]) -> list[Chunk]:
    if not chunk_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, document_id, chunk_index, content, char_start, char_end, token_count
            FROM chunks
            WHERE id = ANY(%s)
            """,
            ([str(cid) for cid in chunk_ids],),
        )
        rows = cur.fetchall()
    return [
        Chunk(id=r[0], document_id=r[1], chunk_index=r[2], content=r[3],
              char_start=r[4], char_end=r[5], token_count=r[6])
        for r in rows
    ]


def get_chunks_by_document(conn, document_id: UUID) -> list[Chunk]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, document_id, chunk_index, content, char_start, char_end, token_count "
            "FROM chunks WHERE document_id = %s ORDER BY chunk_index",
            (str(document_id),),
        )
        rows = cur.fetchall()
    return [
        Chunk(id=r[0], document_id=r[1], chunk_index=r[2], content=r[3],
              char_start=r[4], char_end=r[5], token_count=r[6])
        for r in rows
    ]


# ── Embeddings ───────────────────────────────────────────────────────────────────────────────

def insert_embedding(conn, emb: Embedding) -> UUID:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO embeddings (chunk_id, model_name, embedding)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (str(emb.chunk_id), emb.model_name, np.array(emb.embedding)),
        )
        return cur.fetchone()[0]


def match_chunks(
    conn,
    query_embedding: list[float],
    match_count: int = 5,
    min_similarity: float = 0.4,
) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id, document_id, content, similarity "
            "FROM match_chunks(%s, %s, %s)",
            (np.array(query_embedding), match_count, min_similarity),
        )
        rows = cur.fetchall()
    return [
        {"chunk_id": r[0], "document_id": r[1], "content": r[2], "similarity": r[3]}
        for r in rows
    ]


# ── Graph Nodes ──────────────────────────────────────────────────────────────────────────────

def upsert_graph_node(conn, node: GraphNode) -> UUID:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO graph_nodes (label, node_type, source_chunk_id, properties, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (label, node_type) DO UPDATE
                SET properties = graph_nodes.properties || EXCLUDED.properties,
                    source_chunk_id = COALESCE(EXCLUDED.source_chunk_id, graph_nodes.source_chunk_id)
            RETURNING id
            """,
            (
                node.label,
                node.node_type,
                str(node.source_chunk_id) if node.source_chunk_id else None,
                json.dumps(node.properties),
                np.array(node.embedding) if node.embedding else None,
            ),
        )
        return cur.fetchone()[0]


def get_all_graph_nodes(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, label, node_type, source_chunk_id, properties FROM graph_nodes")
        rows = cur.fetchall()
    return [{"id": r[0], "label": r[1], "node_type": r[2], "source_chunk_id": r[3], "properties": r[4]} for r in rows]


def get_node_by_label(conn, label: str, node_type: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, label, node_type, source_chunk_id FROM graph_nodes WHERE label = %s AND node_type = %s",
            (label, node_type),
        )
        row = cur.fetchone()
    if row:
        return {"id": row[0], "label": row[1], "node_type": row[2], "source_chunk_id": row[3]}
    return None


# ── Graph Edges ──────────────────────────────────────────────────────────────────────────────

def upsert_graph_edge(conn, edge: GraphEdge) -> UUID:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO graph_edges (source_node_id, target_node_id, relation_type, weight, source_chunk_id, properties)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_node_id, target_node_id, relation_type) DO UPDATE
                SET weight = graph_edges.weight + EXCLUDED.weight
            RETURNING id
            """,
            (
                str(edge.source_node_id),
                str(edge.target_node_id),
                edge.relation_type,
                edge.weight,
                str(edge.source_chunk_id) if edge.source_chunk_id else None,
                json.dumps(edge.properties),
            ),
        )
        return cur.fetchone()[0]


def get_all_graph_edges(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, source_node_id, target_node_id, relation_type, weight, source_chunk_id "
            "FROM graph_edges"
        )
        rows = cur.fetchall()
    return [
        {"id": r[0], "source_node_id": r[1], "target_node_id": r[2],
         "relation_type": r[3], "weight": r[4], "source_chunk_id": r[5]}
        for r in rows
    ]


# ── Stats ──────────────────────────────────────────────────────────────────────────────────

def get_stats(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                (SELECT COUNT(*) FROM documents) AS documents,
                (SELECT COUNT(*) FROM chunks)    AS chunks,
                (SELECT COUNT(*) FROM embeddings) AS embeddings,
                (SELECT COUNT(*) FROM graph_nodes) AS nodes,
                (SELECT COUNT(*) FROM graph_edges) AS edges
        """)
        row = cur.fetchone()
    return {
        "documents": row[0],
        "chunks": row[1],
        "embeddings": row[2],
        "graph_nodes": row[3],
        "graph_edges": row[4],
    }
