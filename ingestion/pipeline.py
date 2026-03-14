"""
Ingestion pipeline: load -> chunk -> embed -> store -> build graph
"""

from __future__ import annotations
import logging
from ingestion.loader import load_document
from ingestion.chunker import chunk_document
from embeddings.encoder import encode
from db.connection import get_conn
from db import queries
from db.models import Embedding
from config import settings
from graph.builder import build_graph_from_chunks
from graph.store import invalidate_graph_cache

log = logging.getLogger(__name__)


def ingest(path: str, strategy: str = "sentence") -> dict:
    log.info("Loading: %s", path)
    doc = load_document(path)

    with get_conn() as conn:
        doc_id = queries.insert_document(conn, doc)
        log.info("Document stored: %s", doc_id)

        chunks = chunk_document(
            doc, doc_id,
            strategy=strategy,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        log.info("Created %d chunks", len(chunks))

        chunk_ids = []
        for chunk in chunks:
            cid = queries.insert_chunk(conn, chunk)
            chunk.id = cid
            chunk_ids.append(cid)

        texts = [c.content for c in chunks]
        vectors = encode(texts)
        log.info("Embedded %d chunks", len(vectors))

        for chunk, vector in zip(chunks, vectors):
            emb = Embedding(
                chunk_id=chunk.id,
                embedding=vector,
                model_name=settings.embedding_model,
            )
            queries.insert_embedding(conn, emb)

    log.info("Stored to database. Building graph...")
    build_graph_from_chunks(chunks)
    invalidate_graph_cache()

    return {
        "document_id": str(doc_id),
        "chunks": len(chunks),
        "title": doc.title,
    }
