"""
Ingestion pipeline: load → chunk → embed → store → build graph
"""

from __future__ import annotations
from ingestion.loader import load_document
from ingestion.chunker import chunk_document
from embeddings.encoder import encode
from db.connection import get_conn
from db import queries
from db.models import Embedding
from config import settings
from graph.builder import build_graph_from_chunks


def ingest(path: str, strategy: str = "sentence") -> dict:
    print(f"[ingest] Loading: {path}")
    doc = load_document(path)

    with get_conn() as conn:
        # Store document
        doc_id = queries.insert_document(conn, doc)
        print(f"[ingest] Document stored: {doc_id}")

        # Chunk
        chunks = chunk_document(
            doc, doc_id,
            strategy=strategy,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        print(f"[ingest] Created {len(chunks)} chunks")

        # Store chunks
        chunk_ids = []
        for chunk in chunks:
            cid = queries.insert_chunk(conn, chunk)
            chunk.id = cid
            chunk_ids.append(cid)

        # Embed
        texts = [c.content for c in chunks]
        vectors = encode(texts)
        print(f"[ingest] Embedded {len(vectors)} chunks")

        # Store embeddings
        for chunk, vector in zip(chunks, vectors):
            emb = Embedding(
                chunk_id=chunk.id,
                embedding=vector,
                model_name=settings.embedding_model,
            )
            queries.insert_embedding(conn, emb)

    print("[ingest] Stored to database. Building graph...")
    build_graph_from_chunks(chunks)

    return {
        "document_id": str(doc_id),
        "chunks": len(chunks),
        "title": doc.title,
    }
