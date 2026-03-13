"""
Graph RAG — MCP Server

Exposes the knowledge base as MCP tools so Claude Code or LM Studio
can retrieve and ingest documents in real-time.

TRANSPORT:
  stdio  (default) — for Claude Code
  http              — for LM Studio or any HTTP MCP client

USAGE:
  stdio:  py mcp_server.py
  http:   py mcp_server.py --http
  http:   py mcp_server.py --http --port 9090

REGISTER WITH CLAUDE CODE (one-time, run from the graph_rag directory):
  claude mcp add graph-rag -s user -- py mcp_server.py

REGISTER WITH LM STUDIO:
  1. Run:  py mcp_server.py --http
  2. In LM Studio -> Settings -> MCP -> Add Server
     URL: http://localhost:8080/mcp
"""

from __future__ import annotations
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from fastmcp import FastMCP

mcp = FastMCP(
    name="graph-rag",
    instructions=(
        "Knowledge base retrieval tools. "
        "Call `retrieve_context` first to get relevant information from the "
        "knowledge base before answering questions about ingested documents. "
        "Use `ingest_document` to add new files to the knowledge base."
    ),
)


def _db_error_hint(e: Exception) -> str:
    return (
        f"Database error: {e}\n"
        "Make sure the Supabase containers are running: docker compose up -d\n"
        "(Run from the graph_rag project directory)"
    )


@mcp.tool()
def retrieve_context(question: str, k: int = 5, graph_hops: int = 1, min_similarity: float = 0.4) -> str:
    """
    Search the knowledge base for information relevant to a question.
    Returns the most relevant text chunks from ingested documents.

    Use this before answering any question that may relate to documents
    in the knowledge base. The returned context should be used to ground your answer.

    Args:
        question:       The question or topic to search for.
        k:              Number of vector-similar chunks to retrieve (default 5).
        graph_hops:     How many graph hops to expand from seed chunks (default 1).
        min_similarity: Minimum cosine similarity threshold, 0.0-1.0 (default 0.4).
    """
    try:
        from retrieval.hybrid import retrieve
        from rag.context_builder import build_context
    except Exception as e:
        return f"Import error: {e}"
    try:
        results = retrieve(question, k=k, graph_hops=graph_hops, min_similarity=min_similarity)
    except Exception as e:
        return _db_error_hint(e)
    if not results:
        return (
            "No relevant content found in the knowledge base for this question.\n"
            "The knowledge base may be empty. Use `ingest_document` to add documents."
        )
    context = build_context(results, max_tokens=4000)
    chunk_count = len(results)
    vector_count = sum(1 for r in results if r.get("source") == "vector")
    graph_count = chunk_count - vector_count
    header = (
        f"Retrieved {chunk_count} chunk(s) from the knowledge base "
        f"({vector_count} by similarity, {graph_count} by graph expansion).\n"
        f"{'=' * 60}\n"
    )
    return header + context


@mcp.tool()
def ingest_document(file_path: str) -> str:
    """
    Add a document to the knowledge base.
    Supported formats: .txt, .md, .pdf
    """
    try:
        from ingestion.pipeline import ingest
    except Exception as e:
        return f"Import error: {e}"
    path = os.path.abspath(file_path)
    if not os.path.exists(path):
        return f"File not found: {path}"
    try:
        result = ingest(path)
        return (
            f"Ingested successfully.\n"
            f"  Title    : {result['title']}\n"
            f"  Chunks   : {result['chunks']}\n"
            f"  Doc ID   : {result['document_id']}"
        )
    except Exception as e:
        return _db_error_hint(e)


@mcp.tool()
def get_knowledge_base_stats() -> str:
    """Return statistics about the knowledge base."""
    try:
        from db.connection import get_conn
        from db.queries import get_stats
    except Exception as e:
        return f"Import error: {e}"
    try:
        with get_conn() as conn:
            stats = get_stats(conn)
    except Exception as e:
        return _db_error_hint(e)
    lines = ["Knowledge base statistics:", ""]
    for key, val in stats.items():
        label = key.replace("_", " ").title()
        lines.append(f"  {label:<20} {val}")
    if stats.get("documents", 0) == 0:
        lines.append("\nThe knowledge base is empty.")
        lines.append("Use `ingest_document` to add documents.")
    return "\n".join(lines)


@mcp.tool()
def build_knowledge_graph() -> str:
    """Rebuild the knowledge graph from all documents currently in the database."""
    try:
        from graph.builder import build_graph_from_all_documents
    except Exception as e:
        return f"Import error: {e}"
    try:
        result = build_graph_from_all_documents()
        return (
            f"Knowledge graph rebuilt.\n"
            f"  Nodes upserted : {result['nodes']}\n"
            f"  Edges upserted : {result['edges']}"
        )
    except Exception as e:
        return _db_error_hint(e)


@mcp.tool()
def search_chunks(query: str, k: int = 8, min_similarity: float = 0.3) -> str:
    """
    Raw vector similarity search — returns matching chunks without graph
    expansion or LLM processing.
    """
    try:
        from retrieval.vector_search import search
    except Exception as e:
        return f"Import error: {e}"
    try:
        results = search(query, k=k, min_similarity=min_similarity)
    except Exception as e:
        return _db_error_hint(e)
    if not results:
        return f"No chunks found matching '{query}' above similarity {min_similarity}."
    lines = [f"Found {len(results)} matching chunk(s):\n"]
    for i, r in enumerate(results, 1):
        sim = r.get("similarity", 0)
        content = r.get("content", "").strip()
        preview = content[:300] + "..." if len(content) > 300 else content
        lines.append(f"[{i}] Similarity: {sim:.3f}")
        lines.append(preview)
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Graph RAG MCP Server")
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    if args.http:
        print(f"[graph-rag MCP] HTTP mode — http://{args.host}:{args.port}/mcp", flush=True)
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
