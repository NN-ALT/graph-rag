"""
End-to-end RAG pipeline: query -> retrieve -> augment -> generate.
"""

from __future__ import annotations
from retrieval.hybrid import retrieve
from rag.context_builder import build_context
from llm.client import chat

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the context does not contain enough information, say so clearly.
Do not make up facts. Be concise and accurate."""


def answer(
    query: str,
    k: int = 5,
    graph_hops: int = 1,
    min_similarity: float = 0.4,
    max_context_tokens: int = 3000,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> dict:
    results = retrieve(query, k=k, graph_hops=graph_hops, min_similarity=min_similarity)
    if not results:
        return {"answer": "No relevant information found in the knowledge base.", "context_used": 0, "sources": []}
    context = build_context(results, max_tokens=max_context_tokens)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    response = chat(messages, temperature=temperature, max_tokens=max_tokens)
    return {"answer": response, "context_used": len(results), "sources": [str(r.get("chunk_id", "")) for r in results]}
