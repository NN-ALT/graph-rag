"""
Assembles retrieved chunks into a prompt context string.
"""

from __future__ import annotations
from config import settings
from db.models import RetrievalResult


def build_context(
    results: list[RetrievalResult],
    max_tokens: int = 3000,
) -> str:
    """
    Takes hybrid retrieval results and builds a formatted context string.
    Prioritizes higher-similarity chunks. Stops when token budget is reached.
    """
    max_chars = max_tokens * settings.chars_per_token

    sorted_results = sorted(results, key=lambda r: r.similarity, reverse=True)

    sections: list[str] = []
    total_chars = 0
    for i, r in enumerate(sorted_results):
        content = r.content.strip()
        if not content:
            continue

        sim_str = f" (similarity: {r.similarity:.2f})" if r.similarity > 0 else " (graph-related)"
        block = f"[Context {i + 1}{sim_str}]\n{content}"

        if total_chars + len(block) > max_chars:
            break
        sections.append(block)
        total_chars += len(block)
    return "\n\n".join(sections)
