"""
Assembles retrieved chunks into a prompt context string.
Respects a rough token/character budget to avoid exceeding LLM context.
"""

from __future__ import annotations

# Rough chars-per-token estimate for context budgeting
_CHARS_PER_TOKEN = 4


def build_context(
    results: list[dict],
    max_tokens: int = 3000,
) -> str:
    """
    Takes hybrid retrieval results and builds a formatted context string.
    Prioritizes higher-similarity chunks. Stops when token budget is reached.
    """
    max_chars = max_tokens * _CHARS_PER_TOKEN

    # Sort: vector results first (have similarity > 0), then graph-expanded
    sorted_results = sorted(results, key=lambda r: r.get("similarity", 0), reverse=True)

    sections: list[str] = []
    total_chars = 0

    for i, r in enumerate(sorted_results):
        content = r.get("content", "").strip()
        if not content:
            continue

        source = r.get("source", "vector")
        sim = r.get("similarity", 0)
        sim_str = f" (similarity: {sim:.2f})" if sim > 0 else " (graph-related)"

        header = f"[Context {i + 1}{sim_str}]"
        block = f"{header}\n{content}"

        if total_chars + len(block) > max_chars:
            break

        sections.append(block)
        total_chars += len(block)

    return "\n\n".join(sections)
