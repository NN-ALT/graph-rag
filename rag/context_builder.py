"""
Assembles retrieved chunks into a prompt context string.
"""

from __future__ import annotations

_CHARS_PER_TOKEN = 4


def build_context(results: list[dict], max_tokens: int = 3000) -> str:
    max_chars = max_tokens * _CHARS_PER_TOKEN
    sorted_results = sorted(results, key=lambda r: r.get("similarity", 0), reverse=True)
    sections: list[str] = []
    total_chars = 0
    for i, r in enumerate(sorted_results):
        content = r.get("content", "").strip()
        if not content:
            continue
        sim = r.get("similarity", 0)
        sim_str = f" (similarity: {sim:.2f})" if sim > 0 else " (graph-related)"
        header = f"[Context {i + 1}{sim_str}]"
        block = f"{header}\n{content}"
        if total_chars + len(block) > max_chars:
            break
        sections.append(block)
        total_chars += len(block)
    return "\n\n".join(sections)
