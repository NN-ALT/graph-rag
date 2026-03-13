"""
Loads documents from disk into Document dataclasses.
Supports: .txt, .md, .pdf
"""

from __future__ import annotations
import os
from pathlib import Path
from db.models import Document


def load_document(path: str) -> Document:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = p.suffix.lower()
    title = p.stem

    if ext in (".txt", ".md"):
        content = p.read_text(encoding="utf-8", errors="replace")
        doc_type = "markdown" if ext == ".md" else "text"

    elif ext == ".pdf":
        content = _load_pdf(p)
        doc_type = "pdf"

    else:
        content = p.read_text(encoding="utf-8", errors="replace")
        doc_type = "text"

    return Document(
        title=title,
        source=str(p.resolve()),
        content=content,
        doc_type=doc_type,
    )


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"[Page {i + 1}]\n{text.strip()}")
    return "\n\n".join(pages)
