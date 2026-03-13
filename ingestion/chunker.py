"""
Text chunking strategies.
"""

from __future__ import annotations
from db.models import Document, Chunk
from uuid import UUID


def chunk_document(
    doc: Document,
    doc_id: UUID,
    strategy: str = "sentence",
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    if strategy == "sentence":
        return _sentence_chunks(doc, doc_id, chunk_size, overlap)
    return _fixed_chunks(doc, doc_id, chunk_size, overlap)


def _sentence_chunks(doc: Document, doc_id: UUID, chunk_size: int, overlap: int) -> list[Chunk]:
    import nltk

    try:
        sentences = nltk.sent_tokenize(doc.content)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(doc.content)

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    chunk_index = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current:
            text = " ".join(current)
            chunks.append(_make_chunk(doc_id, chunk_index, text, doc.content))
            chunk_index += 1
            # Keep overlap: retain last N chars worth of sentences
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current = overlap_sentences
            current_len = overlap_len
        current.append(sent)
        current_len += sent_len

    if current:
        text = " ".join(current)
        chunks.append(_make_chunk(doc_id, chunk_index, text, doc.content))

    return chunks


def _fixed_chunks(doc: Document, doc_id: UUID, chunk_size: int, overlap: int) -> list[Chunk]:
    text = doc.content
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append(Chunk(
            document_id=doc_id,
            chunk_index=chunk_index,
            content=chunk_text,
            char_start=start,
            char_end=end,
            token_count=len(chunk_text.split()),
        ))
        chunk_index += 1
        start = end - overlap if end < len(text) else len(text)

    return chunks


def _make_chunk(doc_id: UUID, index: int, text: str, full_text: str) -> Chunk:
    char_start = full_text.find(text[:50])  # approximate offset
    return Chunk(
        document_id=doc_id,
        chunk_index=index,
        content=text,
        char_start=char_start if char_start >= 0 else None,
        char_end=char_start + len(text) if char_start >= 0 else None,
        token_count=len(text.split()),
    )
