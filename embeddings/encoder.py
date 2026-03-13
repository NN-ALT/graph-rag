"""
Sentence-transformers wrapper.
Model is loaded once on first use and cached for offline reuse.

First run: downloads model to ~/.cache/torch/sentence_transformers/
All subsequent runs: fully offline.
"""

from __future__ import annotations
from sentence_transformers import SentenceTransformer
from config import settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[encoder] Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        print(f"[encoder] Model ready (dim={settings.embedding_dim})")
    return _model


def encode(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=len(texts) > 10, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


def encode_one(text: str) -> list[float]:
    return encode([text])[0]
