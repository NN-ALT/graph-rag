"""
Sentence-transformers wrapper.
Model is loaded once on first use and cached for offline reuse.

First run: downloads model to ~/.cache/torch/sentence_transformers/
All subsequent runs: fully offline.
"""

from __future__ import annotations
import functools
import logging
from sentence_transformers import SentenceTransformer
from config import settings

log = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        log.info("Model ready (dim=%d)", settings.embedding_dim)
    return _model


def encode(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=len(texts) > 10, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


@functools.lru_cache(maxsize=256)
def encode_one(text: str) -> list[float]:
    return encode([text])[0]
