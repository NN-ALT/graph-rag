"""
Entity and relation extraction from text chunks.
Uses spaCy NER. Falls back to noun-phrase regex if spaCy model unavailable.
"""

from __future__ import annotations
from db.models import Chunk, GraphNode, GraphEdge
from itertools import combinations

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("[extractor] spaCy model not found. Run: py -m spacy download en_core_web_sm")
            print("[extractor] Falling back to noun-phrase regex extraction.")
            _nlp = "fallback"
    return _nlp


def extract_entities_and_relations(
    chunk: Chunk,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    nlp = _get_nlp()

    if nlp == "fallback":
        return _regex_extract(chunk)

    doc = nlp(chunk.content)
    entities = []

    for ent in doc.ents:
        label = ent.text.strip()
        if len(label) < 2:
            continue
        node = GraphNode(
            label=label,
            node_type=ent.label_.lower(),
            source_chunk_id=chunk.id,
            properties={"spacy_label": ent.label_},
        )
        entities.append(node)

    for nc in doc.noun_chunks:
        label = nc.root.text.strip()
        if len(label) < 2 or label.lower() in {"i", "we", "they", "it", "this", "that"}:
            continue
        node = GraphNode(
            label=label,
            node_type="concept",
            source_chunk_id=chunk.id,
        )
        entities.append(node)

    seen: dict[str, GraphNode] = {}
    for node in entities:
        key = node.label.lower()
        if key not in seen:
            seen[key] = node
    unique_entities = list(seen.values())

    edges: list[GraphEdge] = []
    for a, b in combinations(unique_entities, 2):
        edge = GraphEdge(
            source_node_id=None,
            target_node_id=None,
            relation_type="co_occurs",
            weight=1.0,
            source_chunk_id=chunk.id,
            properties={"_src_label": a.label, "_tgt_label": b.label},
        )
        edges.append(edge)

    return unique_entities, edges


def _regex_extract(chunk: Chunk) -> tuple[list[GraphNode], list[GraphEdge]]:
    import re
    pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
    matches = re.findall(pattern, chunk.content)
    seen: dict[str, GraphNode] = {}
    for m in matches:
        key = m.lower()
        if key not in seen and len(m) > 2:
            seen[key] = GraphNode(
                label=m,
                node_type="concept",
                source_chunk_id=chunk.id,
            )
    unique = list(seen.values())
    edges = []
    for a, b in combinations(unique, 2):
        edges.append(GraphEdge(
            source_node_id=None,
            target_node_id=None,
            relation_type="co_occurs",
            weight=1.0,
            source_chunk_id=chunk.id,
            properties={"_src_label": a.label, "_tgt_label": b.label},
        ))
    return unique, edges
