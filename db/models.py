"""
Dataclass models mirroring the SQL tables.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID
import datetime


@dataclass
class Document:
    title: str
    content: str
    source: Optional[str] = None
    doc_type: str = "text"
    metadata: dict = field(default_factory=dict)
    id: Optional[UUID] = None
    created_at: Optional[datetime.datetime] = None


@dataclass
class Chunk:
    document_id: UUID
    chunk_index: int
    content: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    token_count: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    id: Optional[UUID] = None
    created_at: Optional[datetime.datetime] = None


@dataclass
class Embedding:
    chunk_id: UUID
    embedding: list[float]
    model_name: str = "all-MiniLM-L6-v2"
    id: Optional[UUID] = None


@dataclass
class GraphNode:
    label: str
    node_type: str
    source_chunk_id: Optional[UUID] = None
    properties: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    id: Optional[UUID] = None


@dataclass
class RetrievalResult:
    chunk_id: UUID
    document_id: UUID
    content: str
    similarity: float
    source: str = "vector"  # "vector" | "graph"


@dataclass
class GraphEdge:
    source_node_id: UUID
    target_node_id: UUID
    relation_type: str
    weight: float = 1.0
    source_chunk_id: Optional[UUID] = None
    properties: dict = field(default_factory=dict)
    id: Optional[UUID] = None
