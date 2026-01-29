"""Pydantic domain models for dotMD."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, computed_field


class FileInfo(BaseModel):
    """Metadata about a discovered markdown file."""

    path: Path
    title: str
    last_modified: datetime
    size_bytes: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def checksum(self) -> str:
        return hashlib.md5(self.path.read_bytes()).hexdigest()


class Chunk(BaseModel):
    """A section of a markdown file after chunking."""

    chunk_id: str
    file_path: Path
    heading_hierarchy: list[str] = Field(default_factory=list)
    level: int = 0
    text: str
    chunk_index: int
    char_offset: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def heading(self) -> str:
        return self.heading_hierarchy[-1] if self.heading_hierarchy else ""


class Entity(BaseModel):
    """A named entity extracted from a chunk."""

    name: str
    type: str
    source: str  # "structural", "ner"
    chunk_ids: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    """A relation between two entities or nodes."""

    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    properties: dict[str, str] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Output from an extractor: entities and relations found in chunks."""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class ExpandedQuery(BaseModel):
    """A query after expansion."""

    original: str
    expanded_terms: list[str] = Field(default_factory=list)
    expanded_text: str = ""


class SearchResult(BaseModel):
    """A single search result after fusion and optional reranking."""

    chunk_id: str
    file_path: Path
    heading_path: str
    snippet: str
    fused_score: float
    semantic_score: float | None = None
    bm25_score: float | None = None
    graph_score: float | None = None
    matched_engines: list[str] = Field(default_factory=list)


class IndexStats(BaseModel):
    """Summary statistics about the current index."""

    total_files: int = 0
    total_chunks: int = 0
    total_entities: int = 0
    total_edges: int = 0
    last_indexed: datetime | None = None
