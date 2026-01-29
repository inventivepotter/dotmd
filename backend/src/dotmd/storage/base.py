"""Abstract protocol definitions for dotMD storage backends.

This module defines the three storage protocols that all concrete
implementations must satisfy:

- **VectorStoreProtocol** – similarity search over chunk embeddings.
- **GraphStoreProtocol** – knowledge-graph persistence and traversal.
- **MetadataStoreProtocol** – chunk and index-statistics persistence.

Each protocol is marked ``@runtime_checkable`` so that ``isinstance``
checks work at runtime, but the primary enforcement mechanism is
static type-checking (mypy / pyright).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dotmd.core.models import Chunk, Entity, Relation, IndexStats

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol for vector similarity-search backends."""

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert *chunks* with their corresponding *embeddings*.

        Parameters
        ----------
        chunks:
            The chunk objects to store.
        embeddings:
            A parallel list of embedding vectors, one per chunk.
        """
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return the *top_k* most similar chunks.

        Parameters
        ----------
        query_embedding:
            The embedding vector to search against.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            A list of ``(chunk_id, score)`` pairs ordered by
            descending similarity.
        """
        ...

    def delete_all(self) -> None:
        """Remove **all** vectors from the store."""
        ...

    def count(self) -> int:
        """Return the total number of stored vectors."""
        ...


# ---------------------------------------------------------------------------
# Graph store
# ---------------------------------------------------------------------------


@runtime_checkable
class GraphStoreProtocol(Protocol):
    """Protocol for knowledge-graph storage backends."""

    def add_file_node(
        self,
        file_path: str,
        title: str,
        checksum: str,
    ) -> None:
        """Create or update a node representing a source file.

        Parameters
        ----------
        file_path:
            Absolute or workspace-relative path to the file.
        title:
            Human-readable title for the file.
        checksum:
            Content hash used for change detection.
        """
        ...

    def add_section_node(
        self,
        chunk_id: str,
        heading: str,
        level: int,
        file_path: str,
        text_preview: str,
    ) -> None:
        """Create or update a node representing a document section.

        Parameters
        ----------
        chunk_id:
            Unique identifier for the corresponding chunk.
        heading:
            Section heading text.
        level:
            Heading depth (1 = top-level).
        file_path:
            Path of the source file this section belongs to.
        text_preview:
            Short preview of the section body.
        """
        ...

    def add_entity_node(
        self,
        name: str,
        entity_type: str,
        source: str,
    ) -> None:
        """Create or update a named-entity node.

        Parameters
        ----------
        name:
            The canonical entity name.
        entity_type:
            Category of the entity (e.g. ``"PERSON"``, ``"ORG"``).
        source:
            Provenance information (e.g. chunk id or file path).
        """
        ...

    def add_tag_node(self, name: str) -> None:
        """Create or update a tag node.

        Parameters
        ----------
        name:
            The tag label.
        """
        ...

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> None:
        """Create or update a directed edge between two nodes.

        Parameters
        ----------
        source_id:
            Identifier of the source node.
        target_id:
            Identifier of the target node.
        relation_type:
            Semantic label for the relationship.
        weight:
            Edge weight (default ``1.0``).
        """
        ...

    def get_neighbors(
        self,
        node_id: str,
        max_hops: int = 2,
    ) -> list[tuple[str, str, float]]:
        """Retrieve neighbours reachable within *max_hops*.

        Parameters
        ----------
        node_id:
            Starting node identifier.
        max_hops:
            Maximum traversal depth.

        Returns
        -------
        list[tuple[str, str, float]]
            A list of ``(node_id, relation_type, weight)`` tuples.
        """
        ...

    def delete_all(self) -> None:
        """Remove **all** nodes and edges from the graph."""
        ...

    def node_count(self) -> int:
        """Return the total number of nodes in the graph."""
        ...

    def edge_count(self) -> int:
        """Return the total number of edges in the graph."""
        ...


# ---------------------------------------------------------------------------
# Metadata store
# ---------------------------------------------------------------------------


@runtime_checkable
class MetadataStoreProtocol(Protocol):
    """Protocol for chunk metadata and index-statistics persistence."""

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Persist a batch of chunks (insert or update).

        Parameters
        ----------
        chunks:
            The chunk objects to save.
        """
        ...

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a single chunk by its identifier.

        Parameters
        ----------
        chunk_id:
            The unique chunk identifier.

        Returns
        -------
        Chunk | None
            The chunk if found, otherwise ``None``.
        """
        ...

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Retrieve multiple chunks by their identifiers.

        Parameters
        ----------
        chunk_ids:
            A list of chunk identifiers.

        Returns
        -------
        list[Chunk]
            The chunks that were found.  Missing ids are silently
            skipped.
        """
        ...

    def get_all_chunks(self) -> list[Chunk]:
        """Return every chunk currently stored."""
        ...

    def save_stats(self, stats: IndexStats) -> None:
        """Persist index statistics (overwrites previous stats).

        Parameters
        ----------
        stats:
            The statistics snapshot to save.
        """
        ...

    def get_stats(self) -> IndexStats | None:
        """Retrieve the most recent index statistics.

        Returns
        -------
        IndexStats | None
            The statistics if available, otherwise ``None``.
        """
        ...

    def delete_all(self) -> None:
        """Remove **all** chunks and statistics from the store."""
        ...
