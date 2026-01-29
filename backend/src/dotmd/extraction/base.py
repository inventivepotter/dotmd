"""Extractor protocol defining the interface for all extraction strategies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dotmd.core.models import Chunk, ExtractionResult


@runtime_checkable
class ExtractorProtocol(Protocol):
    """Protocol that all extractors must implement."""

    def extract(self, chunks: list[Chunk]) -> ExtractionResult:
        """Extract entities and relations from chunks.

        Args:
            chunks: List of document chunks to process.

        Returns:
            An ``ExtractionResult`` containing discovered entities and relations.
        """
        ...
