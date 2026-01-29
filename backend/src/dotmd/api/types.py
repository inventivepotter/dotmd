"""Public type re-exports for the dotMD API layer.

Consumers of the API can import commonly used types from this module
instead of reaching into ``dotmd.core.models`` directly.
"""

from __future__ import annotations

from dotmd.core.models import Chunk, ExpandedQuery, IndexStats, SearchResult

__all__ = [
    "Chunk",
    "ExpandedQuery",
    "IndexStats",
    "SearchResult",
]
