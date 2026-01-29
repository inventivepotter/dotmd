"""Search engine protocol definition for dotMD.

All search engines (semantic, BM25, graph) implement
:class:`SearchEngineProtocol` so they can be composed
by the fusion layer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SearchEngineProtocol(Protocol):
    """Protocol that every search engine must satisfy.

    Implementations return ``(chunk_id, score)`` pairs sorted by
    descending relevance score.
    """

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search and return ``(chunk_id, score)`` pairs.

        Parameters
        ----------
        query:
            The natural-language search query.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            A list of ``(chunk_id, score)`` pairs ordered by
            descending relevance.
        """
        ...
