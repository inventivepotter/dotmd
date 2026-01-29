"""Query expansion via heading and tag matching.

Expands a raw query string by finding related headings in the
metadata store and adding sibling/parent headings as extra terms.
No embedding-based expansion -- purely structural matching.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from dotmd.core.models import ExpandedQuery

if TYPE_CHECKING:
    from dotmd.storage.base import MetadataStoreProtocol


def _tokenise(text: str) -> set[str]:
    """Lowercase and split *text* into a set of unique alphabetic tokens."""
    return {t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) > 1}


class QueryExpander:
    """Expand a user query with structurally related heading terms.

    The expander scans all chunks stored in the metadata store,
    builds a vocabulary of known headings, and adds parent and sibling
    headings whenever the query matches part of a heading hierarchy.

    Parameters
    ----------
    metadata_store:
        A store satisfying :class:`MetadataStoreProtocol` used to
        retrieve chunk metadata.
    """

    def __init__(self, metadata_store: MetadataStoreProtocol) -> None:
        self._store = metadata_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, query: str) -> ExpandedQuery:
        """Expand *query* with related heading terms.

        Steps:

        1. Retrieve all chunks from the metadata store.
        2. Build a vocabulary of unique heading strings.
        3. For every heading whose tokens overlap with the query tokens,
           collect its parent and sibling headings as expansion terms.
        4. Return an :class:`ExpandedQuery` combining the original query
           with the discovered terms.

        Parameters
        ----------
        query:
            The raw user query string.

        Returns
        -------
        ExpandedQuery
            An object containing the original query, discovered
            expansion terms, and the combined expanded text.
        """
        query_tokens = _tokenise(query)
        if not query_tokens:
            return ExpandedQuery(
                original=query,
                expanded_terms=[],
                expanded_text=query,
            )

        chunks = self._store.get_all_chunks()

        # Map each heading to its full hierarchy so we can find
        # parents/siblings later.
        # heading_text -> set of hierarchies (as tuples) it appears in
        heading_hierarchies: dict[str, list[list[str]]] = {}
        for chunk in chunks:
            for heading in chunk.heading_hierarchy:
                heading_hierarchies.setdefault(heading, []).append(
                    chunk.heading_hierarchy
                )

        # Find headings whose tokens overlap with the query.
        matched_headings: set[str] = set()
        for heading in heading_hierarchies:
            if query_tokens & _tokenise(heading):
                matched_headings.add(heading)

        # Collect parent and sibling headings for every match.
        expanded_terms: set[str] = set()
        for heading in matched_headings:
            for hierarchy in heading_hierarchies[heading]:
                for h in hierarchy:
                    if h != heading:
                        expanded_terms.add(h)

        sorted_terms = sorted(expanded_terms)
        expanded_text = " ".join([query, *sorted_terms]) if sorted_terms else query

        return ExpandedQuery(
            original=query,
            expanded_terms=sorted_terms,
            expanded_text=expanded_text,
        )
