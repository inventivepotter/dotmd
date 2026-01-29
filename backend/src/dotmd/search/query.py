"""Query expansion via acronyms, headings, and tag matching.

Expands a raw query string using:
1. Acronym expansion - replaces acronyms with their full forms
2. Structural expansion - adds related headings from metadata
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
    """Expand a user query with acronyms and structurally related heading terms.

    The expander performs two-stage expansion:
    1. Acronym expansion - replaces known acronyms with full forms (fuzzy match)
    2. Structural expansion - adds parent/sibling headings from metadata

    Parameters
    ----------
    metadata_store:
        A store satisfying :class:`MetadataStoreProtocol` used to
        retrieve chunk metadata.
    acronym_dict:
        Optional mapping of acronyms to expansions. If None, acronym
        expansion is skipped.
    fuzzy_threshold:
        Maximum edit distance for fuzzy acronym matching (default 1).
    """

    def __init__(
        self,
        metadata_store: MetadataStoreProtocol,
        acronym_dict: dict[str, list[str]] | None = None,
        fuzzy_threshold: int = 1,
    ) -> None:
        self._store = metadata_store
        self._acronyms = acronym_dict or {}
        self._fuzzy_threshold = fuzzy_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, query: str) -> ExpandedQuery:
        """Expand *query* with acronyms and related heading terms.

        Steps:

        1. Expand acronyms in the query (with fuzzy matching).
        2. Retrieve all chunks from the metadata store.
        3. Build a vocabulary of unique heading strings.
        4. For every heading whose tokens overlap with the query tokens,
           collect its parent and sibling headings as expansion terms.
        5. Return an :class:`ExpandedQuery` combining the original query
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
        # Stage 1: Acronym expansion
        acronym_terms: list[str] = []
        if self._acronyms:
            query, acronym_terms = self._expand_acronyms(query)

        query_tokens = _tokenise(query)
        if not query_tokens:
            return ExpandedQuery(
                original=query,
                expanded_terms=acronym_terms,
                expanded_text=query,
            )

        # Stage 2: Structural expansion
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
        structural_terms: set[str] = set()
        for heading in matched_headings:
            for hierarchy in heading_hierarchies[heading]:
                for h in hierarchy:
                    if h != heading:
                        structural_terms.add(h)

        # Combine acronym and structural terms
        all_terms = acronym_terms + sorted(structural_terms)
        expanded_text = " ".join([query, *all_terms]) if all_terms else query

        return ExpandedQuery(
            original=query,
            expanded_terms=all_terms,
            expanded_text=expanded_text,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _expand_acronyms(self, query: str) -> tuple[str, list[str]]:
        """Expand acronyms in query with fuzzy matching.

        Parameters
        ----------
        query:
            The original query string.

        Returns
        -------
        tuple[str, list[str]]
            (expanded_query, list_of_expansions_added)
        """
        expansions_added: list[str] = []

        # Find all potential acronyms in query (2+ uppercase letters)
        tokens = query.split()
        expanded_tokens = []

        for token in tokens:
            # Extract just the letters (remove punctuation)
            clean_token = re.sub(r'[^A-Z]', '', token.upper())

            if len(clean_token) >= 2:
                # Try exact match first
                if clean_token in self._acronyms:
                    expanded_tokens.append(token)
                    for expansion in self._acronyms[clean_token]:
                        expanded_tokens.append(expansion)
                        expansions_added.append(expansion)
                    continue

                # Try fuzzy match
                if self._fuzzy_threshold > 0:
                    best_match = None
                    best_distance = self._fuzzy_threshold + 1

                    for known_acronym in self._acronyms:
                        distance = _edit_distance(clean_token, known_acronym)
                        if distance <= self._fuzzy_threshold and distance < best_distance:
                            best_match = known_acronym
                            best_distance = distance

                    if best_match:
                        expanded_tokens.append(token)
                        for expansion in self._acronyms[best_match]:
                            expanded_tokens.append(expansion)
                            expansions_added.append(expansion)
                        continue

            expanded_tokens.append(token)

        expanded_query = " ".join(expanded_tokens)
        return expanded_query, expansions_added


def _edit_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
