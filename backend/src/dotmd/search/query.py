"""Query expansion via acronym matching.

Expands a raw query string by replacing acronyms with their full forms.
"""

from __future__ import annotations

import re

from dotmd.core.models import ExpandedQuery


class QueryExpander:
    """Expand a user query by replacing acronyms with their full forms.

    Parameters
    ----------
    acronym_dict:
        Optional mapping of acronyms to expansions. If None, acronym
        expansion is skipped.
    fuzzy_threshold:
        Maximum edit distance for fuzzy acronym matching (default 1).
    """

    def __init__(
        self,
        acronym_dict: dict[str, list[str]] | None = None,
        fuzzy_threshold: int = 1,
    ) -> None:
        self._acronyms = acronym_dict or {}
        self._fuzzy_threshold = fuzzy_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, query: str) -> ExpandedQuery:
        """Expand acronyms in *query* using fuzzy matching.

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
        acronym_terms: list[str] = []
        if self._acronyms:
            query, acronym_terms = self._expand_acronyms(query)

        expanded_text = " ".join([query, *acronym_terms]) if acronym_terms else query

        return ExpandedQuery(
            original=query,
            expanded_terms=acronym_terms,
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
