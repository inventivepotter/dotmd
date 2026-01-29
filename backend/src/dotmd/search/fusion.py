"""Reciprocal Rank Fusion and search-result construction.

This module provides two functions:

- :func:`fuse_results` -- merge ranked lists from multiple search engines
  using Reciprocal Rank Fusion (RRF).
- :func:`build_search_results` -- hydrate fused scores into full
  :class:`SearchResult` objects by looking up chunk metadata.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from dotmd.core.models import SearchResult

if TYPE_CHECKING:
    from dotmd.storage.base import MetadataStoreProtocol


def _extract_best_snippet(text: str, query: str, length: int = 300) -> str:
    """Find the window of *length* chars in *text* with the most query term overlap."""
    if len(text) <= length:
        return text

    query_tokens = set(re.findall(r"\w+", query.lower()))
    if not query_tokens:
        # Fallback to start of text
        return _truncate(text, length)

    # Score each window position (slide by sentences/words for speed)
    words = text.split()
    best_score = -1
    best_start = 0

    # Build character position index for word boundaries
    char_pos = 0
    word_starts: list[int] = []
    for w in words:
        idx = text.index(w, char_pos)
        word_starts.append(idx)
        char_pos = idx + len(w)

    for i, start in enumerate(word_starts):
        end = start + length
        if end > len(text):
            start = max(0, len(text) - length)
            end = len(text)

        window = text[start:end].lower()
        score = sum(1 for t in query_tokens if t in window)

        if score > best_score:
            best_score = score
            best_start = start

        if end >= len(text):
            break

    snippet_text = text[best_start : best_start + length]

    # Add ellipsis indicators
    prefix = "..." if best_start > 0 else ""
    suffix = "..." if best_start + length < len(text) else ""

    # Word-aware truncation at the end
    if suffix:
        last_space = snippet_text.rfind(" ")
        if last_space > len(snippet_text) * 0.8:
            snippet_text = snippet_text[:last_space]

    return prefix + snippet_text + suffix


def _truncate(text: str, length: int) -> str:
    """Word-aware truncation fallback."""
    truncated = text[:length]
    last_space = truncated.rfind(" ")
    if last_space > length * 0.8:
        return truncated[:last_space] + "..."
    return truncated + "..."


# Score field names on SearchResult keyed by canonical engine name.
_ENGINE_SCORE_FIELDS: dict[str, str] = {
    "semantic": "semantic_score",
    "bm25": "bm25_score",
    "graph": "graph_score",
}


def fuse_results(
    ranked_lists: dict[str, list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    For every chunk that appears in at least one list the fused score is
    computed as::

        score = sum(1 / (k + rank_i))

    where *rank_i* is the **1-based** position of the chunk in each
    engine's result list.

    Parameters
    ----------
    ranked_lists:
        A mapping of ``engine_name`` to a list of ``(chunk_id, score)``
        pairs, ordered by descending relevance.
    k:
        The RRF constant (default ``60``).  Higher values dampen the
        influence of top-ranked results.

    Returns
    -------
    list[tuple[str, float]]
        A list of ``(chunk_id, fused_score)`` pairs sorted by
        descending fused score.
    """
    rrf_scores: dict[str, float] = {}

    for _engine, results in ranked_lists.items():
        for rank_0, (chunk_id, _score) in enumerate(results):
            rank = rank_0 + 1  # 1-based
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def build_search_results(
    fused: list[tuple[str, float]],
    per_engine: dict[str, list[tuple[str, float]]],
    metadata_store: MetadataStoreProtocol,
    query: str = "",
    top_k: int = 10,
    snippet_length: int = 300,
) -> list[SearchResult]:
    """Convert fused scores into fully hydrated :class:`SearchResult` objects.

    For each of the *top_k* fused results the corresponding chunk is
    looked up in *metadata_store* to populate the heading path, snippet,
    and per-engine scores.

    Parameters
    ----------
    fused:
        Output of :func:`fuse_results`.
    per_engine:
        The same ``ranked_lists`` dict passed to :func:`fuse_results`,
        used to attribute per-engine scores.
    metadata_store:
        A store satisfying :class:`MetadataStoreProtocol`.
    top_k:
        Maximum number of results to return.
    snippet_length:
        Maximum length for the text snippet (default 300 characters).
        Truncation is word-aware to avoid cutting mid-word.

    Returns
    -------
    list[SearchResult]
        Up to *top_k* search results, ordered by descending fused score.
    """
    # Pre-index per-engine scores for O(1) lookup.
    engine_scores: dict[str, dict[str, float]] = {}
    for engine, results in per_engine.items():
        engine_scores[engine] = {cid: score for cid, score in results}

    top_ids = [cid for cid, _ in fused[:top_k]]
    chunks_by_id = {c.chunk_id: c for c in metadata_store.get_chunks(top_ids)}

    results: list[SearchResult] = []
    for chunk_id, fused_score in fused[:top_k]:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue

        heading_path = " > ".join(chunk.heading_hierarchy) if chunk.heading_hierarchy else ""

        snippet = _extract_best_snippet(chunk.text, query, snippet_length)

        # Determine which engines matched and their individual scores.
        matched_engines: list[str] = []
        per_engine_kwargs: dict[str, float | None] = {}
        for engine_name, field_name in _ENGINE_SCORE_FIELDS.items():
            score = engine_scores.get(engine_name, {}).get(chunk_id)
            per_engine_kwargs[field_name] = score
            if score is not None:
                matched_engines.append(engine_name)

        results.append(
            SearchResult(
                chunk_id=chunk_id,
                file_path=chunk.file_path,
                heading_path=heading_path,
                snippet=snippet,
                fused_score=fused_score,
                matched_engines=sorted(matched_engines),
                **per_engine_kwargs,  # type: ignore[arg-type]
            )
        )

    return results
