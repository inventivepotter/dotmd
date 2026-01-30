"""BM25 (sparse keyword) search engine for dotMD.

Builds and persists a ``BM25Okapi`` index over tokenised chunks,
then uses it to rank documents by lexical relevance at query time.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from dotmd.core.models import Chunk
from dotmd.utils.text import tokenize

logger = logging.getLogger(__name__)


@dataclass
class _BM25Data:
    """Serialisable container for the BM25 index and its chunk-id mapping."""

    bm25: BM25Okapi
    chunk_ids: list[str] = field(default_factory=list)


class BM25SearchEngine:
    """Sparse keyword search engine using BM25Okapi.

    The index is serialised as a pickle file so it can be loaded
    without re-tokenising the entire corpus on every startup.

    Parameters
    ----------
    index_path:
        Filesystem path where the pickle file is (or will be) stored.
    """

    def __init__(self, index_path: Path) -> None:
        self._index_path = index_path
        self._data: _BM25Data | None = None

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[Chunk]) -> None:
        """Tokenise *chunks*, build a BM25 index, and persist to disk.

        Parameters
        ----------
        chunks:
            The chunks to index.  Each chunk's :attr:`text` field is
            tokenised using :func:`dotmd.utils.text.tokenize`.
        """
        if not chunks:
            logger.warning("build_index called with an empty chunk list; skipping.")
            return

        corpus: list[list[str]] = [tokenize(c.text) for c in chunks]
        chunk_ids: list[str] = [c.chunk_id for c in chunks]

        bm25 = BM25Okapi(corpus)
        self._data = _BM25Data(bm25=bm25, chunk_ids=chunk_ids)

        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._index_path.open("wb") as fh:
            pickle.dump(self._data, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            "BM25 index built (%d chunks) and saved to %s",
            len(chunk_ids),
            self._index_path,
        )

    def load_index(self) -> None:
        """Load a previously-built index from disk.

        If the index file does not exist, a warning is logged and the
        engine remains uninitialised (subsequent searches return empty
        results).
        """
        if not self._index_path.exists():
            logger.warning(
                "BM25 index file not found at %s; searches will return empty results.",
                self._index_path,
            )
            return

        with self._index_path.open("rb") as fh:
            self._data = pickle.load(fh)  # noqa: S301

        logger.info(
            "BM25 index loaded from %s (%d chunks)",
            self._index_path,
            len(self._data.chunk_ids) if self._data else 0,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Tokenise *query* and return the top-k BM25 results.

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
            descending BM25 score.  Returns an empty list if the
            index has not been built or loaded.
        """
        if self._data is None:
            logger.debug("BM25 index not loaded; returning empty results.")
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores: np.ndarray = self._data.bm25.get_scores(tokenized_query)

        # Retrieve indices of the top-k scores in descending order.
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: list[tuple[str, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            results.append((self._data.chunk_ids[idx], score))

        return results
