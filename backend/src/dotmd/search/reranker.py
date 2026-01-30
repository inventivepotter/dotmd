"""Cross-encoder reranking for search results.

Provides a thin wrapper around a ``sentence_transformers.CrossEncoder``
model that rescores ``(query, chunk_text)`` pairs and returns the top-k
results sorted by descending relevance.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dotmd.storage.base import MetadataStoreProtocol

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker with lazy model loading and length penalty.

    The underlying ``CrossEncoder`` is instantiated on the first call to
    :meth:`rerank` so that import time stays fast and GPU/CPU resources
    are only consumed when actually needed.

    Optionally applies a length penalty to downrank very short chunks
    (e.g., navigation tables) that may be keyword-dense but lack content.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for the cross-encoder.
    length_penalty:
        If True, apply a penalty to chunks shorter than *min_length*.
    min_length:
        Minimum character length below which the penalty is applied.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        length_penalty: bool = True,
        min_length: int = 100,
        score_threshold: float = -8.0,
    ) -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._length_penalty = length_penalty
        self._min_length = min_length
        self._score_threshold = score_threshold

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Load the cross-encoder model on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

            logger.info("Loading cross-encoder model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunk_ids: list[str],
        metadata_store: MetadataStoreProtocol,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Rerank *chunk_ids* against *query* using a cross-encoder.

        Each chunk's text is retrieved from *metadata_store*, paired with
        the query, and scored by the cross-encoder.  The results are
        returned in descending score order, truncated to *top_k*.

        Parameters
        ----------
        query:
            The user query string.
        chunk_ids:
            Chunk identifiers to rerank.
        metadata_store:
            A store satisfying :class:`MetadataStoreProtocol` used to
            look up chunk text.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            Up to *top_k* ``(chunk_id, score)`` pairs sorted by
            descending cross-encoder score.
        """
        if not chunk_ids:
            return []

        chunks = metadata_store.get_chunks(chunk_ids)
        if not chunks:
            return []

        # Preserve ordering alignment between ids and texts.
        id_text_pairs: list[tuple[str, str]] = [
            (chunk.chunk_id, chunk.text) for chunk in chunks
        ]

        model = self._load_model()
        pairs = [(query, text) for _, text in id_text_pairs]
        scores: list[float] = model.predict(pairs).tolist()  # type: ignore[union-attr]

        # Apply length penalty to short chunks if enabled
        if self._length_penalty:
            adjusted_scores = []
            for score, (cid, text) in zip(scores, id_text_pairs):
                text_length = len(text)
                if text_length < self._min_length:
                    # Penalty factor: 0.5 at length=0, 1.0 at min_length
                    penalty_factor = 0.5 + 0.5 * (text_length / self._min_length)
                    score = score * penalty_factor
                adjusted_scores.append(score)
            scores = adjusted_scores

        scored = [
            (cid, float(score))
            for (cid, _text), score in zip(id_text_pairs, scores)
            if score >= self._score_threshold
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
