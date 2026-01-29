"""Semantic (dense vector) search engine for dotMD.

Uses a ``SentenceTransformer`` model to encode queries into dense
vectors, then delegates similarity search to a
:class:`~dotmd.storage.base.VectorStoreProtocol` backend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from dotmd.storage.base import VectorStoreProtocol

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticSearchEngine:
    """Dense-vector search engine backed by a ``SentenceTransformer`` model.

    The underlying model is **lazy-loaded** on the first call to
    :meth:`search`, :meth:`encode`, or :meth:`encode_batch` so that
    importing the module remains lightweight.

    Parameters
    ----------
    vector_store:
        A vector store that satisfies :class:`VectorStoreProtocol`.
    model_name:
        HuggingFace model identifier for the sentence-transformer.
    """

    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self._vector_store = vector_store
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> SentenceTransformer:
        """Load the ``SentenceTransformer`` model if not already loaded.

        Returns
        -------
        SentenceTransformer
            The loaded model instance.
        """
        if self._model is None:
            logger.info("Loading SentenceTransformer model: %s", self._model_name)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[float]:
        """Encode a single text string into a dense vector.

        This method is also used by the indexer when building the
        vector store.

        Parameters
        ----------
        text:
            The text to encode.

        Returns
        -------
        list[float]
            The embedding vector.
        """
        model = self._load_model()
        embedding = model.encode(text, show_progress_bar=False)
        return embedding.tolist()  # type: ignore[union-attr]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into dense vectors.

        Parameters
        ----------
        texts:
            The texts to encode.

        Returns
        -------
        list[list[float]]
            A list of embedding vectors, one per input text.
        """
        if not texts:
            return []
        model = self._load_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embeddings]  # type: ignore[union-attr]

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Encode *query* and return the most similar chunks.

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
            descending similarity.
        """
        query_embedding = self.encode(query)
        return self._vector_store.search(query_embedding, top_k=top_k)
