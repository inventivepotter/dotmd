"""LanceDB-backed vector store for chunk-embedding similarity search.

Implements :class:`~dotmd.storage.base.VectorStoreProtocol` using the
`lancedb <https://lancedb.github.io/lancedb/>`_ embedded vector database.
"""

from __future__ import annotations

from pathlib import Path

import lancedb  # type: ignore[import-untyped]

from dotmd.core.models import Chunk


class LanceDBVectorStore:
    """LanceDB implementation of :class:`VectorStoreProtocol`.

    Parameters
    ----------
    db_path:
        Directory path for the LanceDB database files.
    table_name:
        Name of the table used to store chunk embeddings.
    """

    def __init__(self, db_path: Path, table_name: str = "chunks") -> None:
        self._db = lancedb.connect(str(db_path))
        self._table_name = table_name

    # -- mutations ----------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunks with their corresponding embeddings.

        Creates or overwrites the backing table with the provided data.

        Parameters
        ----------
        chunks:
            The chunk objects to store.
        embeddings:
            A parallel list of embedding vectors, one per chunk.
        """
        data = [
            {
                "id": chunk.chunk_id,
                "vector": embedding,
                "chunk_id": chunk.chunk_id,
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self._db.create_table(self._table_name, data, mode="overwrite")

    def delete_all(self) -> None:
        """Remove all vectors from the store."""
        try:
            self._db.drop_table(self._table_name)
        except Exception:  # noqa: BLE001
            # Table may not exist yet; nothing to delete.
            pass

    # -- queries ------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return the *top_k* most similar chunks.

        Parameters
        ----------
        query_embedding:
            The embedding vector to search against.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            ``(chunk_id, score)`` pairs ordered by descending similarity.
            The score is computed as ``1 / (1 + distance)``.
        """
        try:
            table = self._db.open_table(self._table_name)
        except Exception:  # noqa: BLE001
            return []

        results = table.search(query_embedding).limit(top_k).to_list()
        return [
            (row["chunk_id"], 1.0 / (1.0 + row["_distance"]))
            for row in results
        ]

    def count(self) -> int:
        """Return the total number of stored vectors."""
        try:
            table = self._db.open_table(self._table_name)
            return table.count_rows()
        except Exception:  # noqa: BLE001
            return 0
