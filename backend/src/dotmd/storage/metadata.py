"""SQLite-backed metadata store for chunk and index-statistics persistence.

Implements :class:`~dotmd.storage.base.MetadataStoreProtocol` using the
Python standard-library ``sqlite3`` module.  Heading hierarchies are
serialised as JSON strings.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from dotmd.core.models import Chunk, IndexStats

# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    file_path       TEXT    NOT NULL,
    heading_hierarchy TEXT  NOT NULL DEFAULT '[]',
    level           INTEGER NOT NULL DEFAULT 0,
    text            TEXT    NOT NULL DEFAULT '',
    chunk_index     INTEGER NOT NULL DEFAULT 0,
    char_offset     INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_STATS = """
CREATE TABLE IF NOT EXISTS stats (
    id              INTEGER PRIMARY KEY DEFAULT 1,
    total_files     INTEGER NOT NULL DEFAULT 0,
    total_chunks    INTEGER NOT NULL DEFAULT 0,
    total_entities  INTEGER NOT NULL DEFAULT 0,
    total_edges     INTEGER NOT NULL DEFAULT 0,
    last_indexed    TEXT
)
"""

_UPSERT_CHUNK = """
INSERT INTO chunks (chunk_id, file_path, heading_hierarchy, level, text, chunk_index, char_offset)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(chunk_id) DO UPDATE SET
    file_path         = excluded.file_path,
    heading_hierarchy = excluded.heading_hierarchy,
    level             = excluded.level,
    text              = excluded.text,
    chunk_index       = excluded.chunk_index,
    char_offset       = excluded.char_offset
"""

_UPSERT_STATS = """
INSERT INTO stats (id, total_files, total_chunks, total_entities, total_edges, last_indexed)
VALUES (1, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    total_files    = excluded.total_files,
    total_chunks   = excluded.total_chunks,
    total_entities = excluded.total_entities,
    total_edges    = excluded.total_edges,
    last_indexed   = excluded.last_indexed
"""


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


class SQLiteMetadataStore:
    """SQLite implementation of :class:`MetadataStoreProtocol`.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Parent directories are **not**
        created automatically.  Use ``:memory:`` for an in-memory store.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_CHUNKS)
        self._conn.execute(_CREATE_STATS)
        self._conn.commit()

    # -- chunks -------------------------------------------------------------

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Persist a batch of chunks (insert or update)."""
        rows = [
            (
                c.chunk_id,
                str(c.file_path),
                json.dumps(c.heading_hierarchy),
                c.level,
                c.text,
                c.chunk_index,
                c.char_offset,
            )
            for c in chunks
        ]
        self._conn.executemany(_UPSERT_CHUNK, rows)
        self._conn.commit()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a single chunk by its identifier."""
        cur = self._conn.execute(
            "SELECT chunk_id, file_path, heading_hierarchy, level, text, chunk_index, char_offset "
            "FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        )
        row = cur.fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Retrieve multiple chunks by their identifiers.

        Missing ids are silently skipped.
        """
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        cur = self._conn.execute(
            f"SELECT chunk_id, file_path, heading_hierarchy, level, text, chunk_index, char_offset "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        return [self._row_to_chunk(row) for row in cur.fetchall()]

    def get_all_chunks(self) -> list[Chunk]:
        """Return every chunk currently stored."""
        cur = self._conn.execute(
            "SELECT chunk_id, file_path, heading_hierarchy, level, text, chunk_index, char_offset "
            "FROM chunks"
        )
        return [self._row_to_chunk(row) for row in cur.fetchall()]

    # -- stats --------------------------------------------------------------

    def save_stats(self, stats: IndexStats) -> None:
        """Persist index statistics (overwrites previous stats)."""
        last_indexed = (
            stats.last_indexed.isoformat() if stats.last_indexed else None
        )
        self._conn.execute(
            _UPSERT_STATS,
            (
                stats.total_files,
                stats.total_chunks,
                stats.total_entities,
                stats.total_edges,
                last_indexed,
            ),
        )
        self._conn.commit()

    def get_stats(self) -> IndexStats | None:
        """Retrieve the most recent index statistics."""
        cur = self._conn.execute(
            "SELECT total_files, total_chunks, total_entities, total_edges, last_indexed "
            "FROM stats WHERE id = 1"
        )
        row = cur.fetchone()
        if row is None:
            return None
        last_indexed = (
            datetime.fromisoformat(row[4]) if row[4] else None
        )
        return IndexStats(
            total_files=row[0],
            total_chunks=row[1],
            total_entities=row[2],
            total_edges=row[3],
            last_indexed=last_indexed,
        )

    # -- housekeeping -------------------------------------------------------

    def delete_all(self) -> None:
        """Remove all chunks and statistics from the store."""
        self._conn.execute("DELETE FROM chunks")
        self._conn.execute("DELETE FROM stats")
        self._conn.commit()

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _row_to_chunk(row: tuple) -> Chunk:  # type: ignore[type-arg]
        """Convert a raw SQLite row tuple into a :class:`Chunk`."""
        return Chunk(
            chunk_id=row[0],
            file_path=Path(row[1]),
            heading_hierarchy=json.loads(row[2]),
            level=row[3],
            text=row[4],
            chunk_index=row[5],
            char_offset=row[6],
        )
