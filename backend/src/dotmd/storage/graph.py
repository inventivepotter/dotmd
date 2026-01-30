"""LadybugDB-backed knowledge-graph store.

Implements :class:`~dotmd.storage.base.GraphStoreProtocol` using
LadybugDB (an embedded Cypher graph database forked from Kuzu) for local,
zero-configuration graph storage.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import real_ladybug as lb  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Schema constants
_SCHEMA_INIT = [
    # Node tables
    "CREATE NODE TABLE IF NOT EXISTS File(id STRING, title STRING, checksum STRING, PRIMARY KEY (id))",
    "CREATE NODE TABLE IF NOT EXISTS Section(id STRING, heading STRING, level INT64, file_path STRING, text_preview STRING, PRIMARY KEY (id))",
    "CREATE NODE TABLE IF NOT EXISTS Entity(id STRING, type STRING, source STRING, PRIMARY KEY (id))",
    "CREATE NODE TABLE IF NOT EXISTS Tag(id STRING, PRIMARY KEY (id))",
    # Relationship tables — one generic table per node-pair combination
    # LadybugDB requires explicit FROM/TO types, so we create tables for
    # all combinations we use.
    "CREATE REL TABLE IF NOT EXISTS FILE_SECTION(FROM File TO Section, rel_type STRING, weight DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS SECTION_SECTION(FROM Section TO Section, rel_type STRING, weight DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS SECTION_ENTITY(FROM Section TO Entity, rel_type STRING, weight DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS SECTION_TAG(FROM Section TO Tag, rel_type STRING, weight DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS ENTITY_ENTITY(FROM Entity TO Entity, rel_type STRING, weight DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS FILE_TAG(FROM File TO Tag, rel_type STRING, weight DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS FILE_ENTITY(FROM File TO Entity, rel_type STRING, weight DOUBLE)",
]

# Map (source_label, target_label) -> relationship table name
_REL_TABLE_MAP: dict[tuple[str, str], str] = {
    ("File", "Section"): "FILE_SECTION",
    ("Section", "Section"): "SECTION_SECTION",
    ("Section", "Entity"): "SECTION_ENTITY",
    ("Section", "Tag"): "SECTION_TAG",
    ("Entity", "Entity"): "ENTITY_ENTITY",
    ("File", "Tag"): "FILE_TAG",
    ("File", "Entity"): "FILE_ENTITY",
}


class LadybugDBGraphStore:
    """LadybugDB implementation of :class:`GraphStoreProtocol`.

    Parameters
    ----------
    db_path:
        File-system path for the embedded LadybugDB database directory.
    """

    def __init__(self, db_path: Path, *, read_only: bool = False) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self._read_only = read_only
        self._db: lb.Database | None = None
        self._conn: lb.Connection | None = None
        if not read_only:
            # Write mode: hold a persistent connection
            self._db = lb.Database(self._db_path)
            self._conn = lb.Connection(self._db)
            self._init_schema()

    @contextmanager
    def _connection(self) -> Iterator[lb.Connection]:
        """Yield a connection, opening a temporary one in read-only mode."""
        if self._conn is not None:
            yield self._conn
        else:
            db = lb.Database(self._db_path, read_only=True)
            conn = lb.Connection(db)
            try:
                yield conn
            finally:
                del conn
                del db

    def _init_schema(self) -> None:
        """Create node and relationship tables if they don't exist."""
        for stmt in _SCHEMA_INIT:
            try:
                self._conn.execute(stmt)
            except Exception:
                # Table already exists or other non-fatal issue
                logger.debug("Schema statement skipped: %s", stmt, exc_info=True)

    # -- node creation ------------------------------------------------------

    def add_file_node(
        self,
        file_path: str,
        title: str,
        checksum: str,
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                "MERGE (f:File {id: $id}) SET f.title = $title, f.checksum = $checksum",
                parameters={"id": file_path, "title": title, "checksum": checksum},
            )

    def add_section_node(
        self,
        chunk_id: str,
        heading: str,
        level: int,
        file_path: str,
        text_preview: str,
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                "MERGE (s:Section {id: $id}) "
                "SET s.heading = $heading, s.level = $level, "
                "s.file_path = $file_path, s.text_preview = $text_preview",
                parameters={
                    "id": chunk_id,
                    "heading": heading,
                    "level": level,
                    "file_path": file_path,
                    "text_preview": text_preview,
                },
            )

    def add_entity_node(
        self,
        name: str,
        entity_type: str,
        source: str,
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                "MERGE (e:Entity {id: $id}) SET e.type = $type, e.source = $source",
                parameters={"id": name, "type": entity_type, "source": source},
            )

    def add_tag_node(self, name: str) -> None:
        with self._connection() as conn:
            conn.execute(
                "MERGE (t:Tag {id: $id})",
                parameters={"id": name},
            )

    # -- edge creation ------------------------------------------------------

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> None:
        with self._connection() as conn:
            # Determine node labels by querying which table each id belongs to.
            src_label = self._find_node_label(source_id, conn)
            tgt_label = self._find_node_label(target_id, conn)

            if src_label is None or tgt_label is None:
                logger.warning(
                    "Cannot add edge: node not found (src=%s [%s], tgt=%s [%s])",
                    source_id, src_label, target_id, tgt_label,
                )
                return

            rel_table = _REL_TABLE_MAP.get((src_label, tgt_label))
            if rel_table is None:
                logger.warning(
                    "No relationship table for %s -> %s", src_label, tgt_label,
                )
                return

            conn.execute(
                f"MATCH (a:{src_label} {{id: $src}}), (b:{tgt_label} {{id: $tgt}}) "
                f"MERGE (a)-[r:{rel_table}]->(b) "
                "SET r.rel_type = $rel_type, r.weight = $weight",
                parameters={
                    "src": source_id,
                    "tgt": target_id,
                    "rel_type": relation_type,
                    "weight": weight,
                },
            )

    def _find_node_label(self, node_id: str, conn: lb.Connection) -> str | None:
        """Find which node table a given id belongs to."""
        for label in ("File", "Section", "Entity", "Tag"):
            result = conn.execute(
                f"MATCH (n:{label} {{id: $id}}) RETURN n.id",
                parameters={"id": node_id},
            )
            if len(result.get_as_df()) > 0:
                return label
        return None

    # -- queries ------------------------------------------------------------

    def get_neighbors(
        self,
        node_id: str,
        max_hops: int = 2,
    ) -> list[tuple[str, str, float]]:
        # LadybugDB supports variable-length relationships with [r* SHORTEST 1..N]
        # but for simplicity we query all rel tables with multi-hop.
        neighbors: list[tuple[str, str, float]] = []

        with self._connection() as conn:
            src_label = self._find_node_label(node_id, conn)
            if src_label is None:
                return neighbors

            result = conn.execute(
                f"MATCH (a:{src_label} {{id: $id}})-[r* 1..{int(max_hops)}]-(b) "
                "RETURN DISTINCT b.id, label(b)",
                parameters={"id": node_id},
            )

            df = result.get_as_df()
            for _, row in df.iterrows():
                nid = row.iloc[0]
                if nid == node_id:
                    continue
                neighbors.append((str(nid), "", 1.0))

        return neighbors

    # -- housekeeping -------------------------------------------------------

    def delete_all(self) -> None:
        """Remove all nodes and edges from the graph."""
        with self._connection() as conn:
            for stmt in reversed(_SCHEMA_INIT):
                table_name = stmt.split("IF NOT EXISTS")[1].strip().split("(")[0].strip() if "IF NOT EXISTS" in stmt else None
                if table_name:
                    try:
                        conn.execute(f"MATCH ()-[r:{table_name}]->() DELETE r")
                    except Exception:
                        pass
            for label in ("File", "Section", "Entity", "Tag"):
                try:
                    conn.execute(f"MATCH (n:{label}) DELETE n")
                except Exception:
                    pass

    def node_count(self) -> int:
        """Return the total number of nodes in the graph."""
        total = 0
        with self._connection() as conn:
            for label in ("File", "Section", "Entity", "Tag"):
                try:
                    result = conn.execute(f"MATCH (n:{label}) RETURN count(n)")
                    df = result.get_as_df()
                    total += int(df.iloc[0, 0])
                except Exception:
                    pass
        return total

    def edge_count(self) -> int:
        """Return the total number of edges in the graph."""
        total = 0
        with self._connection() as conn:
            for rel_table in _REL_TABLE_MAP.values():
                try:
                    result = conn.execute(f"MATCH ()-[r:{rel_table}]->() RETURN count(r)")
                    df = result.get_as_df()
                    total += int(df.iloc[0, 0])
                except Exception:
                    pass
        return total
