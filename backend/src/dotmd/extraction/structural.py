"""Structural extraction from markdown syntax.

Extracts entities and relations from wikilinks, tags, YAML frontmatter,
markdown links, and heading hierarchy.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

from dotmd.core.models import Chunk, Entity, ExtractionResult, Relation

# --- Regex patterns -----------------------------------------------------------

_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")
_INLINE_TAG_RE = re.compile(r"(?:^|(?<=\s))#([A-Za-z_][\w/-]*)", re.MULTILINE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+\.md(?:#[^)]*)?)\)")
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)---\n", re.DOTALL)


class StructuralExtractor:
    """Extract entities and relations from markdown structural elements.

    Recognised patterns:
    - ``[[wikilinks]]`` — creates a *link* entity and a ``LINKS_TO`` relation.
    - Inline ``#tags`` (not heading lines) — creates a *tag* entity and a
      ``HAS_TAG`` relation.
    - YAML front-matter — creates entities for each key-value pair.
    - Markdown links to ``.md`` files — creates a *link* entity and a
      ``LINKS_TO`` relation.
    - Heading hierarchy — creates ``PARENT_OF`` relations between parent and
      child chunks.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, chunks: list[Chunk]) -> ExtractionResult:
        """Extract entities and relations from *chunks*.

        Args:
            chunks: Ordered list of document chunks.

        Returns:
            Aggregated ``ExtractionResult``.
        """
        entities: list[Entity] = []
        relations: list[Relation] = []

        # Build a lookup from heading hierarchy tuple → chunk_id for parent
        # resolution.  The parent of a chunk is the chunk whose hierarchy is
        # the immediate prefix of the current one.
        hierarchy_map: dict[tuple[str, ...], str] = {}
        for chunk in chunks:
            key = tuple(chunk.heading_hierarchy)
            hierarchy_map[key] = chunk.chunk_id

        for chunk in chunks:
            text = chunk.text
            cid = chunk.chunk_id

            # --- Wikilinks ---------------------------------------------------
            for match in _WIKILINK_RE.finditer(text):
                target = match.group(1).strip()
                entity = Entity(
                    name=target,
                    type="link",
                    source="structural",
                    chunk_ids=[cid],
                )
                entities.append(entity)
                relations.append(
                    Relation(
                        source_id=cid,
                        target_id=target,
                        relation_type="LINKS_TO",
                    )
                )

            # --- Inline tags --------------------------------------------------
            for match in _INLINE_TAG_RE.finditer(text):
                # Exclude lines that start with '#' as a heading (e.g. "# Title").
                line_start = text.rfind("\n", 0, match.start()) + 1
                line = text[line_start : match.end()]
                if line.lstrip().startswith("# ") or line.lstrip().startswith("## ") or line.lstrip().startswith("### "):
                    continue
                tag = match.group(1)
                entity = Entity(
                    name=tag,
                    type="tag",
                    source="structural",
                    chunk_ids=[cid],
                )
                entities.append(entity)
                relations.append(
                    Relation(
                        source_id=cid,
                        target_id=tag,
                        relation_type="HAS_TAG",
                    )
                )

            # --- YAML frontmatter --------------------------------------------
            fm_match = _FRONTMATTER_RE.match(text)
            if fm_match:
                self._extract_frontmatter(fm_match.group(1), cid, entities, relations)

            # --- Markdown links to .md files ----------------------------------
            for match in _MD_LINK_RE.finditer(text):
                link_text = match.group(1)
                href = match.group(2).split("#")[0]  # strip anchor
                entity = Entity(
                    name=href,
                    type="link",
                    source="structural",
                    chunk_ids=[cid],
                )
                entities.append(entity)
                relations.append(
                    Relation(
                        source_id=cid,
                        target_id=href,
                        relation_type="LINKS_TO",
                        properties={"link_text": link_text},
                    )
                )

            # --- Heading hierarchy → PARENT_OF --------------------------------
            hier = tuple(chunk.heading_hierarchy)
            if len(hier) > 1:
                parent_key = hier[:-1]
                parent_id = hierarchy_map.get(parent_key)
                if parent_id is not None:
                    relations.append(
                        Relation(
                            source_id=parent_id,
                            target_id=cid,
                            relation_type="PARENT_OF",
                        )
                    )

        return ExtractionResult(entities=entities, relations=relations)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_frontmatter(
        raw_yaml: str,
        chunk_id: str,
        entities: list[Entity],
        relations: list[Relation],
    ) -> None:
        """Parse YAML frontmatter and append entities/relations."""
        try:
            data: Any = yaml.safe_load(raw_yaml)
        except yaml.YAMLError:
            return

        if not isinstance(data, dict):
            return

        for key, value in data.items():
            values = value if isinstance(value, list) else [value]
            for val in values:
                if val is None:
                    continue
                entity = Entity(
                    name=str(val),
                    type=str(key),
                    source="structural",
                    chunk_ids=[chunk_id],
                )
                entities.append(entity)
                relations.append(
                    Relation(
                        source_id=chunk_id,
                        target_id=str(val),
                        relation_type="HAS_FRONTMATTER",
                        properties={"key": str(key)},
                    )
                )
