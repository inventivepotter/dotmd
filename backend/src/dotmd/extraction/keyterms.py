"""Key-term extraction using corpus-level TF-IDF and pattern matching.

Supplements GLiNER NER with high-signal entities derived from:
- Acronyms and abbreviations (2+ uppercase letters)
- Heading terms (key concepts by document structure)
- TF-IDF discriminative terms (high in few chunks, rare across corpus)
"""

from __future__ import annotations

import math
import re
from collections import Counter

from dotmd.core.models import Chunk, Entity, ExtractionResult, Relation
from dotmd.utils.text import is_noise_token

# Matches 2+ uppercase letters optionally followed by digits (SIEM, MFA, AES256)
_ACRONYM_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")

# Matches Title Case multi-word terms (e.g., "Defense in Depth", "Least Privilege")
_TITLE_TERM_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+(?:in|of|and|the|by|for|to|on|at|vs|or)\s+[A-Z][a-z]+|"
    r"\s+[A-Z][a-z]+){1,4})\b"
)


class KeyTermExtractor:
    """Extract entities from corpus-level term importance and patterns.

    Unlike GLiNER (which predicts entity types per-span), this extractor
    identifies important terms using statistical and structural signals:

    1. **Acronyms** — uppercase sequences (SIEM, IAM, DLP, MFA)
    2. **Heading terms** — words from markdown headings are key concepts
    3. **TF-IDF terms** — terms with high frequency in a chunk but low
       document frequency across the corpus are discriminative

    Parameters
    ----------
    min_df:
        Minimum number of chunks a term must appear in to be kept.
    max_df_ratio:
        Maximum fraction of chunks a term may appear in (filters
        ubiquitous terms).
    top_k_per_chunk:
        Maximum number of TF-IDF terms to extract per chunk.
    """

    def __init__(
        self,
        min_df: int = 2,
        max_df_ratio: float = 0.6,
        top_k_per_chunk: int = 8,
        top_percentile: float = 0.10,
    ) -> None:
        self._min_df = min_df
        self._max_df_ratio = max_df_ratio
        self._top_k_per_chunk = top_k_per_chunk
        self._top_percentile = top_percentile

    def extract(self, chunks: list[Chunk]) -> ExtractionResult:
        """Extract key-term entities and relations from *chunks*.

        Returns entities typed as ``"acronym"``, ``"heading_term"``,
        or ``"key_term"`` with source ``"keyterm"``, plus ``MENTIONS``
        relations linking chunks to their extracted terms.
        """
        if not chunks:
            return ExtractionResult()

        entities: list[Entity] = []
        relations: list[Relation] = []
        seen: dict[str, Entity] = {}  # normalised name -> Entity

        # --- Phase 1: Acronym extraction (pattern-based) ---
        acronym_chunks: dict[str, list[str]] = {}  # acronym -> [chunk_ids]
        for chunk in chunks:
            for m in _ACRONYM_RE.finditer(chunk.text):
                acr = m.group(1)
                if is_noise_token(acr) or len(acr) < 2:
                    continue
                acronym_chunks.setdefault(acr, []).append(chunk.chunk_id)

        for acr, cids in acronym_chunks.items():
            if len(cids) < self._min_df:
                continue
            key = acr.lower()
            if key not in seen:
                ent = Entity(
                    name=acr,
                    type="acronym",
                    source="keyterm",
                    chunk_ids=list(dict.fromkeys(cids)),
                )
                seen[key] = ent
                entities.append(ent)

        # --- Phase 2: Heading terms ---
        heading_chunks: dict[str, list[str]] = {}
        for chunk in chunks:
            for heading in chunk.heading_hierarchy:
                # Extract meaningful multi-word terms from headings
                for m in _TITLE_TERM_RE.finditer(heading):
                    term = m.group(1)
                    if len(term) > 3:
                        heading_chunks.setdefault(term, []).append(chunk.chunk_id)
                # Also extract single capitalised words > 3 chars from headings
                for word in heading.split():
                    clean = re.sub(r"[^A-Za-z]", "", word)
                    if (
                        clean
                        and clean[0].isupper()
                        and len(clean) > 3
                        and not is_noise_token(clean)
                    ):
                        heading_chunks.setdefault(clean, []).append(chunk.chunk_id)

        max_heading_df = int(len(chunks) * self._max_df_ratio)
        for term, cids in heading_chunks.items():
            key = term.lower()
            unique_cids = list(dict.fromkeys(cids))
            if key in seen or len(unique_cids) < self._min_df or len(unique_cids) > max_heading_df:
                continue
            cids = unique_cids
            ent = Entity(
                name=term,
                type="heading_term",
                source="keyterm",
                chunk_ids=list(dict.fromkeys(cids)),
            )
            seen[key] = ent
            entities.append(ent)

        # --- Phase 3: TF-IDF discriminative terms ---
        # Build document frequency across all chunks
        n_chunks = len(chunks)
        df: Counter[str] = Counter()
        chunk_tfs: list[Counter[str]] = []

        for chunk in chunks:
            tokens = _tokenize_for_tfidf(chunk.text)
            tf = Counter(tokens)
            chunk_tfs.append(tf)
            for token in set(tokens):
                df[token] += 1

        max_df = int(n_chunks * self._max_df_ratio)

        for chunk, tf in zip(chunks, chunk_tfs):
            # Score each term by TF-IDF
            scored: list[tuple[str, float]] = []
            for term, count in tf.items():
                term_df = df[term]
                if term_df < self._min_df or term_df > max_df:
                    continue
                if len(term) <= 2 or is_noise_token(term):
                    continue
                idf = math.log(n_chunks / term_df)
                tfidf = count * idf
                scored.append((term, tfidf))

            scored.sort(key=lambda x: x[1], reverse=True)

            for term, score in scored[: self._top_k_per_chunk]:
                key = term.lower()
                if key in seen:
                    existing = seen[key]
                    if chunk.chunk_id not in existing.chunk_ids:
                        existing.chunk_ids.append(chunk.chunk_id)
                else:
                    ent = Entity(
                        name=term,
                        type="key_term",
                        source="keyterm",
                        chunk_ids=[chunk.chunk_id],
                    )
                    seen[key] = ent
                    entities.append(ent)

        # --- Filter to top percentile by chunk coverage ---
        if self._top_percentile < 1.0 and entities:
            sorted_ents = sorted(entities, key=lambda e: len(e.chunk_ids), reverse=True)
            keep_count = max(1, int(len(sorted_ents) * self._top_percentile))
            top_set = {e.name.lower() for e in sorted_ents[:keep_count]}
            entities = [e for e in entities if e.name.lower() in top_set]

        # --- Build MENTIONS relations ---
        for ent in entities:
            for cid in ent.chunk_ids:
                relations.append(
                    Relation(
                        source_id=cid,
                        target_id=ent.name,
                        relation_type="MENTIONS",
                        weight=1.0,
                    )
                )

        # --- Build CO_OCCURS relations (entities sharing chunks) ---
        # Group entities by chunk_id
        chunk_entities: dict[str, list[str]] = {}
        for ent in entities:
            for cid in ent.chunk_ids:
                chunk_entities.setdefault(cid, []).append(ent.name)

        co_occur_seen: set[tuple[str, str]] = set()
        for cid, ent_names in chunk_entities.items():
            for i, a in enumerate(ent_names):
                for b in ent_names[i + 1 :]:
                    pair = (min(a, b), max(a, b))
                    if pair not in co_occur_seen:
                        co_occur_seen.add(pair)
                        relations.append(
                            Relation(
                                source_id=a,
                                target_id=b,
                                relation_type="CO_OCCURS",
                                weight=1.0,
                            )
                        )

        return ExtractionResult(entities=entities, relations=relations)


def _tokenize_for_tfidf(text: str) -> list[str]:
    """Tokenize text for TF-IDF, keeping meaningful terms."""
    text_lower = text.lower()
    tokens = re.findall(r"\b[a-z][a-z0-9_]{2,}\b", text_lower)
    return [t for t in tokens if not is_noise_token(t) and len(t) > 2]
