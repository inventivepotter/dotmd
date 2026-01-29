"""Named-entity recognition extraction using GLiNER (zero-shot NER)."""

from __future__ import annotations

import logging
from collections import Counter
from itertools import combinations
from typing import TYPE_CHECKING, Any

from dotmd.core.models import Chunk, Entity, ExtractionResult, Relation

if TYPE_CHECKING:
    from gliner import GLiNER  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_DEFAULT_ENTITY_TYPES: list[str] = [
    "person",
    "organization",
    "technology",
    "concept",
    "location",
]

_DEFAULT_MODEL_NAME = "urchade/gliner_multi-v2.1"


class NERExtractor:
    """Zero-shot named-entity recognition extractor backed by GLiNER.

    The GLiNER model is loaded lazily on the first call to :meth:`extract` so
    that import-time cost is zero and the model is only downloaded when
    actually needed.

    Args:
        entity_types: Entity type labels passed to GLiNER for zero-shot
            prediction.  Defaults to ``["person", "organization",
            "technology", "concept", "location"]``.
        model_name: HuggingFace model identifier.  Defaults to
            ``urchade/gliner_multi-v2.1``.
        threshold: Minimum confidence score for GLiNER predictions.
    """

    def __init__(
        self,
        entity_types: list[str] | None = None,
        model_name: str = _DEFAULT_MODEL_NAME,
        threshold: float = 0.5,
    ) -> None:
        self._entity_types: list[str] = entity_types or list(_DEFAULT_ENTITY_TYPES)
        self._model_name: str = model_name
        self._threshold: float = threshold
        self._model: GLiNER | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, chunks: list[Chunk]) -> ExtractionResult:
        """Run GLiNER NER over *chunks* and return entities and relations.

        For each chunk the extractor:

        1. Predicts entities using GLiNER zero-shot NER.
        2. Creates ``Entity`` objects for each unique entity found.
        3. Creates ``CO_OCCURS`` relations between every pair of entities
           found within the same chunk (weight = 1.0).
        4. Creates ``MENTIONS`` relations from the chunk to each entity
           (weight = frequency count of the entity span in the chunk text).

        Args:
            chunks: List of document chunks to process.

        Returns:
            Aggregated ``ExtractionResult``.
        """
        model = self._get_model()

        entities: list[Entity] = []
        relations: list[Relation] = []

        # Track globally unique entities by (normalised_name, type).
        seen_entities: dict[tuple[str, str], Entity] = {}

        for chunk in chunks:
            predictions: list[dict[str, Any]] = model.predict_entities(
                chunk.text,
                self._entity_types,
                threshold=self._threshold,
            )

            if not predictions:
                continue

            # Count occurrences of each entity span in the chunk.
            span_counter: Counter[str] = Counter()
            chunk_entity_keys: list[tuple[str, str]] = []

            for pred in predictions:
                name: str = pred["text"].strip()
                etype: str = pred["label"]
                if not name:
                    continue

                span_counter[name] += 1
                key = (name.lower(), etype.lower())
                chunk_entity_keys.append(key)

                if key not in seen_entities:
                    entity = Entity(
                        name=name,
                        type=etype,
                        source="ner",
                        chunk_ids=[chunk.chunk_id],
                    )
                    seen_entities[key] = entity
                    entities.append(entity)
                else:
                    existing = seen_entities[key]
                    if chunk.chunk_id not in existing.chunk_ids:
                        existing.chunk_ids.append(chunk.chunk_id)

            # Deduplicate keys within this chunk for relation generation.
            unique_keys = list(dict.fromkeys(chunk_entity_keys))

            # --- CO_OCCURS relations (between entity pairs in the chunk) ------
            for key_a, key_b in combinations(unique_keys, 2):
                relations.append(
                    Relation(
                        source_id=seen_entities[key_a].name,
                        target_id=seen_entities[key_b].name,
                        relation_type="CO_OCCURS",
                        weight=1.0,
                    )
                )

            # --- MENTIONS relations (chunk → entity) --------------------------
            for key in unique_keys:
                entity = seen_entities[key]
                freq = span_counter[entity.name]
                relations.append(
                    Relation(
                        source_id=chunk.chunk_id,
                        target_id=entity.name,
                        relation_type="MENTIONS",
                        weight=float(freq),
                    )
                )

        return ExtractionResult(entities=entities, relations=relations)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> GLiNER:
        """Lazily load and cache the GLiNER model."""
        if self._model is None:
            logger.info("Loading GLiNER model '%s' …", self._model_name)
            from gliner import GLiNER  # type: ignore[import-untyped]

            self._model = GLiNER.from_pretrained(self._model_name)
            logger.info("GLiNER model loaded.")
        return self._model
