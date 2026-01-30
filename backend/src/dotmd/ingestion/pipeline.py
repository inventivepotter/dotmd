"""End-to-end indexing pipeline for dotMD.

Orchestrates file discovery, chunking, embedding, BM25 index construction,
structural and NER extraction, and knowledge-graph population.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from dotmd.core.config import Settings
from dotmd.core.models import Chunk, ExtractionResult, IndexStats
from dotmd.extraction.acronyms import extract_acronyms_from_chunks
from dotmd.extraction.keyterms import KeyTermExtractor
from dotmd.extraction.ner import NERExtractor
from dotmd.extraction.structural import StructuralExtractor
from dotmd.ingestion.chunker import chunk_file
from dotmd.ingestion.reader import discover_files, read_file
from dotmd.search.bm25 import BM25SearchEngine
from dotmd.search.semantic import SemanticSearchEngine
from dotmd.storage.graph import LadybugDBGraphStore
from dotmd.storage.metadata import SQLiteMetadataStore
from dotmd.storage.vector import LanceDBVectorStore

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Orchestrates the full indexing workflow from raw files to populated stores.

    Parameters
    ----------
    settings:
        Application-wide configuration.  Storage paths, model names, and
        extraction options are all derived from this object.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        # Ensure the index directory exists.
        settings.index_dir.mkdir(parents=True, exist_ok=True)

        # -- storage backends --------------------------------------------------
        self._metadata_store = SQLiteMetadataStore(settings.sqlite_path)
        self._vector_store = LanceDBVectorStore(settings.lancedb_path)
        self._graph_store = LadybugDBGraphStore(
            settings.graph_db_path, read_only=settings.read_only,
        )

        # -- search engines (used for encoding during indexing) ----------------
        self._semantic_engine = SemanticSearchEngine(
            self._vector_store,
            settings.embedding_model,
        )
        self._bm25_engine = BM25SearchEngine(settings.bm25_path)

        # -- extractors --------------------------------------------------------
        self._structural_extractor = StructuralExtractor()
        self._keyterm_extractor = KeyTermExtractor()
        self._ner_extractor: NERExtractor | None = None
        if settings.extract_depth == "ner":
            self._ner_extractor = NERExtractor(settings.ner_entity_types)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, directory: Path) -> IndexStats:
        """Index every markdown file under *directory*.

        The method performs the following steps in order:

        1. Discover ``.md`` files.
        2. Read and chunk each file.
        3. Persist chunks in the metadata store.
        4. Encode chunks and populate the vector store.
        5. Build the BM25 index.
        6. Run structural extraction.
        7. Optionally run NER extraction.
        8. Populate the knowledge graph with entities, relations, file
           nodes, and section nodes.
        9. Persist and return :class:`IndexStats`.

        Parameters
        ----------
        directory:
            Root directory to scan for markdown files.

        Returns
        -------
        IndexStats
            Summary statistics for the completed index.
        """
        # 1. Discover files
        files = discover_files(directory)
        logger.info("Discovered %d files in %s", len(files), directory)

        # 2. Read and chunk
        all_chunks: list[Chunk] = []
        for file_info in files:
            content = read_file(file_info.path)
            file_chunks = chunk_file(
                file_info.path,
                content,
                max_tokens=self._settings.max_chunk_tokens,
                overlap_tokens=self._settings.chunk_overlap_tokens,
            )
            all_chunks.extend(file_chunks)

        logger.info("Produced %d chunks from %d files", len(all_chunks), len(files))

        # 3. Save chunks to metadata store
        self._metadata_store.save_chunks(all_chunks)

        # 4. Encode and add to vector store
        if all_chunks:
            texts = [chunk.text for chunk in all_chunks]
            embeddings = self._semantic_engine.encode_batch(texts)
            self._vector_store.add_chunks(all_chunks, embeddings)
            logger.info("Added %d vectors to vector store", len(all_chunks))

        # 5. Build BM25 index
        self._bm25_engine.build_index(all_chunks)

        # 6. Structural extraction
        structural_result = self._structural_extractor.extract(all_chunks)
        logger.info(
            "Structural extraction: %d entities, %d relations",
            len(structural_result.entities),
            len(structural_result.relations),
        )

        # 7. NER extraction (optional)
        ner_result = ExtractionResult()
        if self._ner_extractor is not None:
            ner_result = self._ner_extractor.extract(all_chunks)
            logger.info(
                "NER extraction: %d entities, %d relations",
                len(ner_result.entities),
                len(ner_result.relations),
            )

        # 8. Key-term extraction (TF-IDF + acronyms + heading terms)
        keyterm_result = self._keyterm_extractor.extract(all_chunks)
        logger.info(
            "Key-term extraction: %d entities, %d relations",
            len(keyterm_result.entities),
            len(keyterm_result.relations),
        )

        # 9. Populate graph store
        all_entities = (
            structural_result.entities
            + ner_result.entities
            + keyterm_result.entities
        )
        all_relations = (
            structural_result.relations
            + ner_result.relations
            + keyterm_result.relations
        )

        # Add entity nodes
        for entity in all_entities:
            if entity.type == "tag":
                self._graph_store.add_tag_node(entity.name)
            else:
                self._graph_store.add_entity_node(
                    name=entity.name,
                    entity_type=entity.type,
                    source=entity.source,
                )

        # Add file nodes
        for file_info in files:
            self._graph_store.add_file_node(
                file_path=str(file_info.path),
                title=file_info.title,
                checksum=file_info.checksum,
            )

        # Add section nodes
        for chunk in all_chunks:
            self._graph_store.add_section_node(
                chunk_id=chunk.chunk_id,
                heading=chunk.heading,
                level=chunk.level,
                file_path=str(chunk.file_path),
                text_preview=chunk.text[:200],
            )

        # Add edges
        for relation in all_relations:
            self._graph_store.add_edge(
                source_id=relation.source_id,
                target_id=relation.target_id,
                relation_type=relation.relation_type,
                weight=relation.weight,
            )

        # Also link sections to their parent files via CONTAINS edges.
        for chunk in all_chunks:
            self._graph_store.add_edge(
                source_id=str(chunk.file_path),
                target_id=chunk.chunk_id,
                relation_type="CONTAINS",
            )

        # 9. Extract and persist acronym dictionary
        acronym_dict = extract_acronyms_from_chunks(all_chunks)
        if acronym_dict:
            import json

            self._settings.acronyms_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._settings.acronyms_path, "w") as f:
                json.dump(acronym_dict, f, indent=2)
            logger.info("Extracted %d acronyms", len(acronym_dict))

        # 10. Build and persist stats
        stats = IndexStats(
            total_files=len(files),
            total_chunks=len(all_chunks),
            total_entities=len(all_entities),
            total_edges=len(all_relations),
            last_indexed=datetime.now(tz=timezone.utc),
        )
        self._metadata_store.save_stats(stats)
        logger.info("Indexing complete: %s", stats)

        return stats

    def clear(self) -> None:
        """Delete all data from every backing store."""
        self._metadata_store.delete_all()
        self._vector_store.delete_all()
        self._graph_store.delete_all()

        # Delete acronym dictionary
        if self._settings.acronyms_path.exists():
            self._settings.acronyms_path.unlink()

        logger.info("All stores cleared")

    # ------------------------------------------------------------------
    # Accessors (used by the service layer)
    # ------------------------------------------------------------------

    @property
    def metadata_store(self) -> SQLiteMetadataStore:
        """Return the metadata store instance."""
        return self._metadata_store

    @property
    def vector_store(self) -> LanceDBVectorStore:
        """Return the vector store instance."""
        return self._vector_store

    @property
    def graph_store(self) -> LadybugDBGraphStore:
        """Return the graph store instance."""
        return self._graph_store

    @property
    def semantic_engine(self) -> SemanticSearchEngine:
        """Return the semantic search engine instance."""
        return self._semantic_engine

    @property
    def bm25_engine(self) -> BM25SearchEngine:
        """Return the BM25 search engine instance."""
        return self._bm25_engine
