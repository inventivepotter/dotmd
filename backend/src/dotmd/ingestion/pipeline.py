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

    def index(self, directory: Path, force: bool = False) -> IndexStats:
        """Index every markdown file under *directory*.

        By default, only files whose content has changed since the last
        run are reprocessed (partial re-index).  Pass ``force=True`` to
        reprocess all files regardless of their checksum.

        The method performs the following steps in order:

        1. Discover ``.md`` files and compare checksums against the store.
        2. Read and chunk only new or changed files.
        3. Persist chunks in the metadata store.
        4. Encode all chunks and populate the vector store.
        5. Build the BM25 index.
        6. Run structural extraction on new/changed chunks.
        7. Optionally run NER extraction on new/changed chunks.
        8. Populate the knowledge graph with entities, relations, file
           nodes, and section nodes.
        9. Persist and return :class:`IndexStats`.

        Parameters
        ----------
        directory:
            Root directory to scan for markdown files.
        force:
            When ``True``, bypass checksum comparison and reprocess every
            file from scratch.

        Returns
        -------
        IndexStats
            Summary statistics for the completed index.
        """
        # 1. Discover files and classify them as new, changed, or unchanged.
        files = discover_files(directory)
        logger.info("Discovered %d files in %s", len(files), directory)

        if force:
            changed_files = files
            deleted_paths: set[str] = set()
            logger.info("Force mode: reprocessing all %d files", len(files))
        else:
            previous_checksums = self._metadata_store.get_file_checksums()
            current_paths = {str(f.path) for f in files}
            deleted_paths = set(previous_checksums) - current_paths
            changed_files = [
                f for f in files
                if previous_checksums.get(str(f.path)) != f.checksum
            ]
            skipped = len(files) - len(changed_files)
            logger.info(
                "%d file(s) changed or new, %d unchanged, %d deleted",
                len(changed_files), skipped, len(deleted_paths),
            )

        # Remove chunks for deleted files and files that will be re-chunked.
        paths_to_clean = deleted_paths | {str(f.path) for f in changed_files}
        for path in paths_to_clean:
            self._metadata_store.delete_chunks_by_file(path)

        # 2. Read and chunk only changed/new files.
        new_chunks: list[Chunk] = []
        for file_info in changed_files:
            content = read_file(file_info.path)
            file_chunks = chunk_file(
                file_info.path,
                content,
                max_tokens=self._settings.max_chunk_tokens,
                overlap_tokens=self._settings.chunk_overlap_tokens,
            )
            new_chunks.extend(file_chunks)

        logger.info(
            "Produced %d new chunks from %d changed file(s)",
            len(new_chunks), len(changed_files),
        )

        # 3. Save new/changed chunks to metadata store.
        if new_chunks:
            self._metadata_store.save_chunks(new_chunks)

        # Load the full corpus (unchanged + new) for BM25 and vector rebuild.
        all_chunks = self._metadata_store.get_all_chunks()

        # 4. Encode and add to vector store (full rebuild from merged corpus).
        if all_chunks:
            texts = [chunk.text for chunk in all_chunks]
            embeddings = self._semantic_engine.encode_batch(texts)
            self._vector_store.add_chunks(all_chunks, embeddings)
            logger.info("Added %d vectors to vector store", len(all_chunks))

        # 5. Build BM25 index from the full corpus.
        self._bm25_engine.build_index(all_chunks)

        # Steps 6–8 run only on new/changed chunks — unchanged files are
        # already represented in the graph and extraction stores.
        if new_chunks:
            # 6. Structural extraction
            structural_result = self._structural_extractor.extract(new_chunks)
            logger.info(
                "Structural extraction: %d entities, %d relations",
                len(structural_result.entities),
                len(structural_result.relations),
            )

            # 7. NER extraction (optional)
            ner_result = ExtractionResult()
            if self._ner_extractor is not None:
                ner_result = self._ner_extractor.extract(new_chunks)
                logger.info(
                    "NER extraction: %d entities, %d relations",
                    len(ner_result.entities),
                    len(ner_result.relations),
                )

            # 8. Key-term extraction (TF-IDF + acronyms + heading terms)
            keyterm_result = self._keyterm_extractor.extract(new_chunks)
            logger.info(
                "Key-term extraction: %d entities, %d relations",
                len(keyterm_result.entities),
                len(keyterm_result.relations),
            )
        else:
            structural_result = ExtractionResult()
            ner_result = ExtractionResult()
            keyterm_result = ExtractionResult()

        # 9. Populate graph store for changed files only.
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

        # Add file nodes for changed/new files only.
        for file_info in changed_files:
            self._graph_store.add_file_node(
                file_path=str(file_info.path),
                title=file_info.title,
                checksum=file_info.checksum,
            )

        # Add section nodes and CONTAINS edges for new/changed chunks only.
        for chunk in new_chunks:
            self._graph_store.add_section_node(
                chunk_id=chunk.chunk_id,
                heading=chunk.heading,
                level=chunk.level,
                file_path=str(chunk.file_path),
                text_preview=chunk.text[:200],
            )

        for relation in all_relations:
            self._graph_store.add_edge(
                source_id=relation.source_id,
                target_id=relation.target_id,
                relation_type=relation.relation_type,
                weight=relation.weight,
            )

        for chunk in new_chunks:
            self._graph_store.add_edge(
                source_id=str(chunk.file_path),
                target_id=chunk.chunk_id,
                relation_type="CONTAINS",
            )

        # Rebuild acronym dictionary from the full corpus.
        acronym_dict = extract_acronyms_from_chunks(all_chunks)
        if acronym_dict:
            import json

            self._settings.acronyms_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._settings.acronyms_path, "w") as f:
                json.dump(acronym_dict, f, indent=2)
            logger.info("Extracted %d acronyms", len(acronym_dict))

        # 10. Persist file checksums and index stats.
        self._metadata_store.save_file_checksums(files)

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
