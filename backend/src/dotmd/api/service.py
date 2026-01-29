"""UI-agnostic service facade for dotMD.

Provides a high-level API for indexing and searching that hides all
storage, extraction, and fusion details from calling code.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotmd.core.config import Settings
from dotmd.core.models import IndexStats, SearchResult
from dotmd.ingestion.pipeline import IndexingPipeline
from dotmd.search.bm25 import BM25SearchEngine
from dotmd.search.fusion import build_search_results, fuse_results
from dotmd.search.graph_search import GraphSearchEngine
from dotmd.search.query import QueryExpander
from dotmd.search.reranker import Reranker
from dotmd.search.semantic import SemanticSearchEngine

logger = logging.getLogger(__name__)


class DotMDService:
    """High-level service facade for indexing and searching markdown files.

    All storage backends, search engines, and extraction components are
    created internally based on the provided :class:`Settings`.

    Parameters
    ----------
    settings:
        Application configuration.  When ``None`` a default
        :class:`Settings` instance is created.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()

        # Indexing pipeline (also creates storage backends and extractors).
        self._pipeline = IndexingPipeline(self._settings)

        # Search engines -- reuse stores created by the pipeline.
        self._semantic_engine = SemanticSearchEngine(
            self._pipeline.vector_store,
            self._settings.embedding_model,
        )
        self._bm25_engine = BM25SearchEngine(self._settings.bm25_path)
        self._graph_engine = GraphSearchEngine(
            self._pipeline.graph_store,
            self._pipeline.metadata_store,
        )

        # Query expansion and reranking.
        self._query_expander = QueryExpander(self._pipeline.metadata_store)
        self._reranker = Reranker(model_name=self._settings.reranker_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, directory: Path) -> IndexStats:
        """Index all markdown files under *directory*.

        Delegates entirely to :class:`IndexingPipeline`.

        Parameters
        ----------
        directory:
            Root directory to scan.

        Returns
        -------
        IndexStats
            Summary statistics for the completed index.
        """
        return self._pipeline.index(directory)

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        rerank: bool = True,
        expand: bool = True,
    ) -> list[SearchResult]:
        """Search the index and return ranked results.

        Parameters
        ----------
        query:
            Natural-language search query.
        top_k:
            Maximum number of results to return.
        mode:
            Search strategy.  One of ``"semantic"``, ``"bm25"``,
            ``"graph"``, or ``"hybrid"`` (default).
        rerank:
            If ``True`` the top candidates are re-scored with a
            cross-encoder model before final ranking.
        expand:
            If ``True`` the query is expanded via :class:`QueryExpander`
            before being sent to the search engines.

        Returns
        -------
        list[SearchResult]
            Ranked search results, at most *top_k* items.
        """
        # -- Optional query expansion -----------------------------------------
        search_query = query
        if expand:
            expanded = self._query_expander.expand(query)
            search_query = expanded.expanded_text or query
            logger.debug(
                "Expanded query: %r -> %r",
                query,
                search_query,
            )

        # -- Determine pool size for reranking --------------------------------
        pool_size = self._settings.rerank_pool_size if rerank else top_k

        # -- Run search engines based on mode ---------------------------------
        semantic_hits: list[tuple[str, float]] = []
        bm25_hits: list[tuple[str, float]] = []
        graph_hits: list[tuple[str, float]] = []

        if mode in ("semantic", "hybrid"):
            semantic_hits = self._semantic_engine.search(search_query, top_k=pool_size)

        if mode in ("bm25", "hybrid"):
            self._bm25_engine.load_index()
            bm25_hits = self._bm25_engine.search(search_query, top_k=pool_size)

        if mode in ("graph", "hybrid"):
            # Graph search needs seed chunk IDs from other engines.
            seed_ids: list[str] = []
            if mode == "graph":
                # When running in graph-only mode, first obtain seeds from
                # both semantic and BM25 engines.
                sem_seeds = self._semantic_engine.search(search_query, top_k=pool_size)
                self._bm25_engine.load_index()
                bm25_seeds = self._bm25_engine.search(search_query, top_k=pool_size)
                seed_ids = list(
                    dict.fromkeys(
                        cid for cid, _ in sem_seeds + bm25_seeds
                    )
                )
            else:
                # Hybrid mode: use already-collected hits as seeds.
                seed_ids = list(
                    dict.fromkeys(
                        cid for cid, _ in semantic_hits + bm25_hits
                    )
                )
            graph_hits = self._graph_engine.search(
                search_query,
                top_k=pool_size,
                seed_chunk_ids=seed_ids,
            )

        # -- Fuse results via RRF ---------------------------------------------
        engine_results: dict[str, list[tuple[str, float]]] = {}
        if semantic_hits:
            engine_results["semantic"] = semantic_hits
        if bm25_hits:
            engine_results["bm25"] = bm25_hits
        if graph_hits:
            engine_results["graph"] = graph_hits

        fused = fuse_results(engine_results, k=self._settings.fusion_k)

        # -- Optional reranking -----------------------------------------------
        if rerank and fused:
            chunk_ids = [cid for cid, _ in fused[:pool_size]]
            fused = self._reranker.rerank(
                query,
                chunk_ids,
                self._pipeline.metadata_store,
                top_k=pool_size,
            )

        # -- Build final SearchResult list ------------------------------------
        results = build_search_results(
            fused[:top_k],
            per_engine=engine_results,
            metadata_store=self._pipeline.metadata_store,
            top_k=top_k,
        )

        return results

    def status(self) -> IndexStats | None:
        """Return the current index statistics, or ``None`` if not yet indexed.

        Returns
        -------
        IndexStats | None
            The most recent index statistics.
        """
        return self._pipeline.metadata_store.get_stats()

    def clear(self) -> None:
        """Remove all indexed data from every backing store."""
        self._pipeline.clear()
