"""Graph-based search engine for dotMD.

Expands seed chunk IDs (typically supplied by the semantic or BM25
engines via the fusion layer) by traversing the knowledge graph and
scoring neighbouring section nodes.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from dotmd.storage.base import GraphStoreProtocol, MetadataStoreProtocol

logger = logging.getLogger(__name__)


class GraphSearchEngine:
    """Search engine that exploits the knowledge graph for relevance signals.

    Unlike :class:`SemanticSearchEngine` and :class:`BM25SearchEngine`,
    this engine does **not** operate on the raw query text.  Instead it
    requires a set of *seed* chunk IDs (produced by another engine) and
    discovers related sections by walking the graph.

    Parameters
    ----------
    graph_store:
        A graph store satisfying :class:`GraphStoreProtocol`.
    metadata_store:
        A metadata store satisfying :class:`MetadataStoreProtocol`.
    """

    def __init__(
        self,
        graph_store: GraphStoreProtocol,
        metadata_store: MetadataStoreProtocol,
    ) -> None:
        self._graph_store = graph_store
        self._metadata_store = metadata_store

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        seed_chunk_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Traverse the graph from *seed_chunk_ids* and score neighbours.

        For each seed chunk the engine calls
        :meth:`GraphStoreProtocol.get_neighbors` with ``max_hops=2``.
        Every discovered node is scored using:

        .. code-block:: text

            graph_score = sum(edge_weight / hop_distance ** 2)

        Scores are aggregated across all seeds so that nodes reachable
        from multiple seeds receive a boost.

        Parameters
        ----------
        query:
            The original query string.  Accepted for protocol
            compatibility but **not used** by this engine.
        top_k:
            Maximum number of results to return.
        seed_chunk_ids:
            Starting chunk IDs for graph traversal.  If ``None`` or
            empty, an empty list is returned immediately.

        Returns
        -------
        list[tuple[str, float]]
            A list of ``(chunk_id, score)`` pairs ordered by
            descending graph score.
        """
        if not seed_chunk_ids:
            return []

        # Aggregate scores: chunk_id -> cumulative graph score
        aggregated_scores: dict[str, float] = defaultdict(float)
        seed_set = set(seed_chunk_ids)

        for seed_id in seed_chunk_ids:
            # max_hops=2 walks: Section→Entity (hop 1) → Section (hop 2)
            # This is how entity-mediated chunk discovery works.
            neighbors = self._graph_store.get_neighbors(seed_id, max_hops=2)
            for node_id, _rel, weight in neighbors:
                if node_id != seed_id:
                    aggregated_scores[node_id] += weight

        # Exclude the seed chunks themselves so the fusion layer does not
        # double-count results already present from another engine.
        for sid in seed_set:
            aggregated_scores.pop(sid, None)

        # Filter to only valid chunk IDs (Section nodes).  The traversal
        # also reaches Entity/Tag/File nodes which aren't searchable
        # results — discard them by checking the metadata store.
        candidate_ids = list(aggregated_scores.keys())
        if candidate_ids:
            valid_chunks = self._metadata_store.get_chunks(candidate_ids)
            valid_ids = {c.chunk_id for c in valid_chunks}
            aggregated_scores = {
                cid: score
                for cid, score in aggregated_scores.items()
                if cid in valid_ids
            }

        # Sort by descending score and return top-k.
        sorted_results = sorted(
            aggregated_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return sorted_results[:top_k]
