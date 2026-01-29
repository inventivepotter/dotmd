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
            # get_neighbors returns (node_id, relation_type, weight) tuples.
            # We perform a multi-hop expansion by calling with max_hops=2;
            # the store is expected to return all reachable nodes within
            # that radius.  We approximate hop distance based on the
            # relation_type and weight returned.
            neighbors = self._graph_store.get_neighbors(seed_id, max_hops=2)

            # Build a per-seed scoring map.  Since get_neighbors returns
            # a flat list without explicit hop information, we infer hop
            # distance heuristically:
            #   - hop 1: direct neighbours (first pass)
            #   - hop 2: anything else returned within the radius
            #
            # We collect direct neighbours first, then treat the rest as
            # hop-2 nodes.
            direct_neighbor_ids: set[str] = set()
            for node_id, _rel, weight in neighbors:
                if node_id == seed_id:
                    continue
                # First encounter → assume hop 1
                direct_neighbor_ids.add(node_id)

            # Now score.  For a flat neighbour list the safest approach
            # is to assign hop 1 to all returned nodes (since the store
            # already limits the radius).  If the store provides
            # second-hop results they will naturally have lower edge
            # weights, which still downweights them in the sum.
            #
            # To incorporate hop distance more precisely we perform a
            # two-pass strategy: first query with max_hops=1 to identify
            # direct neighbours, then use the full list to derive hop 2
            # nodes.
            hop1_neighbors = self._graph_store.get_neighbors(seed_id, max_hops=1)
            hop1_ids: set[str] = set()
            hop1_weight: dict[str, float] = {}
            for node_id, _rel, weight in hop1_neighbors:
                if node_id == seed_id:
                    continue
                hop1_ids.add(node_id)
                # Keep the maximum weight if multiple edges exist.
                hop1_weight[node_id] = max(hop1_weight.get(node_id, 0.0), weight)

            # Score hop-1 nodes: weight / 1^2 = weight
            for node_id in hop1_ids:
                aggregated_scores[node_id] += hop1_weight[node_id]

            # Score hop-2 nodes: weight / 2^2 = weight / 4
            for node_id, _rel, weight in neighbors:
                if node_id == seed_id or node_id in hop1_ids:
                    continue
                aggregated_scores[node_id] += weight / 4.0

        # Exclude the seed chunks themselves so the fusion layer does not
        # double-count results already present from another engine.
        for sid in seed_set:
            aggregated_scores.pop(sid, None)

        # Sort by descending score and return top-k.
        sorted_results = sorted(
            aggregated_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return sorted_results[:top_k]
