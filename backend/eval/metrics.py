"""Retrieval evaluation metrics at document and sentence level."""

from __future__ import annotations

import math

from dotmd.core.models import SearchResult

from eval.data_prep import GroundTruth
from eval.models import RetrievalMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc_relevant(result: SearchResult, gt: GroundTruth) -> bool:
    return result.file_path.stem in gt.doc_keys


def _sent_relevant(result: SearchResult, gt: GroundTruth) -> bool:
    """Check if the chunk's snippet contains any ground-truth sentence."""
    text = result.snippet
    return any(sent in text for sent in gt.sentences)


# ---------------------------------------------------------------------------
# Generic metric functions (work with any relevance function)
# ---------------------------------------------------------------------------

def _recall_at_k(
    retrieved: list[SearchResult],
    gt_items: set[str],
    is_relevant: callable,
    gt: GroundTruth,
    k: int,
) -> float:
    if not gt_items:
        return 0.0
    found = sum(1 for r in retrieved[:k] if is_relevant(r, gt))
    return min(found / len(gt_items), 1.0)


def _mrr(
    retrieved: list[SearchResult],
    is_relevant: callable,
    gt: GroundTruth,
) -> float:
    for i, r in enumerate(retrieved, 1):
        if is_relevant(r, gt):
            return 1.0 / i
    return 0.0


def _ndcg_at_k(
    retrieved: list[SearchResult],
    gt_items: set[str],
    is_relevant: callable,
    gt: GroundTruth,
    k: int,
) -> float:
    if not gt_items:
        return 0.0
    dcg = 0.0
    for i, r in enumerate(retrieved[:k], 1):
        if is_relevant(r, gt):
            dcg += 1.0 / math.log2(i + 1)
    ideal_count = min(len(gt_items), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    question_id: str,
    retrieved: list[SearchResult],
    gt: GroundTruth,
    k_values: list[int] | None = None,
) -> RetrievalMetrics:
    """Compute doc-level and sentence-level metrics for one question."""
    if k_values is None:
        k_values = [1, 5, 10, 20]

    # Document level
    doc_recall = {k: _recall_at_k(retrieved, gt.doc_keys, _doc_relevant, gt, k) for k in k_values}
    d_mrr = _mrr(retrieved, _doc_relevant, gt)
    doc_ndcg = {k: _ndcg_at_k(retrieved, gt.doc_keys, _doc_relevant, gt, k) for k in k_values}
    doc_found = len({r.file_path.stem for r in retrieved if r.file_path.stem in gt.doc_keys})

    # Sentence level
    sent_recall = {k: _recall_at_k(retrieved, gt.sentences, _sent_relevant, gt, k) for k in k_values}
    s_mrr = _mrr(retrieved, _sent_relevant, gt)
    sent_ndcg = {k: _ndcg_at_k(retrieved, gt.sentences, _sent_relevant, gt, k) for k in k_values}
    sent_found = sum(1 for r in retrieved if _sent_relevant(r, gt))

    return RetrievalMetrics(
        question_id=question_id,
        doc_recall_at_k=doc_recall,
        doc_mrr=d_mrr,
        doc_ndcg_at_k=doc_ndcg,
        doc_found=doc_found,
        doc_total=len(gt.doc_keys),
        sent_recall_at_k=sent_recall,
        sent_mrr=s_mrr,
        sent_ndcg_at_k=sent_ndcg,
        sent_found=sent_found,
        sent_total=len(gt.sentences),
    )


def aggregate_metrics(
    results: list[RetrievalMetrics],
) -> dict:
    """Aggregate metrics across all questions.

    Returns dict with keys: doc_recall, doc_mrr, doc_ndcg,
    sent_recall, sent_mrr, sent_ndcg.
    """
    if not results:
        return {}

    n = len(results)
    k_values = sorted(results[0].doc_recall_at_k.keys())

    return {
        "doc_recall": {k: sum(r.doc_recall_at_k.get(k, 0) for r in results) / n for k in k_values},
        "doc_mrr": sum(r.doc_mrr for r in results) / n,
        "doc_ndcg": {k: sum(r.doc_ndcg_at_k.get(k, 0) for r in results) / n for k in k_values},
        "sent_recall": {k: sum(r.sent_recall_at_k.get(k, 0) for r in results) / n for k in k_values},
        "sent_mrr": sum(r.sent_mrr for r in results) / n,
        "sent_ndcg": {k: sum(r.sent_ndcg_at_k.get(k, 0) for r in results) / n for k in k_values},
    }
