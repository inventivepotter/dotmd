"""Pydantic models for evaluation data and results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class HotPotQAExample(BaseModel):
    """A single HotPotQA example."""

    id: str
    question: str
    answer: str
    type: str  # "comparison" or "bridge"
    level: str  # "easy", "medium", "hard"
    supporting_facts_titles: list[str]
    supporting_facts_sent_ids: list[int]
    context_titles: list[str]
    context_sentences: list[list[str]]


class RetrievalMetrics(BaseModel):
    """Retrieval metrics for a single question at both granularities."""

    question_id: str
    # Document-level: did we retrieve chunks from the right files?
    doc_recall_at_k: dict[int, float] = Field(default_factory=dict)
    doc_mrr: float = 0.0
    doc_ndcg_at_k: dict[int, float] = Field(default_factory=dict)
    doc_found: int = 0
    doc_total: int = 0
    # Sentence-level: did we retrieve chunks containing the right sentences?
    sent_recall_at_k: dict[int, float] = Field(default_factory=dict)
    sent_mrr: float = 0.0
    sent_ndcg_at_k: dict[int, float] = Field(default_factory=dict)
    sent_found: int = 0
    sent_total: int = 0


class EvalResults(BaseModel):
    """Aggregated evaluation results."""

    dataset_size: int
    # Document-level aggregates
    avg_doc_recall_at_k: dict[int, float] = Field(default_factory=dict)
    avg_doc_mrr: float = 0.0
    avg_doc_ndcg_at_k: dict[int, float] = Field(default_factory=dict)
    # Sentence-level aggregates
    avg_sent_recall_at_k: dict[int, float] = Field(default_factory=dict)
    avg_sent_mrr: float = 0.0
    avg_sent_ndcg_at_k: dict[int, float] = Field(default_factory=dict)
    per_question_retrieval: list[RetrievalMetrics] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
