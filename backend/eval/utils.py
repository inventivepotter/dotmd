"""Shared utilities for the evaluation harness."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

from eval.models import EvalResults


def sanitize_filename(title: str) -> str:
    """Convert a document title to a safe filename (without extension)."""
    name = title.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s]+", "_", name)
    return name


def create_temp_dir(prefix: str = "dotmd_eval_") -> Path:
    """Create a temporary directory and return its path."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def cleanup_dir(path: Path) -> None:
    """Recursively delete a directory."""
    if path.exists():
        shutil.rmtree(path)


def save_results_json(results: EvalResults, output_path: Path) -> None:
    """Write evaluation results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(results.model_dump_json(indent=2))


def print_results_table(results: EvalResults) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 50)
    print("  HotPotQA Evaluation Results")
    print("=" * 50)
    print(f"  Dataset size: {results.dataset_size}")

    print()
    print("  Document-Level Retrieval:")
    for k in sorted(results.avg_doc_recall_at_k):
        print(f"    Recall@{k:<3d}  {results.avg_doc_recall_at_k[k]:.3f}")
    print(f"    MRR        {results.avg_doc_mrr:.3f}")
    for k in sorted(results.avg_doc_ndcg_at_k):
        print(f"    NDCG@{k:<3d}   {results.avg_doc_ndcg_at_k[k]:.3f}")

    print()
    print("  Sentence-Level Retrieval:")
    for k in sorted(results.avg_sent_recall_at_k):
        print(f"    Recall@{k:<3d}  {results.avg_sent_recall_at_k[k]:.3f}")
    print(f"    MRR        {results.avg_sent_mrr:.3f}")
    for k in sorted(results.avg_sent_ndcg_at_k):
        print(f"    NDCG@{k:<3d}   {results.avg_sent_ndcg_at_k[k]:.3f}")

    print()
    print("  Config:")
    for key, val in results.config.items():
        print(f"    {key}: {val}")
    print("=" * 50)
