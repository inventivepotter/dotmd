"""Main orchestration for HotPotQA evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

from dotmd.api.service import DotMDService
from dotmd.core.config import Settings

from eval.data_prep import (
    build_ground_truth,
    convert_to_markdown,
    load_hotpotqa,
    save_markdown_files,
)
from eval.metrics import aggregate_metrics, compute_retrieval_metrics
from eval.models import EvalResults
from eval.utils import cleanup_dir, create_temp_dir, print_results_table, save_results_json

logger = logging.getLogger(__name__)


def run_evaluation(
    data_file: Path | None = None,
    sample_size: int = 500,
    split: str = "validation",
    top_k: int = 10,
    search_mode: str = "hybrid",
    rerank: bool = True,
    expand: bool = True,
    output_dir: Path | None = None,
    seed: int = 42,
) -> EvalResults:
    """Run the full HotPotQA evaluation pipeline."""
    # 1. Load data (local file or download)
    examples = load_hotpotqa(data_file, split, sample_size, seed)

    # 2. Convert to markdown
    markdown_docs = convert_to_markdown(examples)
    ground_truth = build_ground_truth(examples)

    # 3. Temp directories
    temp_data_dir = create_temp_dir("dotmd_eval_data_")
    temp_index_dir = create_temp_dir("dotmd_eval_index_")

    try:
        # 4. Save markdown files
        file_paths = save_markdown_files(markdown_docs, temp_data_dir)
        logger.info("Saved %d markdown files to %s", len(file_paths), temp_data_dir)

        # 5. Create service with isolated index
        settings = Settings(
            data_dir=temp_data_dir,
            index_dir=temp_index_dir,
            extract_depth="structural",
        )
        service = DotMDService(settings)

        # 6. Index
        logger.info("Indexing documents...")
        stats = service.index(temp_data_dir)
        logger.info(
            "Indexed %d files, %d chunks",
            stats.total_files,
            stats.total_chunks,
        )

        # 7. Warmup
        service.warmup()

        # 8. Retrieve and compute metrics
        retrieval_results = []

        for ex in tqdm(examples, desc="Retrieving"):
            results = service.search(
                query=ex.question,
                top_k=top_k,
                mode=search_mode,
                rerank=rerank,
                expand=expand,
            )

            gt = ground_truth[ex.id]
            metrics = compute_retrieval_metrics(ex.id, results, gt)
            retrieval_results.append(metrics)

        # 9. Aggregate
        agg = aggregate_metrics(retrieval_results)

        results = EvalResults(
            dataset_size=len(examples),
            avg_doc_recall_at_k=agg.get("doc_recall", {}),
            avg_doc_mrr=agg.get("doc_mrr", 0.0),
            avg_doc_ndcg_at_k=agg.get("doc_ndcg", {}),
            avg_sent_recall_at_k=agg.get("sent_recall", {}),
            avg_sent_mrr=agg.get("sent_mrr", 0.0),
            avg_sent_ndcg_at_k=agg.get("sent_ndcg", {}),
            per_question_retrieval=retrieval_results,
            config={
                "sample_size": sample_size,
                "split": split,
                "top_k": top_k,
                "mode": search_mode,
                "rerank": rerank,
                "expand": expand,
            },
        )

        # 11. Output
        print_results_table(results)
        if output_dir:
            out_path = output_dir / "hotpotqa_results.json"
            save_results_json(results, out_path)
            logger.info("Results saved to %s", out_path)

        return results

    finally:
        cleanup_dir(temp_data_dir)
        cleanup_dir(temp_index_dir)
