"""CLI entry point: python -m eval.run_hotpotqa"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import click

# Load .env from eval/ directory
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from eval.run_hotpotqa import run_evaluation


@click.command()
@click.option("-f", "--data-file", type=click.Path(exists=True, path_type=Path), default=None, help="Local HotPotQA JSON file (skips download)")
@click.option("-n", "--sample-size", default=500, help="Number of examples")
@click.option("--split", default="validation", help="Dataset split (ignored if --data-file given)")
@click.option("-k", "--top-k", default=10, help="Results per query")
@click.option(
    "--mode",
    default="hybrid",
    type=click.Choice(["semantic", "bm25", "graph", "hybrid"]),
)
@click.option("--no-rerank", is_flag=True, help="Disable cross-encoder reranking")
@click.option("--no-expand", is_flag=True, help="Disable query expansion")
@click.option("-o", "--output-dir", type=click.Path(path_type=Path), help="Save results JSON")
@click.option("--seed", default=42, help="Random seed")
@click.option("-v", "--verbose", is_flag=True, help="Debug logging")
def main(
    data_file: Path | None,
    sample_size: int,
    split: str,
    top_k: int,
    mode: str,
    no_rerank: bool,
    no_expand: bool,
    output_dir: Path | None,
    seed: int,
    verbose: bool,
) -> None:
    """Run HotPotQA evaluation on dotMD search."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results = run_evaluation(
        data_file=data_file,
        sample_size=sample_size,
        split=split,
        top_k=top_k,
        search_mode=mode,
        rerank=not no_rerank,
        expand=not no_expand,
        output_dir=output_dir,
        seed=seed,
    )
    click.echo(
        f"\nDone. Doc Recall@10: {results.avg_doc_recall_at_k.get(10, 0):.2%}"
        f" | Sent Recall@10: {results.avg_sent_recall_at_k.get(10, 0):.2%}"
    )


if __name__ == "__main__":
    main()
