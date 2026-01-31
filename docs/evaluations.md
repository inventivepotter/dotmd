# Evaluations

dotMD uses the [HotPotQA](https://hotpotqa.github.io/) multi-hop question-answering dataset to benchmark retrieval quality. The evaluation indexes each question's supporting documents as markdown files, runs queries through the dotMD search pipeline, and measures how well the retrieved results match the known ground-truth documents and sentences.

## Metrics

| Metric | Description |
|--------|-------------|
| **Recall@K** | Fraction of ground-truth items found in the top-K results |
| **MRR** | Mean Reciprocal Rank — average of 1/rank for the first relevant result |
| **NDCG@K** | Normalized Discounted Cumulative Gain — rank-aware relevance score |

Metrics are computed at both **document level** (did we retrieve the right file?) and **sentence level** (did we retrieve the right passage?).

## Latest Results (n=100)

**Config:** hybrid mode, rerank enabled, query expansion enabled, top_k=10, seed=42, embedding model: `BAAI/bge-small-en-v1.5`

### Document-Level Retrieval

| Metric | Value |
|--------|-------|
| Recall@1 | 0.460 |
| Recall@5 | 0.895 |
| Recall@10 | 0.955 |
| Recall@20 | 0.955 |
| MRR | 0.956 |
| NDCG@1 | 0.920 |
| NDCG@5 | 0.862 |
| NDCG@10 | 0.887 |

### Sentence-Level Retrieval

| Metric | Value |
|--------|-------|
| Recall@1 | 0.317 |
| Recall@5 | 0.594 |
| Recall@10 | 0.629 |
| Recall@20 | 0.629 |
| MRR | 0.817 |
| NDCG@1 | 0.740 |
| NDCG@5 | 0.605 |
| NDCG@10 | 0.620 |

### Summary

| Level | Recall@10 |
|-------|-----------|
| Document | **95.50%** |
| Sentence | **62.90%** |

## Previous Results (n=500, old embedding model)

These results used the previous `all-MiniLM-L6-v2` embedding model:

| Level | Recall@10 | MRR | NDCG@10 |
|-------|-----------|-----|---------|
| Document | 84.70% | 0.932 | 0.813 |
| Sentence | 54.32% | 0.736 | 0.545 |

## How to Run

### Prerequisites

Install the package in development mode from the `backend/` directory:

```bash
cd backend
pip install -e .
```

### Basic Usage

Run the evaluation with default settings (500 samples, hybrid mode):

```bash
cd backend
python -m eval
```

### Common Options

```bash
# Run with 100 samples
python -m eval -n 100

# Use a specific search mode (semantic, bm25, graph, or hybrid)
python -m eval --mode semantic

# Disable reranking or query expansion
python -m eval --no-rerank
python -m eval --no-expand

# Change top-K results per query
python -m eval -k 20

# Save results to a JSON file
python -m eval -o ./results/

# Use a local HotPotQA JSON file instead of downloading
python -m eval -f path/to/hotpot_dev.json

# Verbose logging
python -m eval -v

# Combine options
python -m eval -n 100 --mode hybrid -k 10 -o ./results/ -v
```

### Full CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `-n`, `--sample-size` | 500 | Number of examples to evaluate |
| `-f`, `--data-file` | None | Local HotPotQA JSON file (skips download) |
| `--split` | validation | Dataset split |
| `-k`, `--top-k` | 10 | Results returned per query |
| `--mode` | hybrid | Search mode: `semantic`, `bm25`, `graph`, `hybrid` |
| `--no-rerank` | false | Disable cross-encoder reranking |
| `--no-expand` | false | Disable query expansion |
| `-o`, `--output-dir` | None | Directory to save results JSON |
| `--seed` | 42 | Random seed for reproducibility |
| `-v`, `--verbose` | false | Enable debug logging |
