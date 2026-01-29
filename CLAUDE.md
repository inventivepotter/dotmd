# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

dotMD is a markdown knowledgebase search tool that combines three retrieval strategies:
- **Semantic search** — sentence-transformers embeddings + LanceDB vector store
- **BM25 keyword search** — rank_bm25 for exact term matching
- **Graph search** — LadybugDB knowledge graph with entity/relation traversal

Results are merged via Reciprocal Rank Fusion (RRF) and optionally reranked with a cross-encoder.

## Monorepo Structure

```
dotMD/
├── backend/              # Python package (src layout)
│   ├── pyproject.toml
│   └── src/dotmd/        # importable package
│       ├── core/         # models, config, exceptions
│       ├── ingestion/    # reader, chunker, pipeline
│       ├── extraction/   # structural, GLiNER NER
│       ├── storage/      # LanceDB, LadybugDB, SQLite
│       ├── search/       # semantic, BM25, graph, fusion, reranker, query expansion
│       ├── api/          # DotMDService facade + public types
│       └── cli.py        # Click CLI (thin wrapper over api/service.py)
├── data/                 # Sample markdown files for testing
└── README.md
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| Vector DB | LanceDB (embedded, file-based) |
| Graph DB | LadybugDB (embedded, forked from Kuzu) |
| Metadata DB | SQLite |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| NER | GLiNER (urchade/gliner_multi-v2.1) — zero-shot |
| BM25 | rank_bm25 |
| CLI | Click |
| Models | Pydantic v2 |

## Key Architecture Decisions

- **SOLID principles**: Protocol-based abstractions for storage, extractors, and search engines. Dependency injection in pipeline and service.
- **UI-agnostic API**: `DotMDService` in `api/service.py` is the public interface. CLI wraps it. Future web UI will import the same service.
- **Extraction layers**: Structural (always on) + GLiNER NER (default). Configurable via `--extract-depth`.
- **Search pipeline**: query → expand → 3 engines parallel → RRF fuse → cross-encoder rerank → top-K.

## Development

```bash
cd backend
pip install -e .
python -m spacy download en_core_web_sm  # if using spaCy fallback
dotmd index ../data/
dotmd search "your query"
```

## Storage Locations

All index data persists to `~/.dotmd/` by default:
- `~/.dotmd/lancedb/` — vector embeddings
- `~/.dotmd/graphdb/` — knowledge graph (LadybugDB)
- `~/.dotmd/metadata.db` — chunk metadata (SQLite)
- `~/.dotmd/bm25_index.pkl` — BM25 index

## Configuration

Settings via environment variables (prefix `DOTMD_`):
- `DOTMD_DATA_DIR` — markdown source directory
- `DOTMD_INDEX_DIR` — index storage directory
- `DOTMD_EXTRACT_DEPTH` — "structural" or "ner"
- `DOTMD_EMBEDDING_MODEL` — sentence-transformer model name
- `DOTMD_NER_ENTITY_TYPES` — comma-separated entity types for GLiNER

## When Modifying Code

- New storage backends: implement the Protocol from `storage/base.py`
- New extractors: implement `ExtractorProtocol` from `extraction/base.py`
- New search engines: implement `SearchEngineProtocol` from `search/base.py`
- All public APIs go through `api/service.py` — never expose internals directly
