# dotMD

A markdown knowledgebase search tool that combines semantic search, keyword search, and knowledge graph traversal for high-accuracy retrieval at zero ongoing cost.

## How It Works

dotMD indexes your markdown files using three complementary search strategies:

1. **Semantic Search** — Embeds chunks with [sentence-transformers](https://www.sbert.net/) and stores vectors in [LanceDB](https://lancedb.github.io/lancedb/) for meaning-based retrieval
2. **BM25 Keyword Search** — Classic term-frequency scoring via [rank-bm25](https://github.com/dorianbrown/rank_bm25) for exact keyword matching
3. **Knowledge Graph Search** — Builds a graph of files, sections, entities, and tags in [LadybugDB](https://github.com/FalkorDB/LadybugDB) (an embedded Cypher graph database forked from Kuzu), then traverses connections to find related content

Results are merged using **Reciprocal Rank Fusion** and optionally reranked with a **cross-encoder** for maximum precision.

Everything runs locally. No API keys required. No ongoing costs.

## Installation

Requires Python 3.12+.

```bash
cd backend
pip install -e .
```

## Usage

### Index your markdown files

```bash
dotmd index /path/to/your/markdown/files
```

With custom entity types for NER:

```bash
dotmd index /path/to/files --entity-types "person,technology,concept,project"
```

### Search

```bash
dotmd search "how to deploy to production"
```

Search modes:

```bash
dotmd search "query" --mode hybrid      # All 3 engines (default)
dotmd search "query" --mode semantic    # Vector similarity only
dotmd search "query" --mode bm25       # Keyword matching only
dotmd search "query" --mode graph      # Graph traversal only
dotmd search "query" --no-rerank       # Skip cross-encoder reranking
dotmd search "query" --no-expand       # Skip query expansion
dotmd search "query" --top 5           # Limit results
```

### Index management

```bash
dotmd status    # Show index statistics
dotmd clear     # Delete the entire index
```

## Architecture

```
backend/src/dotmd/
├── core/          # Domain models (Pydantic), config, exceptions
├── ingestion/     # File discovery, markdown chunking, indexing pipeline
├── extraction/    # Entity/relation extraction (structural + GLiNER NER)
├── storage/       # LanceDB (vectors), LadybugDB (graph), SQLite (metadata)
├── search/        # Semantic, BM25, graph search, RRF fusion, reranking
├── api/           # DotMDService — UI-agnostic public API
└── cli.py         # Click CLI
```

### Storage

| Layer | Engine | Details |
|-------|--------|---------|
| Vector | LanceDB | Embedded, file-based, ANN search |
| Graph | LadybugDB | Embedded, Cypher queries, zero-config |
| Metadata | SQLite | Chunk text, headings, stats |

All storage is local at `~/.dotmd/`.

### Graph Schema

**Nodes**: File, Section, Entity, Tag

**Edges**: HAS_SECTION, PARENT_OF, LINKS_TO, HAS_TAG, MENTIONS, CO_OCCURS

Entities are extracted in two configurable layers:
- **Structural** (always on) — headings, wikilinks, tags, frontmatter, markdown links
- **NER** (default) — [GLiNER](https://github.com/urchade/GLiNER) zero-shot named entity recognition with customizable entity types

### Search Pipeline

```
query → expand → [semantic, BM25, graph] → RRF fusion → cross-encoder rerank → results
```

## Configuration

Environment variables (prefix `DOTMD_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DOTMD_INDEX_DIR` | `~/.dotmd` | Where index data is stored |
| `DOTMD_EXTRACT_DEPTH` | `ner` | `structural` or `ner` |
| `DOTMD_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `DOTMD_NER_ENTITY_TYPES` | person,organization,technology,concept,location | GLiNER entity types |
| `DOTMD_DEFAULT_TOP_K` | `10` | Default number of results |

## License

MIT
