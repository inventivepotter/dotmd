# dotMD

A markdown knowledgebase search tool that combines semantic search, keyword search, and knowledge graph traversal for high-accuracy retrieval at zero ongoing cost.

## How It Works

dotMD indexes your markdown files using three complementary search strategies:

1. **Semantic Search** — Embeds chunks with [sentence-transformers](https://www.sbert.net/) and stores vectors in [LanceDB](https://lancedb.github.io/lancedb/) for meaning-based retrieval
2. **BM25 Keyword Search** — Classic term-frequency scoring via [rank-bm25](https://github.com/dorianbrown/rank_bm25) for exact keyword matching
3. **Knowledge Graph Search** — Builds a graph of files, sections, entities, and tags in [LadybugDB](https://github.com/FalkorDB/LadybugDB) (an embedded Cypher graph database forked from Kuzu), then traverses connections to find related content

Results are merged using **Reciprocal Rank Fusion** and optionally reranked with a **cross-encoder** for maximum precision.

Everything runs locally. No API keys required. No ongoing costs.

## Use Cases

### Give any AI agent instant access to your notes

Connect dotMD as an MCP server to Claude Code, Cursor, VS Code, or any MCP-compatible agent. Your entire markdown knowledge base becomes searchable context the agent can pull from mid-conversation — no copy-pasting, no uploading files.

### Search your personal knowledge base

If you keep learning notes, research summaries, or a digital garden in markdown, dotMD lets you search across all of it with semantic understanding — not just keyword matching. Ask "how does the transformer attention mechanism work" and find your notes even if they never use that exact phrase.

### Zero-cost RAG without LLM dependencies

Tools like Mem0 and Cognee use LLMs for indexing and retrieval, which means API costs on every query. dotMD runs entirely locally with open-source models — no API keys, no per-query fees, no data leaving your machine. If you can convert your documents to markdown, you have a fully functional RAG pipeline at zero ongoing cost.

### Feed project documentation to coding agents

Index your project's docs, ADRs, runbooks, and design documents. When a coding agent needs context about your architecture decisions or deployment process, it can retrieve the relevant sections directly instead of hallucinating or asking you to explain.

### Search across Obsidian / Logseq / Foam vaults

Any markdown-based note-taking tool works out of the box. Point dotMD at your vault directory and get hybrid search (semantic + keyword + graph) across years of accumulated notes, without being locked into any single app's search.

### Build a searchable knowledge base from any document format

Convert PDFs, Word docs, Confluence pages, or web articles to markdown (using tools like Pandoc, Docling, or Markitdown), then index them with dotMD. This gives you a private, searchable archive of everything you've read or collected.

### Team onboarding and internal knowledge sharing

Index your team's internal documentation, incident postmortem reports, or engineering handbooks. New team members — or AI agents assisting them — can search for answers without digging through wikis or Slack history.

### Research and literature review

Keep your paper summaries, reading notes, and annotations in markdown. dotMD's knowledge graph connects entities across documents, so you can discover relationships between concepts, authors, or methods that span your entire research corpus.

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

### REST API server

```bash
dotmd serve                          # Start on localhost:8000
dotmd serve --host 0.0.0.0 -p 9000  # Custom host and port
```

### MCP server

The MCP server uses stdio transport and is launched by an MCP client (Claude Code, VS Code, Cursor, OpenCode, etc.).

> **Important:** The API server and MCP server cannot run at the same time — they share a graph database that only supports a single connection.

To get the MCP config with absolute paths for your environment, run:

```bash
dotmd mcp-config
```

This outputs JSON you can paste directly into your client's MCP config:

```json
{
  "dotmd": {
    "command": "/absolute/path/to/.venv/bin/dotmd",
    "args": ["mcp"]
  }
}
```

If your MCP client runs from the project root, you can use a relative path instead:

```json
{
  "dotmd": {
    "command": "./backend/.venv/bin/dotmd",
    "args": ["mcp"]
  }
}
```

### Docker

```bash
# Build
docker compose build

# Index your files (place markdown in ./data/)
docker compose run api index /data

# Start the API server
docker compose up api

# Rebuild after code changes
docker compose up api --build
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
├── utils/         # Shared utilities
├── mcp_server.py  # MCP server (FastMCP, stdio transport)
└── cli.py         # Click CLI
```

### MCP Tools

The MCP server ([mcp_server.py](backend/src/dotmd/mcp_server.py)) exposes three tools via [FastMCP](https://github.com/modelcontextprotocol/python-sdk):

| Tool | Description |
|------|-------------|
| `search` | Query the indexed knowledgebase (supports semantic, BM25, graph, or hybrid mode with optional cross-encoder reranking) |
| `index` | Index all markdown files in a directory |
| `status` | Get current index statistics |

The server uses a lazy singleton `DotMDService` — ML models load once on first request and are reused across all subsequent calls.

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
| `DOTMD_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-transformer model |
| `DOTMD_NER_ENTITY_TYPES` | person,organization,technology,concept,location,object,activity,date_time | GLiNER entity types |
| `DOTMD_DEFAULT_TOP_K` | `10` | Default number of results |

## License

MIT
