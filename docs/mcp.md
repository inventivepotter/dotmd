# dotMD MCP Server

dotMD exposes its markdown knowledgebase search as an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server, so AI agents and chat interfaces can query your indexed markdown files directly.

## Setup

Install dotMD with MCP support:

```bash
cd backend
pip install -e .
```

Index your markdown files first:

```bash
dotmd index /path/to/your/markdown/files
```

## Running the Server

```bash
dotmd mcp
```

Or use the MCP inspector for interactive testing:

```bash
mcp dev backend/src/dotmd/mcp_server.py
```

## Agent Configuration

Add dotMD to your agent's MCP server config:

### Claude Code / Claude Desktop

Add to your `claude_desktop_config.json` or `.mcp.json`:

```json
{
  "mcpServers": {
    "dotmd": {
      "command": "python",
      "args": ["mcp"]
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "dotmd": {
      "command": "python",
      "args": ["mcp"]
    }
  }
}
```

## Tools

### `search`

Search the indexed markdown knowledgebase.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | *(required)* | Natural-language search query |
| `top_k` | int | `10` | Maximum number of results |
| `mode` | string | `"hybrid"` | Search strategy: `"semantic"`, `"bm25"`, `"graph"`, or `"hybrid"` |
| `rerank` | bool | `true` | Rerank results with a cross-encoder |

**Returns:** List of objects with `chunk_id`, `file_path`, `heading`, `snippet`, `score`, and `matched_engines`.

### `index`

Index all markdown files in a directory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory` | string | *(required)* | Path to the directory containing markdown files |

**Returns:** Object with `total_files`, `total_chunks`, `total_entities`, `total_edges`, and `last_indexed`.

### `status`

Get current index statistics. Takes no parameters.

**Returns:** Object with index stats, or a message indicating no index exists.

## Environment Variables

The server respects all standard dotMD configuration via environment variables (prefix `DOTMD_`):

| Variable | Description |
|----------|-------------|
| `DOTMD_INDEX_DIR` | Index storage directory (default: `~/.dotmd/`) |
| `DOTMD_EMBEDDING_MODEL` | Sentence-transformer model name |
| `DOTMD_EXTRACT_DEPTH` | `"structural"` or `"ner"` |

See the main [README](../README.md) for the full list.
