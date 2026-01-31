"""CLI interface for dotMD — thin wrapper over DotMDService."""

from __future__ import annotations

from pathlib import Path

import click

from dotmd.api.service import DotMDService
from dotmd.core.config import Settings
from dotmd.utils.logging import setup_logging


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """dotMD — Search your markdown knowledgebase."""
    setup_logging(verbose=verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


def _get_service(**overrides: object) -> DotMDService:
    settings = Settings(**overrides)  # type: ignore[arg-type]
    return DotMDService(settings=settings)


@main.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--extract-depth",
    type=click.Choice(["structural", "ner"]),
    default="ner",
    help="Extraction depth for graph building.",
)
@click.option(
    "--entity-types",
    default=None,
    help="Comma-separated GLiNER entity types (e.g. 'person,technology,concept').",
)
def index(directory: Path, extract_depth: str, entity_types: str | None) -> None:
    """Index all markdown files in DIRECTORY."""
    overrides: dict[str, object] = {"extract_depth": extract_depth}
    if entity_types:
        overrides["ner_entity_types"] = [t.strip() for t in entity_types.split(",")]

    service = _get_service(**overrides)
    click.echo(f"Indexing {directory}...")
    stats = service.index(directory)
    click.echo(
        f"Done. {stats.total_files} files, {stats.total_chunks} chunks, "
        f"{stats.total_entities} entities, {stats.total_edges} edges."
    )


@main.command()
@click.argument("query")
@click.option("--top", "-n", default=10, help="Number of results to return.")
@click.option(
    "--mode",
    type=click.Choice(["semantic", "bm25", "graph", "hybrid"]),
    default="hybrid",
    help="Search mode.",
)
@click.option("--no-rerank", is_flag=True, help="Skip cross-encoder reranking.")
@click.option("--no-expand", is_flag=True, help="Skip query expansion.")
def search(query: str, top: int, mode: str, no_rerank: bool, no_expand: bool) -> None:
    """Search the indexed knowledgebase."""
    service = _get_service(read_only=True)
    results = service.search(
        query=query,
        top_k=top,
        mode=mode,
        rerank=not no_rerank,
        expand=not no_expand,
    )

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        click.echo(f"\n{'─' * 60}")
        click.echo(f"  [{i}] {r.file_path}")
        if r.heading_path:
            click.echo(f"      {r.heading_path}")
        click.echo(f"      Score: {r.fused_score:.4f}  Engines: {', '.join(r.matched_engines)}")
        click.echo(f"      {r.snippet}")


@main.command()
def status() -> None:
    """Show index statistics."""
    service = _get_service(read_only=True)
    stats = service.status()

    if stats is None:
        click.echo("No index found. Run `dotmd index <directory>` first.")
        return

    click.echo(f"Files:    {stats.total_files}")
    click.echo(f"Chunks:   {stats.total_chunks}")
    click.echo(f"Entities: {stats.total_entities}")
    click.echo(f"Edges:    {stats.total_edges}")
    if stats.last_indexed:
        click.echo(f"Last indexed: {stats.last_indexed.isoformat()}")


@main.command()
def clear() -> None:
    """Clear the entire index."""
    if not click.confirm("This will delete the entire index. Continue?"):
        return
    service = _get_service()
    service.clear()
    click.echo("Index cleared.")


@main.command()
@click.option("--host", default="127.0.0.1", help="Bind host.")
@click.option("--port", "-p", default=8000, help="Bind port.")
def serve(host: str, port: int) -> None:
    """Start the REST API server."""
    from dotmd.api.server import main as run_server

    click.echo(f"Starting dotMD API on {host}:{port}")
    run_server(host=host, port=port)


@main.command()
def mcp() -> None:
    """Start the MCP (Model Context Protocol) server."""
    from dotmd.mcp_server import mcp as mcp_app

    click.echo("Starting dotMD MCP server...")
    mcp_app.run()


@main.command("mcp-config")
def mcp_config() -> None:
    """Print MCP client configuration JSON with absolute paths."""
    import json
    import shutil
    import sys

    dotmd_bin = shutil.which("dotmd") or sys.executable
    config = {
        "dotmd": {
            "command": str(Path(dotmd_bin).resolve()),
            "args": ["mcp"],
        }
    }
    click.echo(json.dumps(config, indent=2))


