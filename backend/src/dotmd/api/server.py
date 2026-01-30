"""FastAPI REST API for dotMD.

Thin HTTP layer over :class:`DotMDService`.  Start with::

    dotmd serve          # uses Click CLI
    uvicorn dotmd.api.server:app  # direct uvicorn
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Query
from pydantic import BaseModel

from dotmd.api.service import DotMDService
from dotmd.core.config import Settings
from dotmd.core.models import IndexStats, SearchResult

_service: DotMDService | None = None


def _get_service() -> DotMDService:
    assert _service is not None, "Service not initialised"
    return _service


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _service  # noqa: PLW0603
    _service = DotMDService(Settings(read_only=True))
    _service.warmup()
    yield
    _service = None


app = FastAPI(
    title="dotMD",
    description="Markdown knowledgebase search API",
    lifespan=_lifespan,
)


# -- Request / response models ------------------------------------------------

class IndexRequest(BaseModel):
    directory: str
    extract_depth: str = "ner"
    entity_types: list[str] | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    count: int


class MessageResponse(BaseModel):
    message: str


# -- Endpoints -----------------------------------------------------------------

@app.post("/index", response_model=IndexStats)
async def index(req: IndexRequest) -> IndexStats:
    """Index all markdown files under the given directory."""
    overrides: dict[str, object] = {"extract_depth": req.extract_depth}
    if req.entity_types is not None:
        overrides["ner_entity_types"] = req.entity_types
    service = DotMDService(Settings(**overrides))  # type: ignore[arg-type]
    return service.index(Path(req.directory))


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=100),
    mode: str = Query("hybrid", pattern="^(semantic|bm25|graph|hybrid)$"),
    rerank: bool = Query(True),
    expand: bool = Query(True),
) -> SearchResponse:
    """Search the indexed knowledgebase."""
    results = _get_service().search(
        query=q,
        top_k=top_k,
        mode=mode,
        rerank=rerank,
        expand=expand,
    )
    return SearchResponse(query=q, results=results, count=len(results))


@app.get("/status", response_model=IndexStats | None)
async def status() -> IndexStats | None:
    """Return current index statistics."""
    return _get_service().status()


@app.post("/clear", response_model=MessageResponse)
async def clear() -> MessageResponse:
    """Remove all indexed data."""
    _get_service().clear()
    return MessageResponse(message="Index cleared")


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API server via uvicorn."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)
