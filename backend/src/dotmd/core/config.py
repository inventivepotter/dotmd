"""Application settings via pydantic-settings."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global configuration for dotMD.

    Values can be set via environment variables prefixed with DOTMD_,
    e.g. DOTMD_DATA_DIR=/path/to/md/files.
    """

    model_config = {"env_prefix": "DOTMD_"}

    # Paths
    data_dir: Path = Path(".")
    index_dir: Path = Path.home() / ".dotmd"

    # Embedding
    # Current: all-MiniLM-L6-v2 (384-dim, fast, lightweight)
    # For better quality: all-mpnet-base-v2 (768-dim) or gte-multilingual-base (768-dim)
    # Trade-off: ~2x larger embeddings, slower indexing, higher retrieval accuracy
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking
    max_chunk_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # Extraction
    extract_depth: Literal["structural", "ner"] = "ner"
    ner_entity_types: list[str] = [
        "person",
        "organization",
        "technology",
        "concept",
        "location",
        "object",
        "activity",
        "date_time",
    ]

    # Search
    default_top_k: int = 10
    fusion_k: int = 60  # RRF constant
    rerank_pool_size: int = 20  # candidates to rerank

    # Graph
    graph_max_hops: int = 2

    @property
    def lancedb_path(self) -> Path:
        return self.index_dir / "lancedb"

    @property
    def graph_db_path(self) -> Path:
        return self.index_dir / "graphdb"

    @property
    def sqlite_path(self) -> Path:
        return self.index_dir / "metadata.db"

    @property
    def bm25_path(self) -> Path:
        return self.index_dir / "bm25_index.pkl"
