"""Custom exception hierarchy for dotMD."""


class DotMDError(Exception):
    """Base exception for all dotMD errors."""


class IndexError(DotMDError):
    """Raised when indexing fails."""


class IndexNotFoundError(DotMDError):
    """Raised when no index exists for the requested operation."""


class ChunkingError(DotMDError):
    """Raised when markdown chunking fails."""


class StorageError(DotMDError):
    """Raised when a storage backend operation fails."""


class SearchError(DotMDError):
    """Raised when a search operation fails."""


class ExtractionError(DotMDError):
    """Raised when entity/relation extraction fails."""


class ConfigError(DotMDError):
    """Raised when configuration is invalid."""
