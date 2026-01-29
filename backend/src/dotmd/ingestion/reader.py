"""File discovery and reading for markdown knowledge bases."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from dotmd.core.models import FileInfo

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)


def _extract_title(content: str, path: Path) -> str:
    """Extract a human-readable title from file content or fall back to the filename.

    Looks for the first top-level ``# heading`` in *content*.  If none is
    found the file stem (filename without extension) is returned instead.
    """
    match = _HEADING_RE.search(content)
    if match:
        return match.group(1).strip()
    return path.stem


def discover_files(directory: Path) -> list[FileInfo]:
    """Recursively discover all ``.md`` files under *directory*.

    Parameters
    ----------
    directory:
        Root directory to search.  Must exist and be a directory.

    Returns
    -------
    list[FileInfo]
        Sorted by file path for deterministic ordering.

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist.
    NotADirectoryError
        If *directory* is not a directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    results: list[FileInfo] = []
    for md_path in sorted(directory.rglob("*.md")):
        if not md_path.is_file():
            continue
        try:
            content = read_file(md_path)
            stat = md_path.stat()
            results.append(
                FileInfo(
                    path=md_path,
                    title=_extract_title(content, md_path),
                    last_modified=datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ),
                    size_bytes=stat.st_size,
                )
            )
        except OSError:
            logger.warning("Skipping unreadable file: %s", md_path, exc_info=True)

    logger.info("Discovered %d markdown files in %s", len(results), directory)
    return results


def read_file(path: Path) -> str:
    """Read and return the full text content of a file.

    Parameters
    ----------
    path:
        Path to the file to read.

    Returns
    -------
    str
        The file contents decoded as UTF-8.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    return path.read_text(encoding="utf-8")
