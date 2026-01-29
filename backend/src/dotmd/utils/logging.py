"""Structured logging setup for dotMD."""

from __future__ import annotations

import logging
import sys


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the dotmd logger."""
    logger = logging.getLogger("dotmd")
    if logger.handlers:
        return logger

    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
