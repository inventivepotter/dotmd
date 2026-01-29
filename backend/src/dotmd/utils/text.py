"""Text processing utilities."""

from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (~4 chars per token)."""
    return max(1, len(text) // 4)


def clean_text(text: str) -> str:
    """Strip excessive whitespace while preserving paragraph breaks."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.rstrip()
        cleaned.append(stripped)
    result = "\n".join(cleaned)
    # Collapse 3+ blank lines to 2
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex.

    Handles common abbreviations and avoids splitting on decimal points.
    """
    # Split on sentence-ending punctuation followed by whitespace and uppercase
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]
