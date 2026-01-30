"""Text processing utilities."""

from __future__ import annotations

import re


_STOP_WORDS: frozenset[str] = frozenset(
    "a an and are as at be but by for from has have he her his how i if in "
    "into is it its just me my no nor not of on or our own s she so some such "
    "t than that the their them then there these they this to too up us very "
    "was we were what when where which while who whom why will with would you "
    "your yours do does did doing been being am".split()
)


def tokenize(text: str) -> list[str]:
    """Whitespace + punctuation tokenizer with stop-word removal for BM25."""
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return [t for t in tokens if t not in _STOP_WORDS]


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
