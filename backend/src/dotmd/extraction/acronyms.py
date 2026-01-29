"""Acronym extraction from markdown content.

Detects and extracts acronym definitions using common patterns:
- "Full Name (ACRONYM)"
- "ACRONYM (Full Name)"
- "ACRONYM stands for Full Name"
- "Full Name, abbreviated as ACRONYM"
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dotmd.core.models import Chunk


# Common patterns for acronym definitions
_PATTERNS = [
    # "Security Information and Event Management (SIEM)"
    r"([A-Z][a-zA-Z\s&]+?)\s*\(([A-Z]{2,})\)",
    # "SIEM (Security Information and Event Management)"
    r"([A-Z]{2,})\s*\(([A-Z][a-zA-Z\s&]+?)\)",
    # "SIEM stands for Security Information and Event Management"
    r"([A-Z]{2,})\s+(?:stands for|is short for|means)\s+([A-Z][a-zA-Z\s&]+)",
    # "Mean Time To Identify, or MTTI"
    r"([A-Z][a-zA-Z\s]+?),\s+(?:or|abbreviated as)\s+([A-Z]{2,})",
    # Markdown table: "| **MTTD** | Mean Time to Detect |"
    r"\|\s*\*?\*?([A-Z]{2,})\*?\*?\s*\|\s*([A-Z][a-zA-Z\s]+?)\s*\|",
]


def extract_acronyms(text: str) -> dict[str, list[str]]:
    """Extract acronym definitions from text.

    Searches for common acronym definition patterns and returns a
    mapping of acronym to possible expansions.

    Parameters
    ----------
    text:
        The markdown content to scan.

    Returns
    -------
    dict[str, list[str]]
        Mapping of uppercase acronym to list of full-form expansions.

    Examples
    --------
    >>> text = "Security Information and Event Management (SIEM) is a platform..."
    >>> extract_acronyms(text)
    {'SIEM': ['Security Information and Event Management']}

    >>> text = "MTTI (Mean Time To Identify) measures detection speed"
    >>> extract_acronyms(text)
    {'MTTI': ['Mean Time To Identify']}
    """
    acronyms: dict[str, set[str]] = {}

    for pattern in _PATTERNS:
        for match in re.finditer(pattern, text):
            part1, part2 = match.groups()

            # Determine which is the acronym and which is the expansion
            if part1.isupper() and len(part1) >= 2:
                acronym = part1
                expansion = part2.strip()
            elif part2.isupper() and len(part2) >= 2:
                acronym = part2
                expansion = part1.strip()
            else:
                continue

            # Validate: acronym should match first letters of expansion
            if _is_valid_acronym(acronym, expansion):
                acronyms.setdefault(acronym, set()).add(expansion)

    # Convert sets to sorted lists
    return {k: sorted(v) for k, v in acronyms.items()}


def extract_acronyms_from_chunks(chunks: list[Chunk]) -> dict[str, list[str]]:
    """Extract all acronyms from a list of chunks.

    Parameters
    ----------
    chunks:
        List of chunks to scan.

    Returns
    -------
    dict[str, list[str]]
        Combined acronym dictionary from all chunks.
    """
    combined: dict[str, set[str]] = {}

    for chunk in chunks:
        chunk_acronyms = extract_acronyms(chunk.text)
        for acronym, expansions in chunk_acronyms.items():
            combined.setdefault(acronym, set()).update(expansions)

    return {k: sorted(v) for k, v in combined.items()}


def _is_valid_acronym(acronym: str, expansion: str) -> bool:
    """Validate that acronym matches first letters of expansion.

    Parameters
    ----------
    acronym:
        The uppercase acronym (e.g., "SIEM").
    expansion:
        The full expansion (e.g., "Security Information Event Management").

    Returns
    -------
    bool
        True if acronym reasonably matches expansion's first letters.
    """
    # Extract first letters of each word
    words = expansion.split()
    first_letters = "".join(w[0].upper() for w in words if w and w[0].isalpha())

    # Check if acronym is a subsequence of first letters
    # (allows for minor variations like "and", "of" being skipped)
    acronym_upper = acronym.upper()

    if acronym_upper == first_letters:
        return True

    # Allow acronym to be subsequence (e.g., "CIA" from "Confidentiality Integrity Availability")
    idx = 0
    for letter in acronym_upper:
        try:
            idx = first_letters.index(letter, idx) + 1
        except ValueError:
            return False

    return True
