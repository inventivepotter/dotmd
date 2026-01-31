"""Download HotPotQA and convert context paragraphs to markdown files."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from urllib.request import urlopen

from eval.models import HotPotQAExample
from eval.utils import sanitize_filename

logger = logging.getLogger(__name__)

HOTPOTQA_URLS = {
    "validation": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
}


def load_hotpotqa(
    data_file: Path | None = None,
    split: str = "validation",
    sample_size: int = 500,
    seed: int = 42,
) -> list[HotPotQAExample]:
    """Load HotPotQA from a local JSON file, or download if not provided."""
    if data_file:
        logger.info("Loading HotPotQA from %s ...", data_file)
        raw = json.loads(data_file.read_text(encoding="utf-8"))
    else:
        url = HOTPOTQA_URLS.get(split)
        if not url:
            raise ValueError(f"Unknown split: {split!r}. Use 'validation' or 'train'.")
        logger.info("Downloading HotPotQA %s split from %s ...", split, url)
        with urlopen(url) as resp:
            raw = json.loads(resp.read())

    logger.info("Loaded %d examples", len(raw))

    if sample_size < len(raw):
        rng = random.Random(seed)
        raw = rng.sample(raw, sample_size)

    examples: list[HotPotQAExample] = []
    for row in raw:
        sf = row["supporting_facts"]
        ctx = row["context"]
        examples.append(
            HotPotQAExample(
                id=row["_id"],
                question=row["question"],
                answer=row["answer"],
                type=row["type"],
                level=row["level"],
                supporting_facts_titles=[t for t, _ in sf],
                supporting_facts_sent_ids=[s for _, s in sf],
                context_titles=[t for t, _ in ctx],
                context_sentences=[sents for _, sents in ctx],
            )
        )
    logger.info("Loaded %d examples", len(examples))
    return examples


def convert_to_markdown(examples: list[HotPotQAExample]) -> dict[str, str]:
    """Convert context documents to markdown.

    Returns a mapping of sanitized title -> markdown content.
    Each unique document title produces one markdown file.
    """
    docs: dict[str, str] = {}
    for ex in examples:
        for title, sentences in zip(ex.context_titles, ex.context_sentences):
            key = sanitize_filename(title)
            if key in docs:
                continue
            lines = [f"# {title}\n"]
            for sent in sentences:
                lines.append(sent.strip())
            docs[key] = "\n\n".join(lines) + "\n"
    logger.info("Prepared %d unique documents", len(docs))
    return docs


def save_markdown_files(docs: dict[str, str], output_dir: Path) -> dict[str, Path]:
    """Write markdown docs to disk.

    Returns mapping of sanitized title -> file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, content in docs.items():
        path = output_dir / f"{key}.md"
        path.write_text(content, encoding="utf-8")
        paths[key] = path
    return paths


class GroundTruth:
    """Ground truth for one question at both doc and sentence level."""

    def __init__(self, doc_keys: set[str], sentences: set[str]) -> None:
        self.doc_keys = doc_keys
        self.sentences = sentences


def build_ground_truth(
    examples: list[HotPotQAExample],
) -> dict[str, GroundTruth]:
    """Map question ID to ground truth at doc and sentence level.

    doc_keys: set of sanitized filenames for supporting-fact documents.
    sentences: set of actual sentence strings from supporting facts,
    used to check if a retrieved chunk contains the right content.
    """
    # Build a lookup: title -> list[sentence_str]
    title_to_sents: dict[str, list[str]] = {}
    for ex in examples:
        for title, sents in zip(ex.context_titles, ex.context_sentences):
            if title not in title_to_sents:
                title_to_sents[title] = sents

    gt: dict[str, GroundTruth] = {}
    for ex in examples:
        doc_keys = {sanitize_filename(t) for t in ex.supporting_facts_titles}
        sent_strs: set[str] = set()
        for title, sent_id in zip(
            ex.supporting_facts_titles, ex.supporting_facts_sent_ids
        ):
            sents = title_to_sents.get(title, [])
            if 0 <= sent_id < len(sents):
                sent_strs.add(sents[sent_id].strip())
        gt[ex.id] = GroundTruth(doc_keys=doc_keys, sentences=sent_strs)
    return gt
