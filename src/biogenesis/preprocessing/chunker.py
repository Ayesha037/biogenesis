from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


@dataclass
class Chunk:
    chunk_id: str
    pmid: str
    text: str
    source_field: str


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def chunk_paper(
    pmid: str,
    title: str,
    abstract: str,
    chunk_sentences: int = 3,
    overlap: int = 1,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    if title:
        chunks.append(Chunk(chunk_id=f"{pmid}_title", pmid=pmid, text=title, source_field="title"))

    sentences = split_sentences(abstract)
    if not sentences:
        return chunks

    step = max(chunk_sentences - overlap, 1)
    idx = 0
    chunk_num = 0
    while idx < len(sentences):
        window = sentences[idx: idx + chunk_sentences]
        text = " ".join(window)
        chunks.append(
            Chunk(
                chunk_id=f"{pmid}_abs_{chunk_num}",
                pmid=pmid,
                text=text,
                source_field="abstract",
            )
        )
        chunk_num += 1
        idx += step

    return chunks
