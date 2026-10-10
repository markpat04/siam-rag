"""Thai-friendly chunking.

Thai has no spaces between words, but spaces DO separate phrases/sentences.
Strategy: split into paragraphs, then into phrases on spaces, then greedily
pack phrases into ~chunk_size character windows with phrase-aligned overlap.
Chunk boundaries therefore never cut mid-phrase.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str   # f"{doc_id}#c{i}"
    doc_id: str
    text: str
    index: int


def _phrases(paragraph: str) -> list[str]:
    # split on whitespace, keeping single spaces as phrase separators
    return [p for p in paragraph.split(" ") if p]


def chunk_text(doc_id: str, text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[Chunk]:
    # 1. paragraph -> phrases
    phrases: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if para:
            phrases.extend(_phrases(para))

    # 2. greedy pack phrases into windows
    windows: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for ph in phrases:
        # a single pathological phrase longer than the window: hard-split it
        while len(ph) > chunk_size:
            if cur:
                windows.append(cur)
                cur, cur_len = [], 0
            windows.append([ph[:chunk_size]])
            ph = ph[chunk_size - chunk_overlap:]
        if cur and cur_len + 1 + len(ph) > chunk_size:
            windows.append(cur)
            cur, cur_len = [], 0
        cur.append(ph)
        cur_len += len(ph) + 1
    if cur:
        windows.append(cur)

    # 3. build chunks with phrase-aligned overlap
    chunks: list[Chunk] = []
    for i, win in enumerate(windows):
        if i > 0:
            # carry tail of previous window, backtracked to a phrase boundary
            prev = " ".join(windows[i - 1])
            tail = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
            # drop a possibly partial leading phrase
            cut = tail.find(" ")
            tail_phrases = tail[cut + 1:] if cut != -1 else tail
            text_i = (tail_phrases + " " + " ".join(win)).strip() if tail_phrases else " ".join(win)
        else:
            text_i = " ".join(win)
        chunks.append(Chunk(chunk_id=f"{doc_id}#c{i}", doc_id=doc_id, text=text_i, index=i))
    return chunks
