"""FAISS vector store (cosine similarity via normalised dot product)."""
from __future__ import annotations

import json
import os

import faiss
import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # exact search; fine at this scale
        self.chunks: list[Chunk] = []
        self._vectors: list[np.ndarray] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        assert vectors.shape[1] == self.dim
        self.index.add(vectors.astype(np.float32))
        self.chunks.extend(chunks)
        self._vectors.extend([v.astype(np.float32) for v in vectors])

    def remove_doc(self, doc_id: str) -> int:
        """Remove all chunks of a doc (e.g. re-upload); returns removed count."""
        keep = [(c, v) for c, v in zip(self.chunks, self._vectors) if c.doc_id != doc_id]
        removed = len(self.chunks) - len(keep)
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks, self._vectors = [], []
        if keep:
            cs, vs = zip(*keep)
            self.add(list(cs), np.stack(vs))
        return removed

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        k = min(top_k, self.index.ntotal)
        scores, idxs = self.index.search(query_vec.reshape(1, -1).astype(np.float32), k)
        return [(self.chunks[i], float(scores[0][j])) for j, i in enumerate(idxs[0])]

    # ---- persistence ----
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(
                [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "index": c.index}
                 for c in self.chunks],
                f, ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str, dim: int) -> "VectorStore":
        store = cls(dim)
        store.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.json"), encoding="utf-8") as f:
            for row in json.load(f):
                store.chunks.append(Chunk(**row))
        # IndexFlatIP stores vectors verbatim -> recover them for remove_doc()
        n = store.index.ntotal
        store._vectors = [store.index.reconstruct(i) for i in range(n)]
        assert len(store._vectors) == len(store.chunks)
        return store

    def __len__(self) -> int:
        return self.index.ntotal
