"""Embedding wrapper (sentence-transformers, CPU)."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        # device="cpu" keeps this deployable anywhere; the MiniLM models are fast enough.
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim: int = self.model.get_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,   # -> cosine similarity == dot product
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
