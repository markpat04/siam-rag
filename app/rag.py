"""RAG pipeline: ingest PDFs -> chunk -> embed -> FAISS; ask -> retrieve -> answer."""
from __future__ import annotations

import os
import time

from .chunker import Chunk, chunk_text
from .config import Settings
from .embeddings import Embedder
from .llm import Citation, ExtractiveLLM, LLMProvider, OpenAICompatibleLLM
from .pdf_loader import Document, load_pdf
from .vectorstore import VectorStore


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = Embedder(settings.embedding_model)
        self.store = VectorStore(self.embedder.dim)
        self.documents: dict[str, Document] = {}
        if settings.llm_configured:
            self.llm: LLMProvider = OpenAICompatibleLLM(
                settings.llm_base_url, settings.llm_api_key,
                settings.llm_model, settings.llm_timeout_s,
            )
        else:
            self.llm = ExtractiveLLM()

    # ---- ingest ----
    def ingest_pdf(self, path: str, doc_id: str | None = None) -> dict:
        doc = load_pdf(path, doc_id)
        removed = self.store.remove_doc(doc.doc_id)  # re-upload replaces old chunks
        chunks = chunk_text(doc.doc_id, doc.text, self.settings.chunk_size, self.settings.chunk_overlap)
        vecs = self.embedder.encode([c.text for c in chunks])
        self.store.add(chunks, vecs)
        self.documents[doc.doc_id] = doc
        return {"doc_id": doc.doc_id, "pages": doc.n_pages, "chunks": len(chunks), "replaced": removed}

    # ---- ask ----
    def ask(self, question: str, top_k: int | None = None) -> dict:
        t0 = time.perf_counter()
        k = top_k or self.settings.default_top_k
        qvec = self.embedder.encode([question])[0]
        t_retrieved = time.perf_counter()
        hits = self.store.search(qvec, k)
        citations = [Citation(doc_id=c.doc_id, chunk_id=c.chunk_id, text=c.text, score=s) for c, s in hits]
        answer = self.llm.generate(question, citations)
        t_done = time.perf_counter()
        return {
            "answer": answer,
            "citations": [
                {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text, "score": round(c.score, 4)}
                for c in citations
            ],
            "mode": self.llm.name,
            "latency_ms": {
                "embed_ms": round((t_retrieved - t0) * 1000, 1),
                "total_ms": round((t_done - t0) * 1000, 1),
            },
        }

    # ---- persistence ----
    def save(self) -> None:
        self.store.save(self.settings.index_dir)

    def load(self) -> bool:
        idx = os.path.join(self.settings.index_dir, "index.faiss")
        if not os.path.exists(idx):
            return False
        self.store = VectorStore.load(self.settings.index_dir, self.embedder.dim)
        return True
