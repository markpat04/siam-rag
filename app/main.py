"""FastAPI service for SiamRAG."""
from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import settings
from .rag import RAGPipeline

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = RAGPipeline(settings)
    if not pipeline.load():
        # no persisted index yet; also ingest bundled samples on first boot
        sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_pdfs")
        if os.path.isdir(sample_dir):
            for fn in sorted(os.listdir(sample_dir)):
                if fn.lower().endswith(".pdf"):
                    pipeline.ingest_pdf(os.path.join(sample_dir, fn), doc_id=fn)
            pipeline.save()
    yield


app = FastAPI(title="SiamRAG", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": pipeline.llm.name if pipeline else "unknown",
        "llm_configured": settings.llm_configured,
        "indexed_chunks": len(pipeline.store) if pipeline else 0,
        "embedding_model": settings.embedding_model,
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf files are accepted")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        info = pipeline.ingest_pdf(tmp_path, doc_id=file.filename)
        pipeline.save()
    finally:
        os.unlink(tmp_path)
    return JSONResponse({"ok": True, **info, "indexed_chunks": len(pipeline.store)})


@app.post("/ask")
def ask(req: AskRequest):
    if len(pipeline.store) == 0:
        raise HTTPException(400, "no documents indexed yet — upload a PDF first")
    return pipeline.ask(req.question, top_k=req.top_k)
