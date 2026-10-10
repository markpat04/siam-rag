"""API tests via FastAPI TestClient (retrieval-only mode)."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = os.path.dirname(os.path.dirname(__file__))


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mode"] == "retrieval-only"
    assert body["indexed_chunks"] > 0


def test_upload_rejects_non_pdf(client):
    r = client.post("/upload", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_upload_pdf_and_ask(client):
    pdf = os.path.join(ROOT, "data", "sample_pdfs", "leave-policy.pdf")
    with open(pdf, "rb") as fh:
        r = client.post("/upload", files={"file": ("leave-policy.pdf", fh, "application/pdf")})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r = client.post("/ask", json={"question": "ลาป่วยได้กี่วัน?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and "citations" in body
    assert len(body["citations"]) == 3
    assert body["mode"] == "retrieval-only"
    # the gold fact must be among the citations
    assert any("30 วันทำการต่อปี" in c["text"] for c in body["citations"])
    for c in body["citations"]:
        assert {"doc_id", "chunk_id", "text", "score"} <= set(c.keys())


def test_ask_validation(client):
    r = client.post("/ask", json={"question": "", "top_k": 5})
    assert r.status_code == 422
