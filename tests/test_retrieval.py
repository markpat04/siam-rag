"""Retrieval-quality test: a known question must surface the right chunk in top-k."""
from __future__ import annotations

import os

import pytest

from app.config import Settings
from app.rag import RAGPipeline

ROOT = os.path.dirname(os.path.dirname(__file__))


@pytest.fixture(scope="module")
def pipeline():
    s = Settings()
    pipe = RAGPipeline(s)
    d = os.path.join(ROOT, "data", "sample_pdfs")
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".pdf"):
            pipe.ingest_pdf(os.path.join(d, fn), doc_id=fn)
    return pipe


@pytest.mark.parametrize("question,gold_phrase", [
    ("พนักงานลาพักร้อนได้กี่วันต่อปี?", "12 วันทำการต่อปี"),
    ("แพ็กเกจ Pro ราคาเท่าไหร่?", "2,900 บาทต่อเดือน"),
    ("วัดพระแก้วเปิดกี่โมง?", "8:30 ถึง 15:30"),
])
def test_known_question_retrieves_gold_chunk(pipeline, question, gold_phrase):
    res = pipeline.ask(question, top_k=3)
    assert any(gold_phrase in c["text"] for c in res["citations"]), (
        f"gold phrase {gold_phrase!r} not in top-3 for {question!r}"
    )


def test_scores_are_cosine_similarities(pipeline):
    res = pipeline.ask("ลากิจได้กี่วัน?", top_k=5)
    scores = [c["score"] for c in res["citations"]]
    assert all(-1.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)
