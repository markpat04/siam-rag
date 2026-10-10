"""Eval harness for SiamRAG.

Runs WITHOUT an LLM key: retrieval metrics (recall@k) always work.
When LLM_* env vars are set, additionally runs a heuristic faithfulness check.

Usage:
    python eval.py
"""
from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from app.config import Settings
from app.rag import RAGPipeline

# Each item: question + the document it belongs to + a distinctive phrase that
# MUST appear in a retrieved chunk for the retrieval to count as a hit.
EVAL_ITEMS = [
    {"q": "พนักงานลาพักร้อนได้กี่วันต่อปี?", "doc": "leave-policy.pdf", "gold": "12 วันทำการต่อปี"},
    {"q": "ลาป่วยโดยได้รับค่าจ้างได้กี่วัน?", "doc": "leave-policy.pdf", "gold": "30 วันทำการต่อปี"},
    {"q": "ลากิจได้กี่วันต่อปี?", "doc": "leave-policy.pdf", "gold": "6 วันทำการต่อปี"},
    {"q": "ลาคลอดบุตรได้กี่วัน?", "doc": "leave-policy.pdf", "gold": "98 วันต่อครรภ์"},
    {"q": "ต้องยื่นใบลาพักร้อนล่วงหน้ากี่วัน?", "doc": "leave-policy.pdf", "gold": "3 วันทำการ"},
    {"q": "แพ็กเกจ Pro ของ SiamRAG ราคาเท่าไหร่?", "doc": "product-manual.pdf", "gold": "2,900 บาทต่อเดือน"},
    {"q": "อัปโหลดไฟล์ PDF ได้ขนาดสูงสุดเท่าไหร่?", "doc": "product-manual.pdf", "gold": "50 เมกะไบต์"},
    {"q": "ทดลองใช้ SiamRAG ฟรีกี่วัน?", "doc": "product-manual.pdf", "gold": "14 วัน"},
    {"q": "ค่าโดยสาร BTS เริ่มต้นเท่าไหร่?", "doc": "bangkok-guide.pdf", "gold": "17 บาท"},
    {"q": "วัดพระแก้วเปิดกี่โมง?", "doc": "bangkok-guide.pdf", "gold": "8:30 ถึง 15:30"},
]


def recall_at_k(pipeline: RAGPipeline, k: int) -> tuple[float, list[bool]]:
    hits = []
    for item in EVAL_ITEMS:
        res = pipeline.ask(item["q"], top_k=k)
        found = any(item["gold"] in c["text"] and c["doc_id"] == item["doc"]
                    for c in res["citations"])
        hits.append(found)
    return sum(hits) / len(hits), hits


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[\n。．.!?]+", text)
    return [p.strip() for p in parts if len(p.strip()) > 4]


def faithfulness(pipeline: RAGPipeline) -> tuple[float, int, int]:
    """Heuristic: % of answer sentences sharing a >=6-char substring with a cited chunk."""
    supported, total = 0, 0
    for item in EVAL_ITEMS:
        res = pipeline.ask(item["q"], top_k=5)
        chunks = [c["text"] for c in res["citations"]]
        for sent in split_sentences(res["answer"]):
            total += 1
            ok = any(
                any(len(sub) >= 6 and sub in chunk for sub in
                    [sent[i:i + 12] for i in range(0, len(sent) - 11, 6)])
                for chunk in chunks
            )
            supported += ok
    return (supported / total if total else 0.0), supported, total


def main() -> None:
    t0 = time.perf_counter()
    settings = Settings()
    # fresh in-memory pipeline over sample PDFs (no server needed)
    pipe = RAGPipeline(settings)
    sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample_pdfs")
    if not os.path.isdir(sample_dir) or not os.listdir(sample_dir):
        print("No sample PDFs found. Run: python scripts/make_sample_pdfs.py")
        sys.exit(1)
    for fn in sorted(os.listdir(sample_dir)):
        if fn.endswith(".pdf"):
            pipe.ingest_pdf(os.path.join(sample_dir, fn), doc_id=fn)
    print(f"Indexed {len(pipe.store)} chunks from {len(pipe.documents)} docs "
          f"| mode={pipe.llm.name} | embed_model={settings.embedding_model}")

    latencies = []
    for k in (1, 3, 5):
        t1 = time.perf_counter()
        r, hits = recall_at_k(pipe, k)
        dt = (time.perf_counter() - t1) / len(EVAL_ITEMS) * 1000
        latencies.append(dt)
        marks = "".join("✓" if h else "✗" for h in hits)
        print(f"recall@{k}: {r:.2f}  ({sum(hits)}/{len(hits)})  [{marks}]")

    print(f"avg ask latency: {sum(latencies) / len(latencies):.0f} ms/question (CPU)")

    if settings.llm_configured:
        f, s, t = faithfulness(pipe)
        print(f"faithfulness (heuristic, LLM mode): {f:.2f} ({s}/{t} sentences supported by citations)")
    else:
        print("faithfulness: skipped (no LLM key configured — set LLM_BASE_URL/LLM_API_KEY to enable)")

    print(f"eval done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
