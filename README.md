# SiamRAG 🇹🇭 — Thai Document Q&A with Citations

Upload Thai PDFs, ask questions in Thai, get answers with cited source chunks.
Runs **fully on CPU with zero API cost** in retrieval-only mode; plug in any
OpenAI-compatible LLM endpoint when you want generative answers.

## Problem

Thai small businesses sit on Thai-language PDFs (HR policies, product manuals,
price lists) and staff waste time hunting for answers. Generic chatbots
hallucinate; keyword search misses paraphrases. SiamRAG is a **retrieval-first**
Q&A app: every answer is grounded in cited chunks, and an **eval harness**
measures retrieval quality with real numbers instead of vibes.

## Architecture

```
┌─────────┐   ┌──────────┐   ┌──────────────────────────────┐   ┌─────────┐
│ PDF     │──▶│ Chunking │──▶│ paraphrase-multilingual-     │──▶│ FAISS   │
│ upload  │   │ (TH-aware│   │ MiniLM-L12-v2 (384-d, CPU)   │   │ IndexFlatIP
└─────────┘   │ ~500 ch) │   └──────────────────────────────┘   │ (cosine)│
              └──────────┘                                      └────┬────┘
                                                                     │ top-k
                                                                     ▼
┌──────────┐   ┌─────────────────────────────────────────────────────────────┐
│ Gradio UI│◀──│ FastAPI:  GET /health · POST /upload · POST /ask              │
│ :7860    │   │   ┌──────────────────┐      ┌────────────────────────────┐   │
└──────────┘   │   │ LLM key set?     │─yes─▶│ OpenAI-compatible LLM      │   │
               │   │                  │      │ (base_url/api_key/model)   │   │
               │   └──────────────────┘      └────────────────────────────┘   │
               │            │ no                                                 │
               │            ▼                                                    │
               │   ┌──────────────────┐                                         │
               │   │ Retrieval-only:  │  extractive snippet from top chunk,     │
               │   │ ExtractiveLLM    │  clearly labelled — $0 cost             │
               │   └──────────────────┘                                         │
               └─────────────────────────────────────────────────────────────┘
```

**Key design decisions**
- **Multilingual MiniLM-L12-v2** (278 MB, 384 dims): small enough for CPU, good
  Thai/English retrieval. Swappable via `EMBEDDING_MODEL`.
- **FAISS `IndexFlatIP` on L2-normalised embeddings** = exact cosine search;
  right choice at this scale (no ANN approximation error to debug).
- **Thai-aware chunking**: Thai has no inter-word spaces, so the chunker packs
  whole paragraphs into ~500-char windows with 100-char overlap instead of
  word-splitting.
- **Provider interface** (`LLMProvider`): `OpenAICompatibleLLM` vs
  `ExtractiveLLM`. The app boots and serves without any key — the honest
  default for a $0 portfolio demo.
- **Re-upload replaces** old chunks for the same `doc_id` (no duplicates).

## Measured results

Eval set: **10 Thai Q&A pairs** over 3 sample PDFs
(`leave-policy.pdf`, `product-manual.pdf`, `bangkok-guide.pdf`), run with
`python eval.py` on CPU, **no LLM key**:

| Metric | Result |
|---|---|
| retrieval recall@1 | **0.70** (7/10) |
| retrieval recall@3 | **1.00** (10/10) |
| retrieval recall@5 | **1.00** (10/10) |
| avg ask latency (embed + FAISS + answer) | **≈80 ms** / question (CPU) |
| pytest | **8/8 passed** |

Notes: recall@1 = 0.70 is honest for a 278 MB multilingual model on Thai —
top-3 is perfect on this set, which is what the UI shows by default.
The faithfulness check runs only when an LLM key is configured (heuristic:
% of answer sentences sharing a ≥6-char substring with cited chunks).

## How to run (one command)

```bash
# 0. Python 3.12, then:
python -m venv .venv && .venv/bin/pip install -r requirements.txt
#    (torch CPU is installed from the CPU-only index — see requirements.txt)

# 1. Sample PDFs (regenerates data/sample_pdfs/*.pdf)
.venv/bin/python scripts/make_sample_pdfs.py

# 2. Eval (no key needed) — prints the table above
.venv/bin/python eval.py

# 3. Tests
.venv/bin/python -m pytest tests/ -q

# 4. API (auto-indexes sample PDFs on first boot)
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
#    GET  /health   -> {"status":"ok","mode":"retrieval-only",...}
#    POST /upload   (multipart field "file", .pdf only)
#    POST /ask      {"question":"...","top_k":5}
#      -> {"answer","citations":[{"doc_id","chunk_id","text","score"}],
#          "mode","latency_ms"}

# 5. Gradio UI (needs the API running)
SIAMRAG_API=http://127.0.0.1:8000 .venv/bin/python -m app.ui
#    open http://127.0.0.1:7860
```

With Docker:

```bash
docker build -t siamrag .
docker run -p 8000:8000 siamrag
# optional LLM: -e LLM_BASE_URL=... -e LLM_API_KEY=... -e LLM_MODEL=...
```

Enable the LLM (any OpenAI-compatible endpoint — OpenAI, Azure, vLLM, Ollama):

```bash
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="sk-..."   # never commit this; use env vars / secrets manager
export LLM_MODEL="gpt-4o-mini"
```

## Deploying (deploy-ready, not published)

- The image is CPU-only and stateless except `data/index/` (mount a volume
  to persist the FAISS index).
- Never bake `LLM_API_KEY` into the image — inject at runtime
  (`docker run -e`, Cloud Run / Fly.io / Railway secrets).
- Cheap targets: Fly.io (shared CPU), Railway, or any VPS with 2 GB RAM.
  First boot downloads the ~470 MB embedding model once (baked into the
  Docker image already).

## Honest limitations

1. **No LLM key = extractive answers, not fluent Thai prose.** Retrieval-only
   mode returns the best chunk verbatim. Great for $0 evaluation; not a
   chatbot experience. This is deliberate and labelled in the UI.
2. **Scanned/image PDFs are not readable** — pypdf extracts embedded text
   only; no OCR (Tesseract) yet.
3. **Eval set is tiny (10 questions, 3 docs).** recall@3 = 1.00 reflects the
   demo corpus, not production robustness. Grow the set before trusting it.
4. **Faithfulness metric is a heuristic** (substring overlap), not an NLI
   model — it can over-credit paraphrases and under-credit true abstractions.
5. **Single-process FAISS** — fine for thousands of chunks; needs sharding/
   a managed vector DB beyond that.
6. Chunking is character-based; very long Thai paragraphs get hard-split.

---

## สรุปภาษาไทย

**SiamRAG** คือเว็บแอปถาม-ตอบเอกสารภาษาไทย: อัปโหลด PDF → ระบบค้นหาข้อความที่เกี่ยวข้อง
→ ตอบพร้อมอ้างอิงแหล่งที่มาทุกครั้ง จุดเด่นคือ **eval harness** ที่วัดคุณภาพการค้นคืน
ด้วยตัวเลขจริง (recall@1 = 0.70, recall@3 = 1.00, ตอบใน ~80 ms บน CPU)
แทนการอ้างลอย ๆ และ **โหมดไม่ใช้ LLM key** ที่รันได้ฟรีทั้งระบบ (ตอบแบบดึงข้อความต้นฉบับ)
— อยากได้คำตอบภาษาธรรมชาติเมื่อไหร่ค่อยเสียบ LLM เจ้าไหนก็ได้ผ่าน OpenAI-compatible API
