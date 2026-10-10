"""Gradio UI for SiamRAG. Run with: python -m app.ui  (after starting the API)."""
from __future__ import annotations

import os

# Import first: sanitizes proxy env vars before gradio -> httpx import
# (some runtimes ship NO_PROXY with bracketed IPv6 that crashes httpx).
import app.config  # noqa: F401

import gradio as gr
import httpx

API = os.environ.get("SIAMRAG_API", "http://127.0.0.1:8000")


def do_upload(files):
    if not files:
        return "ยังไม่ได้เลือกไฟล์"
    out = []
    for f in files:
        name = os.path.basename(f)
        with open(f, "rb") as fh:
            r = httpx.post(f"{API}/upload", files={"file": (name, fh, "application/pdf")}, timeout=300)
        r.raise_for_status()
        info = r.json()
        out.append(f"✅ {info['doc_id']}: {info['pages']} หน้า, {info['chunks']} chunks (รวม {info['indexed_chunks']})")
    return "\n".join(out)


def do_ask(question, top_k):
    if not question.strip():
        return "พิมพ์คำถามก่อนครับ", ""
    r = httpx.post(f"{API}/ask", json={"question": question, "top_k": int(top_k)}, timeout=300)
    r.raise_for_status()
    data = r.json()
    cites_md = ""
    for c in data["citations"]:
        cites_md += (
            f"<details><summary><b>{c['chunk_id']}</b> (score {c['score']})</summary>\n\n"
            f"{c['text']}\n\n</details>\n\n"
        )
    header = f"*mode: {data['mode']} · เวลาตอบ {data['latency_ms']['total_ms']} ms*\n\n"
    return header + data["answer"], cites_md


with gr.Blocks(title="SiamRAG — ถามตอบเอกสารภาษาไทย") as demo:
    gr.Markdown("# 🇹🇭 SiamRAG — ถาม-ตอบเอกสารภาษาไทยพร้อมอ้างอิง\nอัปโหลด PDF แล้วถามได้เลย คำตอบอ้างอิง chunk ต้นฉบับทุกครั้ง")
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(label="อัปโหลด PDF", file_count="multiple", file_types=[".pdf"])
            up_btn = gr.Button("📤 Index เอกสาร")
            up_out = gr.Textbox(label="ผลการ index", lines=4)
            up_btn.click(do_upload, files, up_out)
        with gr.Column(scale=2):
            q = gr.Textbox(label="คำถาม (ภาษาไทย)", placeholder="เช่น พนักงานลาพักร้อนได้กี่วันต่อปี?")
            k = gr.Slider(1, 10, value=5, step=1, label="จำนวน chunks ที่ดึง (top-k)")
            ask_btn = gr.Button("🔍 ถาม", variant="primary")
            ans = gr.Markdown(label="คำตอบ")
            cites = gr.HTML(label="เอกสารอ้างอิง (กดขยาย)")
            ask_btn.click(do_ask, [q, k], [ans, cites])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_PORT", "7860")))
