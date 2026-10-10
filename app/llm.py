"""LLM provider abstraction.

- OpenAICompatibleLLM: any chat-completions endpoint (OpenAI, Azure, vLLM,
  Ollama, LiteLLM proxy...) configured via LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.
- ExtractiveLLM: retrieval-only fallback used when no key is configured.
  It returns the top cited chunk verbatim, clearly labelled, so the app is
  fully usable (and honest) without spending a cent.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class Citation:
    doc_id: str
    chunk_id: str
    text: str
    score: float


class LLMProvider:
    name: str = "base"

    def generate(self, question: str, citations: list[Citation]) -> str:
        raise NotImplementedError


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions in Thai using ONLY the provided "
    "source excerpts. Rules:\n"
    "1. Answer in Thai.\n"
    "2. Base every claim on the excerpts; if the excerpts do not contain the answer, say "
    "\"ไม่พบข้อมูลในเอกสาร\" (not found in the documents).\n"
    "3. After each factual statement, append the citation id in brackets, e.g. [doc.pdf#c2].\n"
    "4. Keep the answer concise."
)


def _build_prompt(question: str, citations: list[Citation]) -> str:
    excerpts = "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in citations)
    return f"{SYSTEM_PROMPT}\n\n--- SOURCES ---\n{excerpts}\n\n--- QUESTION ---\n{question}\n\nAnswer:"


class OpenAICompatibleLLM(LLMProvider):
    name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def generate(self, question: str, citations: list[Citation]) -> str:
        prompt = _build_prompt(question, citations)
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


class ExtractiveLLM(LLMProvider):
    """Retrieval-only mode: no key, no cost. Returns the best chunk verbatim."""

    name = "retrieval-only"

    def generate(self, question: str, citations: list[Citation]) -> str:
        if not citations:
            return "ไม่พบข้อมูลในเอกสาร (retrieval-only mode: no relevant chunks found)"
        top = citations[0]
        return (
            "[โหมดดึงข้อมูลเท่านั้น — ยังไม่ได้ตั้งค่า LLM key]\n"
            f"ข้อความที่เกี่ยวข้องที่สุดจาก {top.chunk_id}:\n\n{top.text}"
        )
