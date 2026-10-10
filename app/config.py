"""Central configuration, driven by environment variables.

No secrets are ever hard-coded. If LLM_* vars are unset, the app boots and
serves in retrieval-only mode (extractive answers from cited chunks).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Settings:
    # Embedding model (multilingual, CPU-friendly). Downloaded once from HF Hub.
    embedding_model: str = field(default_factory=lambda: _get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"))
    # Chunking
    chunk_size: int = field(default_factory=lambda: int(_get("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(_get("CHUNK_OVERLAP", "100")))
    # Retrieval
    default_top_k: int = field(default_factory=lambda: int(_get("TOP_K", "5")))
    # LLM provider (OpenAI-compatible chat-completions API). Empty => retrieval-only.
    llm_base_url: str = field(default_factory=lambda: _get("LLM_BASE_URL"))
    llm_api_key: str = field(default_factory=lambda: _get("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: _get("LLM_MODEL", "gpt-4o-mini"))
    llm_timeout_s: float = field(default_factory=lambda: float(_get("LLM_TIMEOUT_S", "60")))
    # Persistence
    data_dir: str = field(default_factory=lambda: _get("DATA_DIR", "data"))
    hf_home: str = field(default_factory=lambda: _get("HF_HOME", ""))

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key)

    @property
    def index_dir(self) -> str:
        return os.path.join(self.data_dir, "index")


def _sanitize_proxy_env() -> None:
    """Drop bracketed IPv6 literals (e.g. "[::1]") from NO_PROXY/no_proxy.

    Some runtimes export NO_PROXY with bracketed IPv6 addresses, which makes
    httpx raise `InvalidURL: Invalid port` at import time (gradio imports
    httpx eagerly). The unbracketed forms (::1) remain, so behaviour is kept.
    """
    for var in ("NO_PROXY", "no_proxy"):
        val = os.environ.get(var)
        if not val:
            continue
        parts = [p for p in val.split(",") if not (p.startswith("[") and p.endswith("]"))]
        os.environ[var] = ",".join(parts)


_sanitize_proxy_env()  # must run before any httpx/gradio import


settings = Settings()
