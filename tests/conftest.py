"""Shared fixtures: sample PDFs exist, no LLM key (retrieval-only mode)."""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

# Force retrieval-only mode for tests regardless of the developer's shell.
for var in ("LLM_BASE_URL", "LLM_API_KEY"):
    os.environ.pop(var, None)


@pytest.fixture(scope="session", autouse=True)
def sample_pdfs():
    from scripts.make_sample_pdfs import main as make_pdfs
    make_pdfs()
    return os.path.join(ROOT, "data", "sample_pdfs")
