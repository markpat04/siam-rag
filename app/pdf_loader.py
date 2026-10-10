"""PDF text extraction."""
from __future__ import annotations

from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class Document:
    doc_id: str      # filename shown to the user
    text: str
    n_pages: int


def load_pdf(path: str, doc_id: str | None = None) -> Document:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n".join(pages)
    # normalise whitespace, keep paragraph breaks
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return Document(
        doc_id=doc_id or path.split("/")[-1],
        text=text,
        n_pages=len(reader.pages),
    )
