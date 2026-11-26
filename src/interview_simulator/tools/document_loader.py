"""Load resume / JD documents (PDF, DOCX, TXT, MD) into plain text."""

from __future__ import annotations

import io
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def _pdf_to_text(data: bytes) -> str:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        # sort=True gives a sensible reading order for multi-column resumes
        return "\n".join(page.get_text("text", sort=True) for page in doc)


def _docx_to_text(data: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def load_bytes(filename: str, data: bytes) -> str:
    """Extract text from an uploaded file (used by the Streamlit UI)."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        text = _pdf_to_text(data)
    elif suffix == ".docx":
        text = _docx_to_text(data)
    elif suffix in {".txt", ".md"}:
        text = data.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type {suffix!r}; use one of {sorted(SUPPORTED)}")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    if not text.strip():
        raise ValueError(f"No text could be extracted from {filename!r} (scanned PDF?)")
    return text.strip()


def load_document(path: str | Path) -> str:
    path = Path(path).expanduser()
    return load_bytes(path.name, path.read_bytes())
