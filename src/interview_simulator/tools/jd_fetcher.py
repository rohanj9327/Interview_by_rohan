"""Resolve a job description from a URL, a file path, or raw text."""

from __future__ import annotations

from pathlib import Path

import httpx

from interview_simulator.tools.document_loader import load_document

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"


def fetch_jd(url: str) -> str:
    import trafilatura

    resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    text = trafilatura.extract(resp.text, include_comments=False, include_tables=True)
    if not text or len(text) < 200:
        raise ValueError(
            "Could not extract the job description from that URL (it may require login, "
            "e.g. LinkedIn). Please paste the job description text instead."
        )
    return text


def resolve_jd(value: str) -> str:
    value = value.strip()
    if value.startswith(("http://", "https://")):
        return fetch_jd(value)
    if len(value) < 500 and Path(value).expanduser().exists():
        return load_document(value)
    return value
