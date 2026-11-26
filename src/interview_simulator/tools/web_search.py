"""Tavily web search with an on-disk cache (protects the free-tier quota)."""

from __future__ import annotations

import hashlib
import json
import logging
import time

from interview_simulator.config import get_settings

log = logging.getLogger(__name__)
CACHE_TTL_SEC = 7 * 24 * 3600


def _cache_file(query: str):
    key = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:24]
    folder = get_settings().data_path / "cache" / "tavily"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{key}.json"


def search(query: str, max_results: int = 5) -> list[dict]:
    """Return [{title, url, content}]; [] if Tavily is not configured or fails."""
    cache = _cache_file(query)
    if cache.exists() and time.time() - cache.stat().st_mtime < CACHE_TTL_SEC:
        return json.loads(cache.read_text())

    key = get_settings().tavily_api_key
    if not key:
        log.warning("TAVILY_API_KEY not set; skipping web research")
        return []
    try:
        from tavily import TavilyClient

        resp = TavilyClient(api_key=key).search(query, max_results=max_results, search_depth="basic")
    except Exception as exc:  # noqa: BLE001 - research is best-effort
        log.warning("Tavily search failed for %r: %s", query, exc)
        return []
    results = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": (r.get("content") or "")[:1500]}
        for r in resp.get("results", [])
    ]
    cache.write_text(json.dumps(results))
    return results
