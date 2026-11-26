"""Local interview history (SQLite): save completed sessions, list them, load reports."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from interview_simulator.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    job_title TEXT,
    company TEXT,
    seniority TEXT,
    provider TEXT,
    model TEXT,
    overall_score INTEGER,
    hire_signal TEXT,
    ended_early INTEGER,
    report_json TEXT,
    report_markdown TEXT,
    transcript_json TEXT,
    evaluations_json TEXT
)
"""


def db_path() -> Path:
    return get_settings().data_path / "history.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def save_session(session_id: str, **fields) -> None:
    row = {"id": session_id, "created_at": datetime.now(UTC).isoformat(timespec="seconds"), **fields}
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    with _connect() as conn:
        conn.execute(f"INSERT OR REPLACE INTO sessions ({cols}) VALUES ({marks})", list(row.values()))


def list_sessions(limit: int = 100) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, created_at, job_title, company, seniority, provider, model, overall_score, "
            "hire_signal, ended_early FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not row:
        return None
    data = dict(row)
    for key in ("report_json", "transcript_json", "evaluations_json"):
        data[key] = json.loads(data[key]) if data.get(key) else None
    return data
