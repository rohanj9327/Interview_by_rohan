"""Shared Streamlit helpers: cached resources, charts, report rendering."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from interview_simulator.graph import sqlite_checkpointer
from interview_simulator.models import FinalReport
from interview_simulator.nodes.feedback import SIGNAL_LABEL, delivery_stats

# Single-series chart colour: categorical slot 1 (validated reference palette), per theme.
SERIES_1 = {"light": "#2a78d6", "dark": "#3987e5"}

CATEGORY_LABEL = {
    "resume": "Resume deep-dive", "fundamentals": "Fundamentals", "coding": "Live coding",
    "system_design": "System design", "behavioral": "Behavioral", "company": "Company",
}


@st.cache_resource
def checkpointer():
    """One SQLite checkpointer for the whole app (thread-safe connection)."""
    return sqlite_checkpointer()


def theme_type() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:  # noqa: BLE001 - older Streamlit
        return "light"


def series_color() -> str:
    return SERIES_1.get(theme_type(), SERIES_1["light"])


def fmt_signal(signal: str) -> str:
    return SIGNAL_LABEL.get(signal, signal)


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------


def per_question_chart(report: FinalReport, plan: list) -> alt.Chart | None:
    """Horizontal bars: score per question (single series, direct value labels)."""
    if not report.per_question:
        return None
    titles = {q.id: (q.coding_problem.title if q.coding_problem else q.text) for q in plan}
    cats = {q.id: CATEGORY_LABEL.get(q.category, q.category) for q in plan}
    df = pd.DataFrame([
        {
            "id": pq.question_id,
            "label": f"{pq.question_id} · {cats.get(pq.question_id, '')}",
            "question": titles.get(pq.question_id, "")[:120],
            "score": pq.score,
        }
        for pq in report.per_question
    ])
    color = series_color()
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=220, ticks=False, domain=False)),
        x=alt.X("score:Q", scale=alt.Scale(domain=[0, 10]), title="Score (out of 10)",
                axis=alt.Axis(grid=True, gridOpacity=0.25, tickCount=5, domain=False)),
        tooltip=[alt.Tooltip("id:N", title="Question"), alt.Tooltip("question:N", title="Text"),
                 alt.Tooltip("score:Q", title="Score /10")],
    )
    bars = base.mark_bar(color=color, cornerRadiusEnd=4, height=14)
    labels = base.mark_text(align="left", dx=6, fontSize=12).encode(
        text="score:Q", color=alt.value("currentColor")
    )
    return (bars + labels).properties(height=max(120, 34 * len(df)))


def score_trend_chart(rows: list[dict]) -> alt.Chart | None:
    """Line + points: overall score over time (single series)."""
    data = [r for r in rows if r.get("overall_score") is not None]
    if len(data) < 2:
        return None
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["created_at"])
    df["role"] = df["job_title"].fillna("") + df["company"].fillna("").map(lambda c: f" @ {c}" if c else "")
    df["signal"] = df["hire_signal"].map(fmt_signal)
    color = series_color()
    base = alt.Chart(df).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(grid=False, format="%b %d")),
        y=alt.Y("overall_score:Q", scale=alt.Scale(domain=[0, 100]), title="Overall score",
                axis=alt.Axis(gridOpacity=0.25, tickCount=5, domain=False)),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%b %d, %H:%M"),
                 alt.Tooltip("overall_score:Q", title="Score"),
                 alt.Tooltip("signal:N", title="Signal"), alt.Tooltip("role:N", title="Role")],
    )
    line = base.mark_line(color=color, strokeWidth=2)
    points = base.mark_point(color=color, filled=True, size=90)
    return (line + points).properties(height=260)


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def render_report(values: dict, markdown: str, key_prefix: str = "report") -> None:
    report: FinalReport = values["final_report"]
    evals = values.get("evaluations", [])
    stats = delivery_stats(evals)
    coding = [e for e in evals if e.tests_total]

    cols = st.columns(4)
    cols[0].metric("Overall score", f"{report.overall_score}/100")
    cols[1].metric("Hiring signal", fmt_signal(report.hire_signal))
    cols[2].metric("Average pace", f"{stats['avg_wpm']:.0f} wpm" if stats else "n/a",
                   help="Ideal speaking pace is about 130–160 words per minute")
    if coding:
        passed = sum(e.tests_passed or 0 for e in coding)
        total = sum(e.tests_total or 0 for e in coding)
        cols[3].metric("Coding tests passed", f"{passed}/{total}")
    else:
        cols[3].metric("Filler words", f"{stats['total_fillers']}" if stats else "n/a",
                       help="um, uh, basically, you know, …")

    chart = per_question_chart(report, values.get("plan", []))
    if chart is not None:
        st.subheader("Score by question")
        st.altair_chart(chart, use_container_width=True)
        with st.expander("Scores as a table"):
            st.dataframe(
                pd.DataFrame([pq.model_dump() for pq in report.per_question]),
                hide_index=True, use_container_width=True,
            )

    st.markdown(markdown.split("\n", 1)[1] if markdown.startswith("# ") else markdown)
    st.download_button("Download report (Markdown)", markdown, file_name="interview_feedback.md",
                       mime="text/markdown", key=f"{key_prefix}_dl")


def render_transcript(turns: list, height: int | None = None) -> None:
    box = st.container(height=height) if height else st.container()
    with box:
        for t in turns:
            if t.role == "interviewer":
                avatar = {"hint": "💡", "tip": "✅"}.get(t.kind, "🎙️")
                with st.chat_message("assistant", avatar=avatar):
                    prefix = {"hint": "**Hint:** ", "tip": "**Quick tip:** ", "follow_up": "**Follow-up:** "}.get(t.kind, "")
                    st.markdown(prefix + t.text)
            else:
                with st.chat_message("user", avatar="🧑‍💻"):
                    if t.code:
                        st.caption(f"Submitted {t.language} code")
                        st.code(t.code, language=t.language or "python")
                    st.markdown(("🎤 " if t.mode == "voice" else "") + t.text)
