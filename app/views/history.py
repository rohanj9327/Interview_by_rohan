"""History page: past interviews, score trend, and saved reports."""

import pandas as pd
import streamlit as st
from ui_common import fmt_signal, render_report, render_transcript, score_trend_chart

from interview_simulator import history
from interview_simulator.models import AnswerEvaluation, FinalReport, PlannedQuestion, Turn

st.title("📈 Interview history")
rows = history.list_sessions(200)
if not rows:
    st.info("No completed interviews yet. Finish an interview to see it here.")
    st.stop()

scores = [r["overall_score"] for r in rows if r["overall_score"] is not None]
c1, c2, c3 = st.columns(3)
c1.metric("Interviews completed", len(rows))
c2.metric("Latest score", f"{rows[0]['overall_score']}/100",
          delta=(rows[0]["overall_score"] - rows[1]["overall_score"]) if len(rows) > 1 else None)
c3.metric("Best score", f"{max(scores)}/100" if scores else "n/a")

chart = score_trend_chart(rows)
if chart is not None:
    st.subheader("Score over time")
    st.altair_chart(chart, use_container_width=True)

df = pd.DataFrame(rows)
df["role"] = df["job_title"].fillna("") + df["company"].fillna("").map(lambda c: f" @ {c}" if c else "")
df["signal"] = df["hire_signal"].map(fmt_signal)
df["date"] = pd.to_datetime(df["created_at"]).dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M")
df["ended early"] = df["ended_early"].astype(bool)
st.subheader("All interviews")
st.dataframe(
    df[["date", "role", "overall_score", "signal", "ended early", "model", "id"]].rename(
        columns={"overall_score": "score", "id": "session"}),
    hide_index=True, use_container_width=True,
)

labels = {r["id"]: f"{pd.to_datetime(r['created_at']).strftime('%b %d %H:%M')} · {r['job_title'] or 'Interview'} · {r['overall_score']}/100" for r in rows}
chosen = st.selectbox("Open a report", list(labels), format_func=labels.get)
data = history.get_session(chosen)
if data and data.get("report_json"):
    values = {
        "final_report": FinalReport(**data["report_json"]),
        "evaluations": [AnswerEvaluation(**e) for e in data.get("evaluations_json") or []],
        "plan": [],
    }
    # Rebuild minimal plan info (question ids -> text) from evaluations for the chart
    seen = {}
    for e in values["evaluations"]:
        if not e.is_follow_up and e.question_id not in seen:
            seen[e.question_id] = PlannedQuestion(id=e.question_id, text=e.question_text, category=e.category)
    values["plan"] = list(seen.values())
    render_report(values, data.get("report_markdown") or "", key_prefix=f"hist_{chosen}")
    with st.expander("Full transcript"):
        render_transcript([Turn(**t) for t in data.get("transcript_json") or []])
