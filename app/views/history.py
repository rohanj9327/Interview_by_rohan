"""History page: past interviews, score trend, saved reports, and deleting interviews."""

import pandas as pd
import streamlit as st
from ui_common import checkpointer, fmt_signal, render_report, render_transcript, score_trend_chart

from interview_simulator import history
from interview_simulator.models import AnswerEvaluation, FinalReport, PlannedQuestion, Turn
from interview_simulator.session import delete_interview

ss = st.session_state


def _delete(ids: list[str]) -> None:
    for thread_id in ids:
        delete_interview(thread_id, checkpointer())
        current = ss.get("session")
        if current is not None and current.thread_id == thread_id:
            ss.pop("session", None)
            ss.pop("clip", None)
    ss.history_flash = f"Deleted {len(ids)} interview{'s' if len(ids) != 1 else ''}."


def ask_delete(ids: list[str]) -> None:
    ss.confirm_delete = list(ids)
    st.rerun()


def confirm_box(labels: dict[str, str]) -> None:
    """Inline confirmation for a pending delete."""
    ids = ss.get("confirm_delete")
    if not ids:
        return
    with st.container(border=True):
        st.warning("Permanently delete the report, transcript and saved progress for:")
        for i in ids:
            st.markdown(f"- {labels.get(i, i)}")
        c1, c2, _ = st.columns([1, 1, 3])
        if c1.button("Yes, delete", type="primary", key="confirm_delete_yes"):
            _delete(ids)
            ss.pop("confirm_delete", None)
            st.rerun()
        if c2.button("Cancel", key="confirm_delete_no"):
            ss.pop("confirm_delete", None)
            st.rerun()


st.title("📈 Interview history")
if ss.get("history_flash"):
    st.success(ss.pop("history_flash"))

rows = history.list_sessions(200)
if not rows:
    st.info("No completed interviews yet. Finish an interview to see it here.")
    st.stop()

labels = {
    r["id"]: f"{pd.to_datetime(r['created_at']).strftime('%b %d %H:%M')} · {r['job_title'] or 'Interview'}"
             + (f" @ {r['company']}" if r["company"] else "") + f" · {r['overall_score']}/100"
    for r in rows
}

scores = [r["overall_score"] for r in rows if r["overall_score"] is not None]
c1, c2, c3 = st.columns(3)
c1.metric("Interviews completed", len(rows))
c2.metric("Latest score", f"{rows[0]['overall_score']}/100",
          delta=(rows[0]["overall_score"] - rows[1]["overall_score"]) if len(rows) > 1 else None)
c3.metric("Best score", f"{max(scores)}/100" if scores else "n/a")

chart = score_trend_chart(rows)
if chart is not None:
    st.subheader("Score over time")
    st.altair_chart(chart, width="stretch")

df = pd.DataFrame(rows)
df["role"] = df["job_title"].fillna("") + df["company"].fillna("").map(lambda c: f" @ {c}" if c else "")
df["signal"] = df["hire_signal"].map(fmt_signal)
df["date"] = pd.to_datetime(df["created_at"]).dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M")
df["ended early"] = df["ended_early"].astype(bool)
table = df[["date", "role", "overall_score", "signal", "ended early", "model", "id"]].rename(
    columns={"overall_score": "score", "id": "session"})

st.subheader("All interviews")
st.caption("Select rows to delete interviews you don't need to keep.")
selection = st.dataframe(table, hide_index=True, width="stretch", key="history_table",
                         on_select="rerun", selection_mode="multi-row")
selected_ids = [table.iloc[i]["session"] for i in selection.selection.rows] if selection else []
if st.button(f"🗑 Delete selected ({len(selected_ids)})", disabled=not selected_ids, key="delete_selected"):
    ask_delete(selected_ids)
confirm_box(labels)

st.subheader("Open a report")
chosen = st.selectbox("Interview", list(labels), format_func=labels.get, label_visibility="collapsed")
data = history.get_session(chosen)
if data and data.get("report_json"):
    if st.button("🗑 Delete this interview", key=f"delete_{chosen}"):
        ask_delete([chosen])
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
