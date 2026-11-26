"""High-level wrapper around the graph, shared by the CLI and the Streamlit UI."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

from langgraph.types import Command

from interview_simulator.graph import NODE_LABELS, build_graph, sqlite_checkpointer
from interview_simulator.models import DeliveryMetrics, InterviewConfig, Turn


class InterviewSession:
    def __init__(self, thread_id: str | None = None, checkpointer=None):
        self.graph = build_graph(checkpointer if checkpointer is not None else sqlite_checkpointer())
        self.thread_id = thread_id or uuid.uuid4().hex[:12]
        self.run_config = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 200}
        self._seen_turns = len(self.values.get("transcript", []))

    # ---- running -----------------------------------------------------------------

    def _run(self, inp: Any) -> Iterator[str]:
        """Run until the next interrupt (or the end), yielding progress labels as nodes start."""
        for event in self.graph.stream(inp, self.run_config, stream_mode="tasks"):
            if "input" in event and event.get("name") in NODE_LABELS:  # task start event
                yield NODE_LABELS[event["name"]]

    def start(self, resume_text: str, jd_text: str, config: InterviewConfig) -> Iterator[str]:
        inp = {"resume_text": resume_text, "jd_text": jd_text, "config": config, "transcript": [], "evaluations": []}
        yield from self._run(inp)

    def answer(self, text: str, mode: str = "text", delivery: DeliveryMetrics | None = None,
               code: str | None = None, language: str | None = None) -> Iterator[str]:
        payload = {"type": "answer", "text": text, "mode": mode,
                   "delivery": delivery.model_dump() if delivery else None,
                   "code": code, "language": language}
        yield from self._run(Command(resume=payload))

    def hint(self, text: str = "", code: str | None = None) -> Iterator[str]:
        yield from self._run(Command(resume={"type": "hint", "text": text, "code": code}))

    def end(self) -> Iterator[str]:
        yield from self._run(Command(resume={"type": "end"}))

    def retry(self) -> Iterator[str]:
        """Continue after a failed step (e.g. an API error); the checkpoint keeps progress."""
        yield from self._run(None)

    @property
    def needs_retry(self) -> bool:
        state = self.graph.get_state(self.run_config)
        return bool(state.values) and bool(state.next) and self.pending() is None

    # ---- inspection --------------------------------------------------------------

    @property
    def values(self) -> dict:
        return self.graph.get_state(self.run_config).values

    def pending(self) -> dict | None:
        """The interrupt payload if the graph is waiting for the candidate, else None."""
        state = self.graph.get_state(self.run_config)
        for task in state.tasks:
            if task.interrupts:
                return task.interrupts[0].value
        return None

    @property
    def started(self) -> bool:
        return bool(self.values)

    @property
    def done(self) -> bool:
        state = self.graph.get_state(self.run_config)
        return bool(state.values.get("final_report")) and not state.next

    def new_turns(self) -> list[Turn]:
        transcript = self.values.get("transcript", [])
        turns = transcript[self._seen_turns:]
        self._seen_turns = len(transcript)
        return turns

    def report_markdown(self) -> str:
        from interview_simulator.nodes.feedback import report_to_markdown

        return report_to_markdown(self.values)

    def vocabulary(self) -> list[str]:
        """Technical terms from the JD/resume to improve speech recognition."""
        v = self.values
        terms: list[str] = []
        if v.get("job"):
            terms += v["job"].tech_stack + v["job"].must_have
        if v.get("resume"):
            terms += v["resume"].skills
        import re

        pieces = [p.strip() for t in terms for p in re.split(r"[,;:()/]| and | or ", t)]
        seen, out = set(), []
        for t in pieces:
            if 2 <= len(t) <= 30 and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out[:50]


def list_paused_sessions(checkpointer, limit: int = 10) -> list[dict]:
    """Unfinished interviews stored in the SQLite checkpointer, newest first."""
    from interview_simulator import history

    conn = getattr(checkpointer, "conn", None)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT thread_id, MAX(checkpoint_id) AS last FROM checkpoints "
            "GROUP BY thread_id ORDER BY last DESC LIMIT ?", (limit * 3,)
        ).fetchall()
    except Exception:  # noqa: BLE001 - table may not exist yet
        return []
    finished = {r["id"] for r in history.list_sessions(500)}
    out = []
    for thread_id, _ in rows:
        if thread_id in finished:
            continue
        session = InterviewSession(thread_id=thread_id, checkpointer=checkpointer)
        v = session.values
        if not v.get("plan") or session.done:
            continue
        out.append({
            "id": thread_id,
            "job": v["job"].title if v.get("job") else "",
            "company": v["job"].company if v.get("job") else "",
            "progress": f"{v.get('current_index', 0) + 1}/{len(v['plan'])}",
        })
        if len(out) >= limit:
            break
    return out


def transcribe_answer(audio: bytes | str, vocabulary: list[str] | None = None) -> tuple[str, DeliveryMetrics]:
    """Speech -> (transcript text, delivery metrics). Used by the UI before calling answer()."""
    from interview_simulator.voice.delivery import compute_metrics
    from interview_simulator.voice.stt import transcribe

    transcript = transcribe(audio, vocabulary=vocabulary)
    return transcript.text, compute_metrics(transcript)
