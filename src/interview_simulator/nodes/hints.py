"""Progressive, on-demand hints (never the full answer)."""

from interview_simulator.models import Hint, Turn
from interview_simulator.nodes._util import ask_structured
from interview_simulator.nodes.interview import current_question
from interview_simulator.prompts import templates as P
from interview_simulator.state import InterviewState

MAX_HINTS = 3


def give_hint(state: InterviewState) -> dict:
    q = current_question(state)
    used = dict(state.get("hints_used", {}))
    level = used.get(q.id, 0) + 1
    if level > MAX_HINTS:
        text = ("I've already given you all the hints I can for this one. Give it your best shot, "
                "and it's fine to explain how far you got.")
        return {"transcript": [Turn(role="interviewer", kind="hint", text=text, question_id=q.id)]}

    payload = state.get("last_answer") or {}
    progress = " | ".join(x for x in [
        (payload.get("text") or "").strip(),
        f"code so far:\n{payload['code']}" if payload.get("code") else "",
    ] if x) or "(nothing yet)"
    previous = [t.text for t in state.get("transcript", []) if t.kind == "hint" and t.question_id == q.id]
    problem = ""
    if q.coding_problem:
        problem = f"CODING PROBLEM: {q.coding_problem.title}\n{q.coding_problem.statement}"
    prompt = P.HINT.format(
        level=level, category=q.category, question=state.get("current_prompt", q.text),
        problem=problem, progress=progress[:4000], previous=previous or "none",
        spoken_style=P.SPOKEN_STYLE,
    )
    hint: Hint = ask_structured("interviewer", Hint, prompt)
    used[q.id] = level
    return {
        "hints_used": used,
        "transcript": [Turn(role="interviewer", kind="hint", text=hint.hint, question_id=q.id)],
    }
