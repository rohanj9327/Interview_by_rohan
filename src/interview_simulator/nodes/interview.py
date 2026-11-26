"""The interview loop: ask -> await (interrupt) -> evaluate -> follow-up / tip / next.

Interrupt payload (sent to the UI/CLI while waiting for the candidate):
    {"type": "await_answer", "question_id", "category", "prompt", "is_follow_up",
     "coding_problem": dict | None, "time_limit_sec", "hints_used", "index", "total"}

Resume payload (from the UI/CLI via Command(resume=...)):
    {"type": "answer", "text": str, "mode": "voice"|"text",
     "delivery": DeliveryMetrics-dict | None, "code": str | None, "language": "python"|"java"|None}
    {"type": "hint", "text": partial answer (optional), "code": partial code (optional)}
    {"type": "end"}
"""

from __future__ import annotations

from langgraph.types import interrupt

from interview_simulator.models import (
    AnswerEvaluation,
    AnswerReview,
    DeliveryMetrics,
    PlannedQuestion,
    Turn,
)
from interview_simulator.nodes._util import ask_structured
from interview_simulator.nodes.coding import review_coding
from interview_simulator.prompts import templates as P
from interview_simulator.state import InterviewState
from interview_simulator.tools.code_runner import run_code


def current_question(state: InterviewState) -> PlannedQuestion:
    return state["plan"][state["current_index"]]


def is_follow_up(state: InterviewState) -> bool:
    return state.get("follow_up_count", 0) > 0


# --------------------------------------------------------------------------------------
# ask / await
# --------------------------------------------------------------------------------------


def ask_question(state: InterviewState) -> dict:
    q = current_question(state)
    follow_up = state.get("pending_follow_up")
    if follow_up:
        turn = Turn(role="interviewer", kind="follow_up", text=follow_up, question_id=q.id)
        return {"transcript": [turn], "current_prompt": follow_up, "pending_follow_up": None}
    turns = []
    if state["current_index"] == 0 and not any(t.kind == "intro" for t in state.get("transcript", [])):
        turns.append(Turn(role="interviewer", kind="intro", text=state.get("intro", "")))
    turns.append(Turn(role="interviewer", kind="question", text=q.text, question_id=q.id))
    return {"transcript": turns, "current_prompt": q.text}


def await_answer(state: InterviewState) -> dict:
    q = current_question(state)
    config = state["config"]
    follow_up = is_follow_up(state)
    show_problem = q.category == "coding" and not follow_up
    payload = interrupt({
        "type": "await_answer",
        "question_id": q.id,
        "category": q.category,
        "prompt": state.get("current_prompt", q.text),
        "is_follow_up": follow_up,
        "coding_problem": q.coding_problem.model_dump() if show_problem and q.coding_problem else None,
        "time_limit_sec": config.coding_time_limit_sec if show_problem else config.answer_time_limit_sec,
        "hints_used": state.get("hints_used", {}).get(q.id, 0),
        "index": state["current_index"] + 1,
        "total": len(state["plan"]),
    })
    payload = payload or {}
    kind = payload.get("type", "answer")
    if kind == "end":
        return {"next_action": "end", "ended_early": True}
    if kind == "hint":
        return {"next_action": "hint", "last_answer": payload}
    text = (payload.get("text") or "").strip() or "(no answer given)"
    turn = Turn(
        role="candidate", kind="answer", text=text, question_id=q.id,
        mode=payload.get("mode", "text"), code=payload.get("code"), language=payload.get("language"),
    )
    return {"next_action": "evaluate", "last_answer": payload, "transcript": [turn]}


def route_after_await(state: InterviewState) -> str:
    return {"hint": "give_hint", "end": "final_feedback"}.get(state.get("next_action"), "evaluate_answer")


# --------------------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------------------


def _delivery_text(delivery: DeliveryMetrics | None) -> str:
    if not delivery:
        return "DELIVERY: typed answer (no voice metrics)."
    fillers = ", ".join(f"{w} x{n}" for w, n in delivery.filler_words.items()) or "none"
    return (
        f"VOICE DELIVERY METRICS: {delivery.wpm} words/min (ideal 130-160), "
        f"{delivery.filler_count} filler words ({fillers}; {delivery.fillers_per_min}/min), "
        f"{delivery.long_pauses} pauses over 2s (longest {delivery.longest_pause_sec}s), "
        f"spoke for {delivery.speaking_sec}s."
    )


def evaluate_answer(state: InterviewState) -> dict:
    q = current_question(state)
    job, config = state["job"], state["config"]
    payload = state.get("last_answer") or {}
    follow_up = is_follow_up(state)
    can_follow_up = state.get("follow_up_count", 0) < config.max_follow_ups
    hints = state.get("hints_used", {}).get(q.id, 0)
    answer_text = (payload.get("text") or "").strip() or "(no answer given)"
    delivery = DeliveryMetrics(**payload["delivery"]) if payload.get("delivery") else None
    prompt_text = state.get("current_prompt", q.text)
    code = payload.get("code")

    if q.category == "coding" and not follow_up and q.coding_problem:
        language = payload.get("language") or "python"
        run = run_code(q.coding_problem, code or "", language) if code and code.strip() else None
        if run is None:
            from interview_simulator.models import RunResult
            run = RunResult(language=language, compiled=False, compile_error="No code submitted")
        review = review_coding(state, q, code or "", language, answer_text, run, can_follow_up)
        correctness = 1 + round(4 * run.passed / run.total) if run.total else 1
        ev = AnswerEvaluation(
            question_id=q.id, category=q.category, is_follow_up=False,
            question_text=q.coding_problem.title, answer_text=answer_text,
            scores={"correctness": correctness, **review.scores.model_dump()},
            strengths=review.strengths, weaknesses=review.weaknesses, quick_tip=review.quick_tip,
            hints_used=hints, delivery=delivery, language=language,
            tests_passed=run.passed, tests_total=run.total,
            complexity_assessment=review.complexity_assessment,
        )
        needs, follow_q = review.needs_follow_up, review.follow_up_question
    else:
        follow_ctx = (
            f"(This is follow-up #{state.get('follow_up_count', 0)} to the main question: \"{q.text}\")\n"
            if follow_up else ""
        )
        prompt = P.EVALUATE_ANSWER.format(
            seniority=job.seniority, job_title=job.title, category=q.category, question=prompt_text,
            rubric="; ".join(q.rubric_hints) or "clear, specific, correct",
            follow_up_context=follow_ctx, mode="spoken, speech-to-text" if delivery else "typed",
            answer=answer_text, delivery=_delivery_text(delivery), hints=hints,
            follow_up_rule="" if can_follow_up else " — NOT allowed now, always false",
            spoken_style=P.SPOKEN_STYLE,
        )
        review: AnswerReview = ask_structured("evaluator", AnswerReview, prompt)
        ev = AnswerEvaluation(
            question_id=q.id, category=q.category, is_follow_up=follow_up,
            question_text=prompt_text, answer_text=answer_text,
            scores=review.scores.model_dump(), strengths=review.strengths,
            weaknesses=review.weaknesses, quick_tip=review.quick_tip, hints_used=hints,
            confidence_score=review.confidence_score, delivery_notes=review.delivery_notes,
            delivery=delivery,
        )
        needs, follow_q = review.needs_follow_up, review.follow_up_question

    update: dict = {"evaluations": [ev]}
    if needs and can_follow_up and follow_q.strip():
        update["pending_follow_up"] = follow_q.strip()
        update["follow_up_count"] = state.get("follow_up_count", 0) + 1
    return update


def route_after_evaluate(state: InterviewState) -> str:
    if state.get("pending_follow_up"):
        return "ask_question"
    if state["config"].feedback_mode == "practice":
        return "quick_tip"
    return "next_question"


# --------------------------------------------------------------------------------------
# tip / next
# --------------------------------------------------------------------------------------


def quick_tip(state: InterviewState) -> dict:
    last = state["evaluations"][-1]
    if not last.quick_tip:
        return {}
    return {"transcript": [Turn(role="interviewer", kind="tip", text=last.quick_tip, question_id=last.question_id)]}


def next_question(state: InterviewState) -> dict:
    return {"current_index": state["current_index"] + 1, "follow_up_count": 0, "pending_follow_up": None}


def route_after_next(state: InterviewState) -> str:
    return "ask_question" if state["current_index"] < len(state["plan"]) else "final_feedback"
