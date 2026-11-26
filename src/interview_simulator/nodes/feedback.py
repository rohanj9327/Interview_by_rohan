"""Final report generation, Markdown rendering, and saving to history."""

from __future__ import annotations

import json
from statistics import mean

from langchain_core.runnables import RunnableConfig

from interview_simulator import history, llm
from interview_simulator.config import get_settings
from interview_simulator.models import AnswerEvaluation, FinalReport, Turn
from interview_simulator.nodes._util import ask_structured, dump
from interview_simulator.prompts import templates as P
from interview_simulator.state import InterviewState

CLOSING = ("That's the end of our interview. Thank you for your time. "
           "Your detailed feedback report is ready.")

SIGNAL_LABEL = {
    "strong_no": "Strong no hire", "no": "No hire", "lean_no": "Lean no hire",
    "lean_yes": "Lean hire", "yes": "Hire", "strong_yes": "Strong hire",
}


def delivery_stats(evals: list[AnswerEvaluation]) -> dict:
    voiced = [e.delivery for e in evals if e.delivery]
    if not voiced:
        return {}
    fillers: dict[str, int] = {}
    for d in voiced:
        for w, n in d.filler_words.items():
            fillers[w] = fillers.get(w, 0) + n
    return {
        "voice_answers": len(voiced),
        "avg_wpm": round(mean(d.wpm for d in voiced), 1),
        "wpm_per_answer": [d.wpm for d in voiced],
        "total_fillers": sum(d.filler_count for d in voiced),
        "avg_fillers_per_min": round(mean(d.fillers_per_min for d in voiced), 1),
        "top_fillers": dict(sorted(fillers.items(), key=lambda kv: -kv[1])[:5]),
        "long_pauses": sum(d.long_pauses for d in voiced),
        "avg_pace_score": round(mean(d.pace_score for d in voiced), 1),
        "avg_fluency_score": round(mean(d.fluency_score for d in voiced), 1),
    }


def _empty_report() -> FinalReport:
    return FinalReport(
        overall_score=0, hire_signal="no", summary="The interview ended before any questions were answered.",
        top_strengths=[], improvement_areas=[], per_question=[],
        next_steps=["Start a new interview and answer at least a few questions to get feedback."],
    )


def final_feedback(state: InterviewState) -> dict:
    evals = state.get("evaluations", [])
    closing = Turn(role="interviewer", kind="closing", text=CLOSING)
    if not evals:
        return {"final_report": _empty_report(), "transcript": [closing]}
    job = state["job"]
    resume = state["resume"]
    plan = state["plan"]
    answered = {e.question_id for e in evals}
    plan_text = "\n".join(
        f"- {q.id} [{q.category}] {q.coding_problem.title + ' (coding)' if q.coding_problem else q.text}"
        + ("" if q.id in answered else "  (NOT ANSWERED)")
        for q in plan
    )
    evals_text = "\n".join(
        json.dumps(e.model_dump(exclude={"delivery"}, exclude_none=True, exclude_defaults=True)) for e in evals
    )
    prompt = P.FINAL_REPORT.format(
        job_title=job.title, seniority=job.seniority, job=dump(job),
        resume=f"{resume.name}: {resume.summary} ({resume.years_experience} yrs; skills: {', '.join(resume.skills[:25])})",
        gaps=dump(state["gap_report"]), plan=plan_text, evaluations=evals_text,
        delivery=json.dumps(delivery_stats(evals)) or "{}",
        hints=json.dumps(state.get("hints_used", {})) or "none",
        ended_early="NOTE: the candidate ended the interview early; judge only answered questions."
        if state.get("ended_early") else "",
    )
    report = ask_structured("feedback", FinalReport, prompt)
    # Keep one entry per real question id (models sometimes invent ids for follow-ups)
    valid, seen, cleaned = {q.id for q in plan}, set(), []
    for pq in report.per_question:
        if pq.question_id in valid and pq.question_id not in seen:
            seen.add(pq.question_id)
            cleaned.append(pq)
    report.per_question = sorted(cleaned, key=lambda pq: int(pq.question_id[1:]) if pq.question_id[1:].isdigit() else 0)
    return {"final_report": report, "transcript": [closing]}


def report_to_markdown(state: InterviewState) -> str:
    r: FinalReport = state["final_report"]
    job = state.get("job")
    plan = {q.id: q for q in state.get("plan", [])}
    evals = state.get("evaluations", [])
    lines = [
        f"# Interview Feedback — {job.title if job else 'Software Engineer'}"
        + (f" at {job.company}" if job and job.company else ""),
        "",
        f"**Overall score:** {r.overall_score}/100  ·  **Signal:** {SIGNAL_LABEL.get(r.hire_signal, r.hire_signal)}",
        "",
        r.summary,
        "",
        "## Strengths",
    ]
    lines += [f"- **{s.title}** — {s.evidence}" for s in r.top_strengths] or ["- (none recorded)"]
    lines += ["", "## What to improve"]
    for i, imp in enumerate(r.improvement_areas, 1):
        lines += [
            f"### {i}. {imp.area}",
            f"**Why:** {imp.why}",
            "",
            f"**How to improve:** {imp.how_to_improve}",
            "",
            f"**Example of a stronger answer:**\n> {imp.example_better_answer}",
            "",
        ]
    if r.per_question:
        lines += ["## Question-by-question", "", "| # | Question | Score | Went well | To improve |", "|---|---|---|---|---|"]
        for pq in r.per_question:
            q = plan.get(pq.question_id)
            title = (q.coding_problem.title + " (coding)") if q and q.coding_problem else (q.text if q else pq.question_id)
            title = title.replace("|", "/")[:90]
            lines.append(f"| {pq.question_id} | {title} | {pq.score}/10 | {pq.went_well.replace('|', '/')} | {pq.to_improve.replace('|', '/')} |")
        lines.append("")
    coding = [e for e in evals if e.tests_total]
    if r.coding_summary or coding:
        lines += ["## Coding"]
        for e in coding:
            lines.append(f"- **{e.question_text}** ({e.language}): {e.tests_passed}/{e.tests_total} tests passed. {e.complexity_assessment}")
        if r.coding_summary:
            lines += ["", r.coding_summary]
        lines.append("")
    stats = delivery_stats(evals)
    if r.delivery_summary or stats:
        lines += ["## Voice delivery"]
        if stats:
            fillers = ", ".join(f"\"{w}\" ×{n}" for w, n in stats["top_fillers"].items()) or "none"
            lines += [
                f"- Average pace: **{stats['avg_wpm']} wpm** (ideal 130–160)",
                f"- Filler words: **{stats['total_fillers']}** total, {stats['avg_fillers_per_min']}/min (top: {fillers})",
                f"- Long pauses (>2s): **{stats['long_pauses']}**",
            ]
        if r.delivery_summary:
            lines += ["", r.delivery_summary]
        lines.append("")
    if r.hints_summary:
        lines += ["## Hints", r.hints_summary, ""]
    if r.next_steps:
        lines += ["## Next steps"] + [f"{i}. {s}" for i, s in enumerate(r.next_steps, 1)] + [""]
    if r.recommended_resources:
        lines += ["## Recommended resources"] + [f"- {s}" for s in r.recommended_resources] + [""]
    return "\n".join(lines)


def save_session(state: InterviewState, config: RunnableConfig) -> dict:
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    r = state["final_report"]
    job = state.get("job")
    s = get_settings()
    history.save_session(
        thread_id,
        job_title=job.title if job else None,
        company=job.company if job else None,
        seniority=job.seniority if job else None,
        provider=s.llm_provider,
        model=llm.model_name("feedback"),
        overall_score=r.overall_score,
        hire_signal=r.hire_signal,
        ended_early=int(bool(state.get("ended_early"))),
        report_json=r.model_dump_json(),
        report_markdown=report_to_markdown(state),
        transcript_json=json.dumps([t.model_dump() for t in state.get("transcript", [])]),
        evaluations_json=json.dumps([e.model_dump() for e in state.get("evaluations", [])]),
    )
    return {"session_saved": True}
