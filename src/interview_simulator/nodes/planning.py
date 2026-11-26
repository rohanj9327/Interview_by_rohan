"""Build the interview plan: spoken questions from the LLM + coding problems."""

from __future__ import annotations

from interview_simulator.models import (
    InterviewConfig,
    InterviewPlanDraft,
    JobProfile,
    QuestionDraft,
)
from interview_simulator.nodes._util import ask_structured, dump, log
from interview_simulator.prompts import templates as P
from interview_simulator.prompts.question_bank import COMMON_QUESTIONS
from interview_simulator.state import InterviewState

CATEGORY_ORDER = ["resume", "fundamentals", "coding", "system_design", "behavioral", "company"]


def num_coding(config: InterviewConfig, job: JobProfile) -> int:
    if config.num_coding_questions is not None:
        return max(0, min(config.num_coding_questions, config.num_questions - 1))
    if config.num_questions < 5:
        return 1
    return 2 if job.seniority in ("mid", "senior", "staff", "principal") else 1


def category_counts(total: int, coding: int, has_company: bool) -> dict[str, int]:
    """Split the non-coding questions: ~resume 25%, fundamentals 20%, design 20%, behavioral 25%, company."""
    remaining = max(total - coding, 0)
    weights = {"resume": 0.27, "fundamentals": 0.22, "system_design": 0.2, "behavioral": 0.25,
               "company": 0.06 if has_company else 0.0}
    norm = sum(weights.values())
    counts = {k: int(remaining * w / norm) for k, w in weights.items()}
    priority = ["resume", "behavioral", "fundamentals", "system_design", "company"]
    i = 0
    while sum(counts.values()) < remaining:
        key = priority[i % len(priority)]
        if weights[key] > 0:
            counts[key] += 1
        i += 1
    return {k: v for k, v in counts.items() if v > 0}


def _difficulty(config: InterviewConfig, job: JobProfile) -> str:
    if config.difficulty != "auto":
        return config.difficulty
    return {"intern": "easy", "junior": "easy", "mid": "medium"}.get(job.seniority, "medium-hard")


def _fit_to_counts(drafts: list[QuestionDraft], counts: dict[str, int]) -> list[QuestionDraft]:
    """Trim/pad the LLM output so each category has exactly the requested count."""
    out: list[QuestionDraft] = []
    for cat, n in counts.items():
        picked = [d for d in drafts if d.category == cat][:n]
        if len(picked) < n:
            bank = [q for q in COMMON_QUESTIONS if q.category == cat and q.question not in {p.text for p in picked}]
            picked += [QuestionDraft(text=q.question, category=cat, source="bank") for q in bank[: n - len(picked)]]
        out += picked
    return out


def plan_interview(state: InterviewState) -> dict:
    config, job = state["config"], state["job"]
    coding = num_coding(config, job)
    counts = category_counts(config.num_questions, coding, bool(job.company))
    prompt = P.PLAN_INTERVIEW.format(
        company=job.company or "a tech company", seniority=job.seniority, job_title=job.title,
        counts="\n".join(f"- {k}: {v}" for k, v in counts.items()),
        total=config.num_questions, interviewer_name=config.interviewer_name, difficulty=_difficulty(config, job), spoken_style=P.SPOKEN_STYLE,
        resume=dump(state["resume"]), job=dump(job), gaps=dump(state["gap_report"]),
        research=dump(state.get("researched_questions", [])),
    )
    draft: InterviewPlanDraft = ask_structured("interviewer", InterviewPlanDraft, prompt)
    questions = _fit_to_counts([q for q in draft.questions if q.category != "coding"], counts)
    log.info("Planned %d spoken questions + %d coding", len(questions), coding)
    # Coding slots are filled by generate_coding_problems; store their count via placeholders.
    placeholders = [
        QuestionDraft(text="", category="coding", source="generated", target_skill="coding")
        for _ in range(coding)
    ]
    ordered = sorted(questions + placeholders, key=lambda q: CATEGORY_ORDER.index(q.category))
    return {"intro": draft.intro, "plan_drafts": ordered}
