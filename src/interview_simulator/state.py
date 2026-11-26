"""LangGraph state for one interview session."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from interview_simulator.models import (
    AnswerEvaluation,
    FinalReport,
    GapReport,
    InterviewConfig,
    JobProfile,
    PlannedQuestion,
    QuestionDraft,
    ResearchedQuestion,
    ResumeProfile,
    Turn,
)


class InterviewState(TypedDict, total=False):
    # Inputs
    resume_text: str
    jd_text: str
    config: InterviewConfig

    # Parsed / derived
    resume: ResumeProfile
    job: JobProfile
    gap_report: GapReport
    researched_questions: list[ResearchedQuestion]

    # Plan
    intro: str
    plan_drafts: list[QuestionDraft]  # ordered plan before coding problems are generated
    plan: list[PlannedQuestion]

    # Interview loop
    current_index: int
    follow_up_count: int
    pending_follow_up: str | None  # follow-up question to ask next, if any
    current_prompt: str  # the question/follow-up currently awaiting an answer
    hints_used: dict[str, int]  # question_id -> hints given
    last_answer: dict[str, Any] | None  # resume payload of the latest answer
    next_action: str  # set by await_answer: "evaluate" | "hint" | "end"
    ended_early: bool
    transcript: Annotated[list[Turn], operator.add]
    evaluations: Annotated[list[AnswerEvaluation], operator.add]

    # Output
    final_report: FinalReport | None
    session_saved: bool
