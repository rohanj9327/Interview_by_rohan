"""Pydantic schemas shared across the app.

Schemas used for LLM structured output are kept fairly flat (no deep unions, simple
types, `Literal` enums) so they work reliably on both Claude and Gemini.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# Common enums
# --------------------------------------------------------------------------------------

Category = Literal["resume", "fundamentals", "coding", "system_design", "behavioral", "company"]
Seniority = Literal["intern", "junior", "mid", "senior", "staff", "principal"]
Language = Literal["python", "java"]
ParamType = Literal[
    "int", "long", "double", "bool", "string",
    "int[]", "double[]", "bool[]", "string[]",
    "int[][]", "string[][]",
]
HireSignal = Literal["strong_no", "no", "lean_no", "lean_yes", "yes", "strong_yes"]


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


class InterviewConfig(BaseModel):
    num_questions: int = 10
    num_coding_questions: int | None = None  # None = auto from seniority (1 or 2)
    answer_time_limit_sec: int = 180
    coding_time_limit_sec: int = 1200
    feedback_mode: Literal["practice", "realistic"] = "practice"
    max_follow_ups: int = 2
    difficulty: Literal["auto", "easy", "medium", "hard"] = "auto"
    company: str | None = None
    interviewer_name: str = "Rohan"


# --------------------------------------------------------------------------------------
# Resume / JD parsing (LLM output)
# --------------------------------------------------------------------------------------


class Experience(BaseModel):
    title: str
    company: str
    duration: str = ""
    highlights: list[str] = Field(default_factory=list)


class Project(BaseModel):
    name: str
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    name: str = ""
    summary: str = ""
    years_experience: float = Field(0, description="Total years of professional experience")
    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)


class JobProfile(BaseModel):
    title: str
    company: str = Field("", description="Company name, empty string if unknown")
    seniority: Seniority = "mid"
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    interview_process_notes: str = ""


class GapReport(BaseModel):
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list, description="Candidate strengths for this role")
    areas_to_probe: list[str] = Field(
        default_factory=list, description="Topics the interviewer should dig into (gaps, vague claims)"
    )
    talking_points: list[str] = Field(
        default_factory=list, description="Resume items worth a deep-dive question"
    )


# --------------------------------------------------------------------------------------
# Web research
# --------------------------------------------------------------------------------------


class ResearchedQuestion(BaseModel):
    question: str
    category: Category
    source_url: str = ""


class ResearchedQuestions(BaseModel):
    questions: list[ResearchedQuestion] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Coding problems
# --------------------------------------------------------------------------------------


class Param(BaseModel):
    name: str
    type: ParamType


class Signature(BaseModel):
    function_name: str = Field(description="camelCase method name, e.g. twoSum")
    params: list[Param]
    return_type: ParamType


class TestCase(BaseModel):
    args_json: str = Field(
        description='JSON object mapping parameter name to value, e.g. {"nums": [1,2], "target": 3}'
    )
    expected_json: str = Field(description="JSON value of the expected return, e.g. [0,1]")


class CodingProblem(BaseModel):
    title: str
    statement: str = Field(description="Full problem statement in plain English")
    examples: list[str] = Field(default_factory=list, description="Examples with input/output/explanation")
    constraints: list[str] = Field(default_factory=list)
    signature: Signature
    visible_tests: list[TestCase]
    hidden_tests: list[TestCase]
    reference_python: str = Field(description="Correct Python 3 solution: class Solution with the method")
    reference_java: str = Field(description="Correct Java solution: class Solution with the method")
    expected_time_complexity: str = ""
    expected_space_complexity: str = ""


class TestResult(BaseModel):
    index: int
    hidden: bool = False
    passed: bool
    args_json: str
    expected_json: str
    output_json: str | None = None
    error: str | None = None
    time_ms: float | None = None


class RunResult(BaseModel):
    language: Language
    compiled: bool = True
    compile_error: str | None = None
    results: list[TestResult] = Field(default_factory=list)
    timed_out: bool = False
    stderr: str = ""

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def all_passed(self) -> bool:
        return self.compiled and self.total > 0 and self.passed == self.total


# --------------------------------------------------------------------------------------
# Interview plan
# --------------------------------------------------------------------------------------


class QuestionDraft(BaseModel):
    text: str = Field(description="The question exactly as the interviewer will say it (spoken style)")
    category: Category
    target_skill: str = ""
    source: str = Field("", description="resume | jd | gap | web:<url> | bank")
    rubric_hints: list[str] = Field(default_factory=list, description="What a strong answer covers")


class InterviewPlanDraft(BaseModel):
    intro: str = Field(description="Short spoken greeting + what to expect (2-3 sentences)")
    questions: list[QuestionDraft]


class PlannedQuestion(QuestionDraft):
    id: str
    coding_problem: CodingProblem | None = None


# --------------------------------------------------------------------------------------
# Interview transcript
# --------------------------------------------------------------------------------------


class Turn(BaseModel):
    role: Literal["interviewer", "candidate"]
    kind: Literal["intro", "question", "follow_up", "answer", "hint", "tip", "closing"]
    text: str
    question_id: str | None = None
    mode: Literal["text", "voice"] = "text"
    code: str | None = None
    language: Language | None = None


# --------------------------------------------------------------------------------------
# Voice delivery (computed deterministically, not by the LLM)
# --------------------------------------------------------------------------------------


class DeliveryMetrics(BaseModel):
    duration_sec: float
    speaking_sec: float
    word_count: int
    wpm: float
    filler_count: int
    filler_words: dict[str, int] = Field(default_factory=dict)
    fillers_per_min: float
    long_pauses: int
    longest_pause_sec: float
    pace_score: int  # 1-5
    fluency_score: int  # 1-5


# --------------------------------------------------------------------------------------
# Evaluation (LLM output schemas + stored record)
# --------------------------------------------------------------------------------------


class AnswerScores(BaseModel):
    relevance: int = Field(ge=1, le=5)
    depth: int = Field(ge=1, le=5)
    structure: int = Field(ge=1, le=5)
    technical_accuracy: int = Field(ge=1, le=5)
    communication: int = Field(ge=1, le=5)
    role_alignment: int = Field(ge=1, le=5)


class AnswerReview(BaseModel):
    """LLM output for a spoken/typed (non-coding) answer."""

    scores: AnswerScores
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    needs_follow_up: bool = False
    follow_up_question: str = Field("", description="Spoken follow-up question, empty if none")
    quick_tip: str = Field(description="1-2 sentences: one thing done well, one thing to improve")
    confidence_score: int = Field(3, ge=1, le=5, description="Confidence/clarity read from wording")
    delivery_notes: list[str] = Field(default_factory=list)


class CodingScores(BaseModel):
    efficiency: int = Field(ge=1, le=5)
    code_quality: int = Field(ge=1, le=5)
    problem_solving: int = Field(ge=1, le=5)
    communication: int = Field(ge=1, le=5)


class CodingReview(BaseModel):
    """LLM output for a coding answer (correctness comes from the test runner)."""

    scores: CodingScores
    complexity_assessment: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    needs_follow_up: bool = False
    follow_up_question: str = ""
    quick_tip: str


class AnswerEvaluation(BaseModel):
    """Stored evaluation of one candidate answer."""

    question_id: str
    category: Category
    is_follow_up: bool = False
    question_text: str
    answer_text: str
    scores: dict[str, int]
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    quick_tip: str = ""
    hints_used: int = 0
    confidence_score: int | None = None
    delivery_notes: list[str] = Field(default_factory=list)
    delivery: DeliveryMetrics | None = None
    # coding only
    language: Language | None = None
    tests_passed: int | None = None
    tests_total: int | None = None
    complexity_assessment: str = ""


class Hint(BaseModel):
    hint: str = Field(description="Spoken hint, never the full solution")


# --------------------------------------------------------------------------------------
# Final report (LLM output)
# --------------------------------------------------------------------------------------


class Strength(BaseModel):
    title: str
    evidence: str = Field(description="Specific reference to what the candidate said or wrote")


class Improvement(BaseModel):
    area: str
    why: str = Field(description="Why this matters / what was weak, citing the answer")
    how_to_improve: str = Field(description="Concrete, actionable steps")
    example_better_answer: str = Field(description="A short example of a stronger answer")
    related_question_ids: list[str] = Field(default_factory=list)


class QuestionFeedback(BaseModel):
    question_id: str
    score: int = Field(ge=1, le=10)
    went_well: str
    to_improve: str


class FinalReport(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    hire_signal: HireSignal
    summary: str
    top_strengths: list[Strength]
    improvement_areas: list[Improvement]
    per_question: list[QuestionFeedback]
    coding_summary: str = ""
    delivery_summary: str = ""
    hints_summary: str = ""
    recommended_resources: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
