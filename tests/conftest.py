import pytest
from langchain_core.runnables import RunnableLambda

from interview_simulator import llm
from interview_simulator.config import get_settings
from interview_simulator.models import (
    AnswerReview,
    AnswerScores,
    CodingReview,
    CodingScores,
    FinalReport,
    GapReport,
    Hint,
    Improvement,
    InterviewPlanDraft,
    JobProfile,
    QuestionDraft,
    QuestionFeedback,
    ResearchedQuestions,
    ResumeProfile,
    Strength,
)
from interview_simulator.prompts.question_bank import TWO_SUM


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Point runtime data (history.db, caches) at a temp dir for every test."""
    settings = get_settings()
    monkeypatch.setattr(settings, "data_dir", tmp_path / "runtime")
    yield


class FakeLLM:
    def __init__(self):
        self.answer_reviews = 0
        self.calls: list[str] = []

    def respond(self, schema):
        self.calls.append(schema.__name__)
        if schema is ResumeProfile:
            return ResumeProfile(name="Alex", summary="Backend engineer", years_experience=4, skills=["Java", "Kafka"])
        if schema is JobProfile:
            return JobProfile(title="Backend Engineer", seniority="senior", tech_stack=["Java", "Kafka"])
        if schema is GapReport:
            return GapReport(matched_skills=["Java"], missing_skills=["payments"], areas_to_probe=["ledger design"])
        if schema is ResearchedQuestions:
            return ResearchedQuestions(questions=[])
        if schema is InterviewPlanDraft:
            return InterviewPlanDraft(intro="Hi, I'm your interviewer.", questions=[
                QuestionDraft(text="Tell me about your Kafka service.", category="resume", rubric_hints=["ownership"]),
                QuestionDraft(text="How does a HashMap work?", category="fundamentals"),
                QuestionDraft(text="Tell me about a conflict.", category="behavioral"),
                QuestionDraft(text="Extra question that should be trimmed", category="behavioral"),
            ])
        if schema.__name__ == "CodingProblem":
            return TWO_SUM  # both coding slots get the same problem -> second must fall back
        if schema is AnswerReview:
            self.answer_reviews += 1
            return AnswerReview(
                scores=AnswerScores(relevance=4, depth=3, structure=4, technical_accuracy=4, communication=4, role_alignment=3),
                strengths=["clear"], weaknesses=["no metrics"],
                needs_follow_up=self.answer_reviews == 1,  # only the very first answer gets a follow-up
                follow_up_question="What was the measurable impact?" if self.answer_reviews == 1 else "",
                quick_tip="Good structure; add numbers.",
            )
        if schema is CodingReview:
            return CodingReview(scores=CodingScores(efficiency=5, code_quality=4, problem_solving=4, communication=4),
                                complexity_assessment="O(n) time", quick_tip="Nice use of a hash map.")
        if schema is Hint:
            return Hint(hint="Think about what you could store while scanning.")
        if schema is FinalReport:
            return FinalReport(
                overall_score=72, hire_signal="lean_yes", summary="Solid.",
                top_strengths=[Strength(title="Clear", evidence="Kafka answer")],
                improvement_areas=[Improvement(area="Metrics", why="none given", how_to_improve="quantify",
                                               example_better_answer="Cut latency 40%")],
                per_question=[QuestionFeedback(question_id="q1_follow_up_1", score=1, went_well="x", to_improve="y"),
                              QuestionFeedback(question_id="q1", score=7, went_well="x", to_improve="y")],
            )
        raise AssertionError(f"unexpected schema {schema}")


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "structured", lambda role, schema, provider=None: RunnableLambda(lambda _: fake.respond(schema)))
    monkeypatch.setattr(llm, "model_name", lambda role, provider=None: "fake-model")
    monkeypatch.setattr("interview_simulator.tools.web_search.search", lambda q, max_results=5: [])
    return fake


