"""End-to-end graph test with a fake LLM (no API calls).

Covers: parallel parsing, research fallback, planning, coding problem dedupe/fallback,
hints, follow-ups, practice tips, live coding evaluation with real test runs,
ending early, final report, and saving to history.
"""


from interview_simulator import history
from interview_simulator.graph import memory_checkpointer
from interview_simulator.models import (
    InterviewConfig,
)
from interview_simulator.prompts.question_bank import TWO_SUM
from interview_simulator.session import InterviewSession


def drain(events):
    return list(events)


def test_full_interview_flow(fake_llm):
    session = InterviewSession(checkpointer=memory_checkpointer())
    config = InterviewConfig(num_questions=5, num_coding_questions=2, feedback_mode="practice")
    labels = drain(session.start("resume text", "jd text", config))
    assert "Reading your resume" in labels and "Planning your interview" in labels

    plan = session.values["plan"]
    assert [q.category for q in plan] == ["resume", "fundamentals", "coding", "coding", "behavioral"]
    coding_titles = [q.coding_problem.title for q in plan if q.category == "coding"]
    assert coding_titles == ["Pair With Target Sum", "Merge Overlapping Intervals"]  # dedupe -> fallback

    # q1: intro + question, then a hint
    p = session.pending()
    assert p["question_id"] == "q1" and not p["is_follow_up"]
    kinds = [t.kind for t in session.new_turns()]
    assert kinds == ["intro", "question"]
    drain(session.hint("I'm not sure"))
    assert [t.kind for t in session.new_turns()] == ["hint"]
    assert session.pending()["hints_used"] == 1

    # q1 answer -> follow-up
    drain(session.answer("I built a Kafka consumer service."))
    p = session.pending()
    assert p["question_id"] == "q1" and p["is_follow_up"]
    assert [t.kind for t in session.new_turns()] == ["answer", "follow_up"]

    # follow-up answer -> practice tip -> q2
    drain(session.answer("Latency dropped by 40%."))
    assert [t.kind for t in session.new_turns()] == ["answer", "tip", "question"]
    assert session.pending()["question_id"] == "q2"

    # q2 -> q3 (coding, problem is sent to the UI)
    drain(session.answer("Buckets and hashing."))
    p = session.pending()
    assert p["question_id"] == "q3" and p["coding_problem"]["title"] == "Pair With Target Sum"
    assert p["time_limit_sec"] == config.coding_time_limit_sec

    # coding answer with correct Java code -> real test run
    drain(session.answer("Hash map, O(n).", code=TWO_SUM.reference_java, language="java"))
    ev = session.values["evaluations"][-1]
    assert ev.category == "coding" and ev.tests_passed == ev.tests_total == 5
    assert ev.scores["correctness"] == 5 and ev.language == "java"
    assert session.pending()["question_id"] == "q4"

    # end early -> report + history
    drain(session.end())
    assert session.done
    values = session.values
    assert values["ended_early"] and values["final_report"].overall_score == 72
    assert [pq.question_id for pq in values["final_report"].per_question] == ["q1"]  # invented id dropped
    assert values["transcript"][-1].kind == "closing"
    assert len(values["evaluations"]) == 4

    saved = history.get_session(session.thread_id)
    assert saved and saved["overall_score"] == 72 and saved["ended_early"] == 1
    assert "Interview Feedback" in saved["report_markdown"]
    assert "5/5 tests passed" in saved["report_markdown"]


def test_realistic_mode_has_no_tips(fake_llm):
    fake_llm.answer_reviews = 1  # disable follow-ups
    session = InterviewSession(checkpointer=memory_checkpointer())
    drain(session.start("r", "j", InterviewConfig(num_questions=2, num_coding_questions=0, feedback_mode="realistic")))
    session.new_turns()
    drain(session.answer("answer one"))
    assert [t.kind for t in session.new_turns()] == ["answer", "question"]
    drain(session.answer("answer two"))
    assert session.done
    assert not any(t.kind == "tip" for t in session.values["transcript"])


def test_end_immediately_gives_empty_report(fake_llm):
    session = InterviewSession(checkpointer=memory_checkpointer())
    drain(session.start("r", "j", InterviewConfig(num_questions=2, num_coding_questions=0)))
    drain(session.end())
    assert session.done and session.values["final_report"].overall_score == 0
    assert "FinalReport" not in fake_llm.calls


def test_delete_interview_removes_history_and_checkpoints(fake_llm):
    from interview_simulator.config import get_settings
    from interview_simulator.session import delete_interview

    saver = memory_checkpointer()
    session = InterviewSession(checkpointer=saver)
    drain(session.start("r", "j", InterviewConfig(num_questions=2, num_coding_questions=0)))
    drain(session.answer("an answer"))
    drain(session.end())
    report = get_settings().data_path / "reports" / f"{session.thread_id}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("x")
    assert history.get_session(session.thread_id)

    delete_interview(session.thread_id, saver)
    assert history.get_session(session.thread_id) is None
    assert not report.exists()
    assert not InterviewSession(thread_id=session.thread_id, checkpointer=saver).started
