"""CLI coding flow with a stub session (no LLM)."""

import builtins

from interview_simulator.cli import CLI
from interview_simulator.prompts.question_bank import TWO_SUM


class StubSession:
    thread_id = "testcli"

    def __init__(self):
        self.values: dict = {}
        self.answers = []
        self.hints = []

    def answer(self, text, **kw):
        self.answers.append((text, kw))
        return iter(["Evaluating your answer"])

    def hint(self, text="", code=None):
        self.hints.append(code)
        return iter([])

    def new_turns(self):
        return []


def test_coding_flow_run_switch_language_submit(monkeypatch, capsys):
    session = StubSession()
    cli = CLI(session, speak=False)
    pending = {"question_id": "q3", "coding_problem": TWO_SUM.model_dump()}

    inputs = iter(["/run", "/lang java", "WRITE_JAVA", "/run", "/hint", "/submit", "Hash map, O(n).", ""])

    def fake_input(prompt=""):
        value = next(inputs)
        if value == "WRITE_JAVA":  # simulate the candidate editing the Java file
            cli.code_file("q3").write_text(TWO_SUM.reference_java)
            return "/run"
        return value

    monkeypatch.setattr(builtins, "input", fake_input)
    cli.handle_coding(pending)
    out = capsys.readouterr().out
    assert "0/2 visible tests passed" in out  # python starter code
    assert "2/2 visible tests passed" in out  # java reference
    assert session.hints and "class Solution" in session.hints[0]
    text, kw = session.answers[0]
    assert text == "Hash map, O(n)." and kw["language"] == "java" and "HashMap" in kw["code"]


def test_unknown_command_is_not_submitted(monkeypatch):
    session = StubSession()
    cli = CLI(session, speak=False)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "/run")
    cli.handle_answer({"index": 1, "total": 2, "time_limit_sec": 180})
    assert session.answers == []


def test_interviewer_turns_are_labelled_with_name(capsys):
    from interview_simulator.models import InterviewConfig, Turn

    session = StubSession()
    session.values = {"config": InterviewConfig()}
    session.new_turns = lambda: [Turn(role="interviewer", kind="question", text="Tell me about Kafka.")]
    CLI(session, speak=False).show_new_turns()
    assert "Rohan: Tell me about Kafka." in capsys.readouterr().out
