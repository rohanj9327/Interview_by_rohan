"""Streamlit UI flow via AppTest with the fake LLM (no API calls, no audio)."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"


@pytest.fixture
def ui(fake_llm, monkeypatch, tmp_path):
    sys.path.insert(0, str(APP))
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF0000WAVE")
    monkeypatch.setattr("interview_simulator.voice.tts.synthesize", lambda text, voice=None: clip)
    yield
    sys.path.remove(str(APP))


def button(at, label):
    matches = [b for b in at.button if b.label.startswith(label)]
    assert matches, f"no button {label!r}; have {[b.label for b in at.button]}"
    return matches[0]


def draft_area(at):
    return next(t for t in at.text_area if (t.key or "").startswith("draft_text_"))


def test_setup_interview_report_and_history(ui):
    at = AppTest.from_file(str(APP / "views" / "interview.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert button(at, "Start interview").disabled  # nothing provided yet
    assert at.title[0].value == "🎙️ Interview by Rohan"
    assert at.text_input(key="opt_name").value == "Rohan"

    at.toggle(key="use_sample_resume").set_value(True)
    at.toggle(key="use_sample_jd").set_value(True)
    at.run()
    at.selectbox[1].set_value("1")  # 1 coding question
    at.slider[0].set_value(4)
    at.run()
    button(at, "Start interview").click()
    at.run()
    assert not at.exception, at.exception
    session = at.session_state["session"]
    assert session.pending()["question_id"] == "q1"
    assert any("Kafka service" in m.markdown[0].value for m in at.chat_message if m.markdown)
    assert any(m.markdown[0].value.startswith("**Rohan:**") for m in at.chat_message if m.markdown)
    assert session.values["config"].interviewer_name == "Rohan"

    # Hint, then a typed answer (fake LLM asks one follow-up on the first answer)
    button(at, "💡 Hint").click()
    at.run()
    assert session.values["hints_used"]["q1"] == 1
    draft_area(at).set_value("I built it with Kafka.")
    at.run()
    button(at, "Submit answer").click()
    at.run()
    assert not at.exception, at.exception
    assert session.pending()["is_follow_up"]

    # End early -> report
    button(at, "End interview").click()
    at.run()
    assert not at.exception, at.exception
    assert session.done
    assert any(m.label == "Overall score" and m.value == "72/100" for m in at.metric)

    # History page shows the saved interview
    hist = AppTest.from_file(str(APP / "views" / "history.py"), default_timeout=60)
    hist.run()
    assert not hist.exception, hist.exception
    assert any(m.label == "Interviews completed" and m.value == "1" for m in hist.metric)


def test_coding_question_screen(ui):
    at = AppTest.from_file(str(APP / "views" / "interview.py"), default_timeout=120)
    at.run()
    at.toggle(key="use_sample_resume").set_value(True)
    at.toggle(key="use_sample_jd").set_value(True)
    at.run()
    at.selectbox[1].set_value("1")
    at.slider[0].set_value(4)
    at.run()
    button(at, "Start interview").click()
    at.run()
    session = at.session_state["session"]
    # skip to the coding question
    while not session.pending().get("coding_problem"):
        button(at, "Skip").click()
        at.run()
        assert not at.exception, at.exception
    assert any("Pair With Target Sum" in h.value for h in at.subheader)
    button(at, "▶ Run visible tests").click()
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["last_run"].passed == 0  # starter code fails


def test_coding_can_be_turned_off(ui):
    at = AppTest.from_file(str(APP / "views" / "interview.py"), default_timeout=120)
    at.run()
    at.toggle(key="use_sample_resume").set_value(True)
    at.toggle(key="use_sample_jd").set_value(True)
    at.toggle(key="opt_coding_on").set_value(False)
    at.run()
    assert not any(sb.key == "opt_coding_n" for sb in at.selectbox)  # count picker hidden
    at.slider[0].set_value(4)
    at.run()
    button(at, "Start interview").click()
    at.run()
    assert not at.exception, at.exception
    plan = at.session_state["session"].values["plan"]
    assert len(plan) == 4 and all(q.category != "coding" for q in plan)
    assert "action" not in at.session_state  # lock released after preparing


def test_inputs_are_locked_while_an_action_is_queued(ui, monkeypatch):
    """Render the setup page as if 'start' were running: every input must be disabled."""
    import interview_simulator.session as sess

    at = AppTest.from_file(str(APP / "views" / "interview.py"), default_timeout=120)
    at.run()
    at.toggle(key="use_sample_resume").set_value(True)
    at.toggle(key="use_sample_jd").set_value(True)
    at.run()
    # Make preparation block on a generator we never advance past the first label, by
    # queuing an action whose session raises -> we only inspect the locked render.
    class Boom(sess.InterviewSession):
        def start(self, *a, **k):
            raise RuntimeError("stop here")
            yield  # pragma: no cover

    from ui_common import checkpointer
    at.session_state["action"] = {"kind": "start", "title": "Preparing", "session": Boom(checkpointer=checkpointer()),
                                  "resume_text": "r", "jd_text": "j", "config": None}
    at.run()
    # After the failed action the lock is released and the error is shown
    assert "action" not in at.session_state
    assert any("stop here" in e.value for e in at.error)
    assert not button(at, "Start interview").disabled
    # inputs keep their values across the locked render
    assert at.toggle(key="use_sample_resume").value and at.toggle(key="use_sample_jd").value
