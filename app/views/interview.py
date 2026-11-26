"""Interview page: setup -> spoken interview (voice + live coding) -> feedback report.

Long-running steps (preparing the interview, evaluating an answer, hints, ending) run as
queued "actions": the click stores the action and reruns, the page is re-rendered with
every input disabled, and the action runs while a status box shows progress. This stops
double-clicks or other clicks from interrupting a step halfway (Streamlit cancels a running
script when the user interacts with the page).
"""

from __future__ import annotations

import hashlib
import time

import streamlit as st
import streamlit.components.v1 as components
from ui_common import (
    CATEGORY_LABEL,
    checkpointer,
    interviewer_name,
    render_report,
    render_transcript,
    theme_type,
)

from interview_simulator.config import PROJECT_ROOT, get_settings
from interview_simulator.models import CodingProblem, InterviewConfig
from interview_simulator.session import InterviewSession, delete_interview, list_paused_sessions
from interview_simulator.tools.code_runner import run_code, starter_code
from interview_simulator.tools.document_loader import load_bytes, load_document
from interview_simulator.tools.jd_fetcher import fetch_jd
from interview_simulator.voice.tts import VOICES

SAMPLES = PROJECT_ROOT / "data" / "samples"
ss = st.session_state


# ======================================================================================
# actions (long-running steps with a locked UI)
# ======================================================================================


def busy() -> bool:
    return "action" in ss


def queue_action(kind: str, title: str, **payload) -> None:
    ss.action = {"kind": kind, "title": title, **payload}
    ss.pop("last_error", None)
    st.rerun()


def _friendly_error(exc: Exception) -> str:
    msg = str(exc)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "The AI provider is rate-limiting requests (free-tier quota). Wait a minute, then press Retry."
    if "503" in msg or "UNAVAILABLE" in msg:
        return "The AI model is temporarily overloaded. Press Retry in a moment."
    return f"Error: {msg[:500]}"


def process_action() -> None:
    """Run the queued action (called where its status box should appear)."""
    action = ss.get("action")
    if not action:
        return
    kind = action["kind"]
    session: InterviewSession = action.get("session") or ss.get("session")

    # If an earlier run of this action was interrupted after the answer was consumed,
    # continue from the checkpoint instead of sending the answer again.
    if kind == "start":
        events = session.retry() if session.started else session.start(
            action["resume_text"], action["jd_text"], action["config"])
    elif kind != "retry" and session.pending() is None and session.needs_retry:
        events = session.retry()
    elif kind == "answer":
        events = session.answer(**action["args"])
    elif kind == "hint":
        events = session.hint(**action["args"])
    elif kind == "end":
        events = session.end()
    else:  # retry
        events = session.retry()

    with st.status(action["title"], expanded=True) as status:
        try:
            for label in events:
                status.write(f"{label}…")
            status.update(label="Done", state="complete", expanded=False)
            ok = True
        except Exception as exc:  # noqa: BLE001
            status.update(label="Something went wrong", state="error")
            ss.last_error = _friendly_error(exc)
            ok = False

    ss.pop("action", None)
    if kind == "start" and (ok or session.started):
        ss.session = session
        ss.spoken_upto = 0
    if ok and kind in ("start", "answer"):
        reset_answer_inputs()
    st.rerun()


def show_last_error() -> None:
    if ss.get("last_error"):
        st.error(ss.last_error)


# ======================================================================================
# helpers
# ======================================================================================


def reset_answer_inputs() -> None:
    ss.mic_nonce = ss.get("mic_nonce", 0) + 1
    for key in ("draft", "last_run", "mic_error"):
        ss.pop(key, None)


def voice_update(session: InterviewSession) -> None:
    """Synthesize newly added interviewer turns into one clip (played in the sidebar)."""
    transcript = session.values.get("transcript", [])
    start = ss.get("spoken_upto", 0)
    new = [t for t in transcript[start:] if t.role == "interviewer"]
    ss.spoken_upto = len(transcript)
    if not new or not ss.get("speak", True):
        return
    from interview_simulator.voice import tts

    text = " ".join(t.text for t in new)
    try:
        ss.clip = str(tts.synthesize(text, ss.get("voice")))
    except Exception as exc:  # noqa: BLE001
        st.toast(f"Could not generate speech: {exc}")


def countdown(seconds_left: int, total: int, key: str) -> None:
    """Client-side countdown timer (re-renders only when the question changes)."""
    components.html(
        f"""
<div id="t" style="font:600 15px system-ui,-apple-system,sans-serif;padding:6px 10px;border-radius:8px;
     display:inline-block;background:rgba(42,120,214,.10);color:#2a78d6"></div>
<script>
  const end = Date.now() + {max(seconds_left, 0)} * 1000, total = {total};
  const el = document.getElementById('t');
  function tick() {{
    const s = Math.max(0, Math.round((end - Date.now()) / 1000));
    const m = Math.floor(s / 60), r = String(s % 60).padStart(2, '0');
    if (s === 0) {{ el.textContent = "⏱ Time's up, please wrap up your answer";
                    el.style.background = 'rgba(227,73,72,.12)'; el.style.color = '#e34948'; return; }}
    el.textContent = '⏱ ' + m + ':' + r + ' left (limit ' + Math.round(total / 60) + ' min)';
    if (s <= 30) {{ el.style.background = 'rgba(237,161,0,.15)'; el.style.color = '#b07800'; }}
    setTimeout(tick, 1000);
  }}
  tick();
</script>
<!-- {key} -->""",
        height=44,
    )


def timer_for(pending: dict) -> None:
    key = hashlib.sha1(f"{pending['question_id']}|{pending['prompt']}".encode()).hexdigest()[:12]
    started = ss.setdefault("prompt_started", {}).setdefault(key, time.time())
    total = int(pending["time_limit_sec"])
    countdown(int(total - (time.time() - started)), total, key)


def transcribe_recording(audio_file, limit_sec: int, session: InterviewSession) -> None:
    """Transcribe a new mic recording into ss.draft (text + delivery metrics)."""
    data = audio_file.getvalue()
    digest = hashlib.sha1(data).hexdigest()
    if ss.get("draft", {}).get("digest") == digest or ss.get("mic_error", {}).get("digest") == digest:
        return
    from interview_simulator.voice.audio import TARGET_SR, decode_audio_bytes
    from interview_simulator.voice.delivery import compute_metrics
    from interview_simulator.voice.quality import NoSpeechError
    from interview_simulator.voice.stt import transcribe

    audio = decode_audio_bytes(data)
    trimmed = len(audio) > limit_sec * TARGET_SR
    if trimmed:
        audio = audio[: limit_sec * TARGET_SR]
    try:
        with st.spinner("Transcribing your answer…"):
            tr = transcribe(audio, vocabulary=session.vocabulary())
    except NoSpeechError as exc:
        ss.pop("draft", None)
        ss.mic_error = {"digest": digest, "message": str(exc)}
        return
    ss.pop("mic_error", None)
    ss.draft = {"digest": digest, "text": tr.text, "delivery": compute_metrics(tr),
                "trimmed": trimmed, "warnings": tr.warnings}
    ss[f"draft_text_{ss.get('mic_nonce', 0)}"] = tr.text


def answer_box(session: InterviewSession, pending: dict, label: str, code: str | None = None,
               language: str | None = None) -> None:
    """Mic recorder + editable transcript + submit / hint / skip buttons."""
    locked = busy()
    nonce = ss.get("mic_nonce", 0)
    limit = int(get_settings().answer_time_limit_sec if code is not None else pending["time_limit_sec"])
    audio = st.audio_input(label, key=f"mic_{nonce}", disabled=locked)
    if audio is not None and not locked:
        transcribe_recording(audio, limit, session)
    if ss.get("mic_error"):
        st.error(f"🎤 {ss.mic_error['message']}")
        st.caption("Tip: on macOS check System Settings → Privacy & Security → Microphone for your browser, "
                   "and the input device in System Settings → Sound.")
    draft = ss.get("draft")
    if draft and draft.get("trimmed"):
        st.warning(f"Your recording was longer than {limit // 60} min, so only the first {limit // 60} min were used.")
    for w in (draft or {}).get("warnings", []):
        st.warning(w)
    text = st.text_area(
        "Transcript (edit if speech recognition got something wrong) — or type your answer",
        key=f"draft_text_{nonce}", height=150, disabled=locked,
        placeholder="Record your answer above, or type it here…",
    )
    if draft:
        d = draft["delivery"]
        st.caption(f"🎤 {d.wpm:.0f} words/min · {d.filler_count} filler words · {d.long_pauses} long pauses")

    c1, c2, c3 = st.columns([2, 1, 1])
    submit_label = "Submit solution" if code is not None else "Submit answer"
    if locked and ss.action["kind"] == "answer":
        submit_label = "Evaluating…"
    if c1.button(submit_label, type="primary", width="stretch", key=f"submit_{nonce}", disabled=locked):
        if code is None and not text.strip():
            st.warning("Record or type an answer first.")
        else:
            voice = bool(draft) and draft["text"].strip() != "" and text.strip() != ""
            queue_action("answer", "Evaluating your answer…", args={
                "text": text.strip(), "mode": "voice" if voice else "text",
                "delivery": draft["delivery"] if voice else None, "code": code, "language": language,
            })
    hints_left = 3 - int(pending.get("hints_used", 0))
    if c2.button(f"💡 Hint ({hints_left} left)", width="stretch",
                 disabled=locked or hints_left <= 0, key=f"hint_{nonce}"):
        queue_action("hint", "Thinking of a hint…", args={"text": text, "code": code})
    if c3.button("Skip", width="stretch", key=f"skip_{nonce}", disabled=locked,
                 help="Move on without answering (counts as no answer)"):
        queue_action("answer", "Moving on…", args={"text": "(skipped)", "code": code, "language": language})
    show_last_error()
    process_action()


# ======================================================================================
# setup
# ======================================================================================


def available_providers() -> dict[str, str]:
    s = get_settings()
    out = {}
    if s.anthropic_api_key:
        out["anthropic"] = "Claude"
    if s.google_api_key and s.gemini_interviewer_model:
        out["google_genai"] = "Gemini"
    return out


def setup_view() -> None:
    locked = busy()
    st.title("🎙️ Interview by Rohan")
    st.caption("Your personal mock interviewer for software engineering roles. Rohan reads your resume and "
               "the job, asks tailored questions out loud, gives you live coding problems, and then tells you "
               "what was strong and what to improve.")

    paused = [] if locked else list_paused_sessions(checkpointer())
    if paused:
        with st.expander(f"Continue a paused interview ({len(paused)})"):
            for p in paused:
                c1, c2, c3 = st.columns([4, 1, 1])
                c1.write(f"**{p['job'] or 'Interview'}**" + (f" at {p['company']}" if p["company"] else "")
                         + f" · question {p['progress']} · `{p['id']}`")
                if c2.button("Continue", key=f"cont_{p['id']}"):
                    session = InterviewSession(thread_id=p["id"], checkpointer=checkpointer())
                    ss.session = session
                    ss.spoken_upto = max(0, len(session.values.get("transcript", [])) - 1)
                    reset_answer_inputs()
                    st.rerun()
                if ss.get("confirm_discard") == p["id"]:
                    if c3.button("Confirm", key=f"discard_ok_{p['id']}", type="primary",
                                 help="Permanently delete this paused interview"):
                        delete_interview(p["id"], checkpointer())
                        ss.pop("confirm_discard", None)
                        st.rerun()
                elif c3.button("🗑 Discard", key=f"discard_{p['id']}"):
                    ss.confirm_discard = p["id"]
                    st.rerun()

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("1. Your resume")
        use_sample = st.toggle("Use the sample resume", key="use_sample_resume", disabled=locked)
        resume_text = ""
        if use_sample:
            resume_text = load_document(SAMPLES / "resume.md")
        else:
            up = st.file_uploader("Upload resume", type=["pdf", "docx", "txt", "md"], key="resume_file",
                                  disabled=locked)
            if up is not None:
                try:
                    resume_text = load_bytes(up.name, up.getvalue())
                except ValueError as exc:
                    st.error(str(exc))
        if resume_text:
            with st.expander("Review the extracted text"):
                resume_text = st.text_area("Resume text", resume_text, height=260, disabled=locked,
                                           key=f"resume_text_{hash(resume_text)}")

    with right:
        st.subheader("2. The job")
        use_sample_jd = st.toggle("Use the sample job description", key="use_sample_jd", disabled=locked)
        jd_text = ""
        if use_sample_jd:
            jd_text = load_document(SAMPLES / "job_description.md")
            st.caption("Senior Backend Engineer, Payments Platform (FinFlow)")
        else:
            tab_paste, tab_url, tab_file = st.tabs(["Paste text", "From URL", "Upload file"])
            with tab_paste:
                pasted = st.text_area("Job description", height=180, key="jd_paste", disabled=locked,
                                      placeholder="Paste the full job description…")
            with tab_url:
                url = st.text_input("Job posting URL", key="jd_url", placeholder="https://…", disabled=locked)
                if url and st.button("Fetch", key="jd_fetch", disabled=locked):
                    try:
                        with st.spinner("Fetching job description…"):
                            ss.jd_fetched = fetch_jd(url)
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))
                if ss.get("jd_fetched"):
                    st.success(f"Fetched {len(ss.jd_fetched):,} characters.")
            with tab_file:
                jd_up = st.file_uploader("Upload job description", type=["pdf", "docx", "txt", "md"],
                                         key="jd_file", disabled=locked)
            jd_text = (pasted or "").strip() or ss.get("jd_fetched", "") or (
                load_bytes(jd_up.name, jd_up.getvalue()) if jd_up is not None else ""
            )
        company = st.text_input("Company name (optional, improves research)", key="company", disabled=locked)

    st.subheader("3. Interview settings")
    s = get_settings()
    providers = available_providers()
    if not providers:
        st.error("No LLM API key configured. Add GOOGLE_API_KEY (and Gemini models) or ANTHROPIC_API_KEY to .env.")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        keys = list(providers)
        provider = st.selectbox("AI provider", keys, format_func=providers.get, key="opt_provider",
                                index=keys.index(s.llm_provider) if s.llm_provider in keys else 0,
                                disabled=locked)
        num_q = st.slider("Number of questions", 4, 15, s.num_questions, key="opt_num_q", disabled=locked)
        include_coding = st.toggle("Include live coding questions", value=True, key="opt_coding_on",
                                   disabled=locked,
                                   help="Coding problems are solved in an in-browser editor (Python or Java) "
                                        "and checked against hidden tests.")
        if include_coding:
            coding_choice = st.selectbox("How many coding questions?", ["Auto (by seniority)", "1", "2", "3"],
                                         key="opt_coding_n", disabled=locked)
            num_coding = None if coding_choice.startswith("Auto") else int(coding_choice)
        else:
            num_coding = 0
            st.caption("Spoken questions only: resume, fundamentals, system design and behavioral.")
    with c2:
        mode = st.radio("Feedback mode", ["practice", "realistic"], key="opt_mode", disabled=locked,
                        format_func={"practice": "Practice (tips after each answer)",
                                     "realistic": "Realistic (feedback at the end)"}.get)
        limit_min = st.slider("Max minutes per spoken answer", 1, 5, s.answer_time_limit_sec // 60,
                              key="opt_limit", disabled=locked)
        coding_min = st.slider("Minutes per coding problem", 10, 40, s.coding_time_limit_sec // 60, step=5,
                               key="opt_coding_min", disabled=locked or num_coding == 0)
    with c3:
        voices = list(VOICES)
        voice = st.selectbox("Interviewer voice", voices, format_func=lambda v: f"{v} ({VOICES[v]})",
                             index=voices.index(s.tts_voice) if s.tts_voice in voices else 0,
                             key="opt_voice", disabled=locked)
        speak = st.toggle("Read questions aloud", value=True, key="opt_speak", disabled=locked)
        name_input = st.text_input("Interviewer name", "Rohan", key="opt_name", disabled=locked)

    if not get_settings().tavily_api_key:
        st.info("TAVILY_API_KEY is not set, so built-in common questions will be used instead of web research.")

    ready = bool(resume_text.strip()) and bool(jd_text.strip())
    label = "Preparing your interview…" if locked else "Start interview"
    if st.button(label, type="primary", disabled=locked or not ready, width="stretch", key="start_btn"):
        s.llm_provider = provider
        s.tts_voice = voice
        s.answer_time_limit_sec = limit_min * 60
        ss.voice, ss.speak = voice, speak
        config = InterviewConfig(
            num_questions=num_q,
            num_coding_questions=num_coding,
            answer_time_limit_sec=limit_min * 60,
            coding_time_limit_sec=coding_min * 60,
            feedback_mode=mode,
            max_follow_ups=s.max_follow_ups,
            company=company.strip() or None,
            interviewer_name=name_input.strip() or "Rohan",
        )
        queue_action("start", "Preparing your interview (about a minute)…",
                     session=InterviewSession(checkpointer=checkpointer()),
                     resume_text=resume_text, jd_text=jd_text, config=config)
    if not ready and not locked:
        st.caption("Add a resume and a job description to start.")
    show_last_error()
    process_action()


# ======================================================================================
# interview
# ======================================================================================


def sidebar(session: InterviewSession, pending: dict | None) -> None:
    locked = busy()
    with st.sidebar:
        st.markdown("### Interview")
        values = session.values
        job = values.get("job")
        if job:
            st.caption(f"{job.title}" + (f" · {job.company}" if job.company else ""))
        if pending:
            st.progress(pending["index"] / pending["total"],
                        text=f"Question {pending['index']} of {pending['total']} · "
                             f"{CATEGORY_LABEL.get(pending['category'], pending['category'])}"
                             + (" (follow-up)" if pending["is_follow_up"] else ""))
        if ss.get("clip"):
            st.caption(f"🔊 {interviewer_name(session.values)} is speaking")
            st.audio(ss.clip, autoplay=True)
        st.divider()
        if st.button("End interview & get feedback", width="stretch", disabled=locked):
            queue_action("end", "Writing your feedback report…")
        if st.button("Pause (continue later)", width="stretch", disabled=locked):
            ss.pop("session", None)
            ss.pop("clip", None)
            st.rerun()
        st.caption(f"Session `{session.thread_id}`")


def coding_view(session: InterviewSession, pending: dict) -> None:
    locked = busy()
    problem = CodingProblem(**pending["coding_problem"])
    qid = pending["question_id"]
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader(f"💻 {problem.title}")
        st.markdown(problem.statement)
        for ex in problem.examples:
            st.markdown(f"- `{ex}`")
        if problem.constraints:
            st.caption("Constraints: " + " · ".join(problem.constraints))
        timer_for(pending)
        language = st.radio("Language", ["python", "java"], horizontal=True, key=f"lang_{qid}",
                            format_func=str.capitalize, disabled=locked)
        code_store = ss.setdefault("code", {})
        code_key = f"{qid}:{language}"
        initial = code_store.get(code_key) or starter_code(problem.signature, language)
        try:
            from streamlit_ace import st_ace

            code = st_ace(value=initial, language=language, theme="tomorrow_night" if theme_type() == "dark" else "chrome",
                          key=f"ace_{code_key}", height=380, font_size=14, tab_size=4, readonly=locked,
                          auto_update=True, show_gutter=True, wrap=False)
        except Exception:  # noqa: BLE001 - component unavailable
            code = st.text_area("Code", initial, height=380, key=f"code_area_{code_key}", disabled=locked)
        code = code if code is not None else initial
        code_store[code_key] = code

        if st.button("▶ Run visible tests", key=f"run_{qid}", disabled=locked):
            with st.spinner("Running your code…"):
                ss.last_run = run_code(problem, code, language, include_hidden=False)
        result = ss.get("last_run")
        if result is not None and result.language == language:
            if not result.compiled:
                st.error("Compile error")
                st.code(result.compile_error or "", language="text")
            else:
                for r in result.results:
                    icon = "✅" if r.passed else "❌"
                    detail = "" if r.passed else f" → expected `{r.expected_json}`, got `{r.output_json or r.error}`"
                    st.markdown(f"{icon} Test {r.index + 1}: `{r.args_json}`{detail}")
                st.caption(f"{result.passed}/{result.total} visible tests passed. Hidden tests run when you submit.")
                if result.stderr.strip():
                    with st.expander("Output / stderr"):
                        st.code(result.stderr[-1500:], language="text")
    with right:
        st.markdown("#### Conversation")
        render_transcript(session.values.get("transcript", []), height=300, name=interviewer_name(session.values))
        st.markdown("#### Explain your approach")
        st.caption("Talk through your approach, then give the time and space complexity. Submitting runs the hidden tests.")
        answer_box(session, pending, "Record your explanation", code=code, language=language)


def spoken_view(session: InterviewSession, pending: dict) -> None:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Conversation")
        render_transcript(session.values.get("transcript", []), height=520, name=interviewer_name(session.values))
    with right:
        st.markdown("#### Your answer")
        st.info(pending["prompt"], icon="🎙️")
        timer_for(pending)
        answer_box(session, pending, "Record your answer")


def interview_view(session: InterviewSession) -> None:
    voice_update(session)
    pending = session.pending()
    sidebar(session, pending)
    if pending is None:
        show_last_error()
        if session.needs_retry or busy():
            if not busy():
                st.warning("The last step didn't finish (usually a temporary API error).")
            if st.button("Retry", type="primary", disabled=busy()):
                queue_action("retry", "Retrying…")
            process_action()
        else:
            st.error("This interview isn't waiting for an answer.")
        return
    if pending.get("coding_problem"):
        coding_view(session, pending)
    else:
        spoken_view(session, pending)


# ======================================================================================
# report
# ======================================================================================


def report_view(session: InterviewSession) -> None:
    voice_update(session)
    with st.sidebar:
        if ss.get("clip"):
            st.caption(f"🔊 {interviewer_name(session.values)} is speaking")
            st.audio(ss.clip, autoplay=True)
        if st.button("Start a new interview", type="primary", width="stretch"):
            for k in ("session", "clip", "draft", "code", "last_run", "prompt_started", "last_error"):
                ss.pop(k, None)
            st.rerun()
    values = session.values
    job = values.get("job")
    st.title("Your interview feedback")
    if job:
        st.caption(f"{job.title}" + (f" at {job.company}" if job.company else ""))
    render_report(values, session.report_markdown(), key_prefix=session.thread_id)
    with st.expander("Full transcript"):
        render_transcript(values.get("transcript", []), name=interviewer_name(values))


# ======================================================================================
# entry
# ======================================================================================

session: InterviewSession | None = ss.get("session")
if session is None or not session.started:
    setup_view()
elif session.done:
    report_view(session)
else:
    interview_view(session)
