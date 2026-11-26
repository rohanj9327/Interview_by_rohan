# Interview by Rohan

Your personal AI interviewer, **Rohan**: a fully spoken mock interview for **software engineers**, built with LangGraph. It reads your resume and a job description, researches commonly asked interview questions, and then runs a tailored interview:

- The interviewer reads each question aloud, and you answer with your voice.
- Technical questions include live coding in **Python or Java**.
- You can ask for a hint during a question, and in practice mode you get a quick tip after each answer.
- At the end you get a detailed report on content, delivery (pace, filler words, pauses) and coding.

It works with either **Claude** or **Gemini** and runs locally on your Mac. See [PLAN.md](PLAN.md) for the full design.

## Requirements

- macOS on Apple Silicon. Speech-to-text uses `mlx-whisper`.
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 automatically.
- JDK 21 or later, for Java coding questions (`java -version`).
- API keys: Anthropic and/or Google AI Studio, and Tavily (free tier).

`ffmpeg` is **not** required. Microphone audio (WAV) is decoded in Python.

## Setup

```bash
cd "/Volumes/rohan/RohuMain/Projects/Technical/Artificial Intelligence/claude/interview_simulator"
uv sync                      # create .venv and install dependencies
cp .env.example .env         # then add your API keys to .env
uv run python -m interview_simulator.check_env   # verify setup
```

## Run

| Phase | Command |
|---|---|
| Phase 1: terminal | `uv run interview-sim --resume data/samples/resume.md --jd data/samples/job_description.md` |
| Phase 2: web UI | `uv run streamlit run app/streamlit_app.py` |

## Web UI

```bash
uv run streamlit run app/streamlit_app.py      # opens http://localhost:8501
```

1. **Setup:** upload your resume, then paste, fetch or upload the job description, or switch on the samples. Pick the provider, number of questions, coding questions, feedback mode, time limits and interviewer voice.
2. **Interview:** the interviewer's voice plays in the sidebar. Record your answer with the mic (allow microphone access in the browser), check or edit the transcript, then submit. You can use 💡 Hint (up to 3 per question), Skip, End, or Pause.
3. **Coding:** the problem appears with a Python/Java editor. ▶ Run checks the visible tests. Record your explanation, then submit, which runs the hidden tests.
4. **Report:** score tiles, a per-question chart, detailed feedback and a Markdown download. Past interviews are on the **History** page, where you can delete the ones you don't want to keep (select rows, then **Delete selected**, or **Delete this interview**). Paused interviews can be discarded from the start page.

## Terminal commands

| Command | What it does |
|---|---|
| Type an answer, then an empty line | Submit (multi-line answers are fine) |
| `/hint` | Get a hint. Hints get stronger in three steps and never reveal the answer |
| `/audio <file.wav>` | Answer with a recorded audio file to test the voice analysis |
| `/end` | Finish now and get the feedback report |
| `/quit` or Ctrl+C | Pause. Resume later with `--resume-session <id>` |
| Coding: `/run`, `/submit`, `/lang java\|python` | Run the visible tests, submit, switch language. You edit the starter file shown in any editor |

Other options: `--company Acme`, `--questions 10`, `--coding 2`, `--mode practice|realistic`, `--speak` (interviewer reads turns aloud), `--history` (list past interviews), `--delete <id> [<id> ...]` (delete interviews), `-v` (verbose logs).

## Tests

```bash
uv run pytest                                   # offline: uses a fake LLM, no API calls
RUN_LIVE=1 uv run pytest tests/test_live.py     # calls the real configured provider
```

## Project layout

```
src/interview_simulator/   core package (config, LangGraph graph, nodes, tools, voice)
app/                       Streamlit UI
data/samples/              sample resume, job description, audio clips
data/runtime/              local runtime data (SQLite, audio cache), git-ignored
tests/                     pytest suite
```

## Status

- [x] Phase 0: project setup
- [x] Phase 1: core graph and terminal interview
- [x] Phase 2: Streamlit UI
