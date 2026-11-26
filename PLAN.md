# AI Interview Simulator — Project Plan

## 1. Goal

Build an AI application that:

Target role for MVP: **Software Engineer** (see §6c). Phase 1–2 run **locally on your Mac (Apple Silicon)** only.

1. **Reads** a candidate's resume (PDF/DOCX/TXT) and a job description (text or URL).
2. **Researches** commonly asked interview questions for that role/company from the internet.
3. **Builds** a tailored interview plan (mix of technical, behavioral, and resume-deep-dive questions).
4. **Conducts** a fully spoken interview — the interviewer **reads each question aloud** (TTS), asks one question at a time, listens to the answer (**spoken via microphone** or typed; English only), asks follow-ups when an answer is vague or interesting.
5. **Evaluates** each answer against a rubric — content for all answers, plus delivery (pace, filler words, pauses, confidence) for spoken answers.
6. **Coaches during the interview** — on-demand progressive hints and a quick tip after each answer (practice mode).
7. **Includes live coding** for technical questions — in-browser editor, code runs against test cases.
8. **Delivers** a final report: what was strong, and detailed, actionable feedback on what to improve (with example better answers).

---

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Best LangGraph/LangChain support |
| Orchestration | **LangGraph** | Stateful graph, loops, human-in-the-loop `interrupt()`, checkpointing |
| LLM | **Multi-provider: Claude + Gemini** via `langchain-anthropic` and `langchain-google-genai`, selected through a provider-agnostic model factory (`init_chat_model`) | User picks provider in `.env` or UI; two roles per provider — a fast "interviewer" model and a stronger "feedback" model (see §6a) |
| Structured output | Pydantic v2 + `.with_structured_output()` | Reliable parsing of resume/JD/evaluations |
| Web research | **Tavily** search API (free tier, ~1,000 searches/month) | Built for LLM agents, returns clean content; results cached per role/company to save quota |
| Document parsing | `pypdf` / `pymupdf`, `python-docx` | Resume ingestion |
| JD from URL | `httpx` + `trafilatura` | Extract main text from job posting pages |
| Voice input (STT) | **`mlx-whisper`** (local, Apple Silicon GPU) — alternatives: `faster-whisper` (non-Mac), Gemini native audio | Fast on M-series chips, free, offline, word timestamps for pace/filler/pause analysis (see §6b) |
| Voice output (TTS) | **`edge-tts`** (free, natural English neural voices) — offline fallback: macOS `say` / `pyttsx3` | Interviewer speaks every question aloud (MVP) |
| Code editor | `streamlit-ace` | In-browser editor with syntax highlighting for coding questions (§6d) |
| Code execution | **Python + Java** via `subprocess` (`python3`; `javac` + `java` from JDK 21) with timeout + resource limits (local only) | Run candidate code against test cases; replace with Docker sandbox before deploying |
| Audio capture | Streamlit `st.audio_input` (built-in mic widget) | No custom JS needed |
| Persistence | LangGraph `SqliteSaver` checkpointer + `history.db` (SQLite) | Resume interrupted interviews; saved history of all sessions & reports for progress tracking |
| Deployment | **Local only** in Phase 1–2 (`streamlit run` on your Mac); hosting considered in Phase 3 | Keeps audio, resume, and code execution on your machine |
| UI | **Streamlit** (decided) | Fastest path; built-in mic widget (`st.audio_input`) and audio playback (`st.audio(autoplay=True)`) |
| Observability | LangSmith (optional) | Trace graph runs, debug prompts |
| Testing | pytest + recorded LLM fixtures | Deterministic tests for nodes |
| Packaging | `uv` + `pyproject.toml` | Fast, reproducible env |

---

## 3. High-Level Architecture

```
┌─────────────┐     ┌──────────────┐
│  Resume     │     │  Job Desc    │
│ (PDF/DOCX)  │     │ (text / URL) │
└──────┬──────┘     └──────┬───────┘
       ▼                   ▼
┌─────────────┐     ┌──────────────┐
│parse_resume │     │  parse_jd    │      (run in parallel)
└──────┬──────┘     └──────┬───────┘
       └────────┬──────────┘
                ▼
        ┌───────────────┐
        │ gap_analysis  │  match skills, find gaps & strengths
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │web_research   │  Tavily: "<role> <company> interview questions"
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │plan_interview │  build 10 Qs incl. 1–2 coding problems (+ tests)
        └───────┬───────┘
                ▼
        ┌───────────────┐◄───────────────────────┐
        │ ask_question  │                         │
        └───────┬───────┘                         │
                ▼                                 │
        ┌───────────────┐   hint request          │
        │ await_answer  │◄──────────┐             │
        │ interrupt()   │───────────►give_hint    │
        └───────┬───────┘  (voice answer or code) │
                ▼                                 │
        ┌───────────────┐                         │
        │evaluate_answer│  content / coding /     │
        └───────┬───────┘  delivery rubric        │
                ▼                                 │
        ┌───────────────┐  (practice mode only)   │
        │  quick_tip    │  1–2 sentence tip       │
        └───────┬───────┘                         │
                ▼                                 │
        ┌───────────────┐  follow_up / next ──────┘
        │    router     │
        └───────┬───────┘
                ▼ done (all Qs asked / user says "end")
        ┌───────────────┐
        │final_feedback │  strengths, improvements, sample answers
        └───────────────┘
```

---

## 4. LangGraph State Design

```python
class InterviewState(TypedDict):
    # Inputs
    resume_text: str
    jd_text: str
    config: InterviewConfig          # difficulty, num_questions=10, answer_time_limit=180s, feedback_mode, voice, persona

    # Parsed / derived
    resume: ResumeProfile            # name, skills, experience[], projects[], education
    job: JobProfile                  # title, company, seniority, must_have[], nice_to_have[]
    gap_report: GapReport            # matched_skills, missing_skills, talking_points
    researched_questions: list[ResearchedQuestion]   # question, source_url, category

    # Interview loop
    plan: list[PlannedQuestion]      # id, text, category (incl. coding), target_skill, rubric_hints, coding_problem?
    current_index: int
    follow_up_count: int             # follow-ups on current question (cap at 2)
    hints_used: dict[str, int]       # question_id → hint level reached
    transcript: Annotated[list[Turn], operator.add]  # Q / A / follow-up / hint / tip turns (mode: text|voice, audio_path?, code?)
    evaluations: Annotated[list[AnswerEvaluation], operator.add]

    # Output
    final_report: FinalReport | None
```

**Key Pydantic models**

- `AnswerEvaluation`: `question_id, scores{relevance, depth, structure(STAR), technical_accuracy, communication} (1–5), strengths[], weaknesses[], needs_follow_up: bool, follow_up_question?, delivery?: DeliveryEvaluation`
- `DeliveryEvaluation` (voice answers only): `wpm, filler_count, filler_words{word: count}, long_pauses, duration_sec, pace_score, fluency_score, confidence_score (1–5), notes[]`
- `CodingProblem` / `CodingEvaluation`: see §6d
- `FinalReport`: `overall_score, hire_signal, top_strengths[], improvement_areas[{area, why, how_to_improve, example_better_answer}], per_question_breakdown[], coding_summary?, hints_used_summary, delivery_summary? (avg pace, fillers, trend across interview), recommended_resources[]`

---

## 5. Node Responsibilities

| Node | Input | Output | Notes |
|---|---|---|---|
| `parse_resume` | resume_text | `ResumeProfile` | Structured extraction |
| `parse_jd` | jd_text | `JobProfile` | Detect seniority, must-have skills |
| `gap_analysis` | resume, job | `GapReport` | Drives which areas to probe hardest |
| `web_research` | job | `researched_questions` | 2–4 Tavily queries; dedupe; keep sources |
| `plan_interview` | all above + config | `plan` | SWE mix per §6c (resume, CS fundamentals, system design, behavioral, company-specific); calibrated to seniority; include common questions from research |
| `ask_question` | plan, index | appends Q to transcript | Interviewer persona, natural, *spoken-friendly* phrasing (short sentences, no markdown/code blocks); UI converts it to speech via TTS |
| `await_answer` | — | appends A to transcript | Uses LangGraph `interrupt()`; resumed with `Command(resume={type: answer\|hint_request\|end, text, mode, code?, test_results?, delivery_metrics?})`. Transcription and code execution happen in the UI/tool layer before resuming |
| `give_hint` | Q, partial answer/code, hint level | hint turn | Progressive hint (§6e); loops back to `await_answer` |
| `evaluate_answer` | Q, A, rubric, delivery metrics?, code + test results? | `AnswerEvaluation` / `CodingEvaluation` | Scores not shown mid-interview; adds delivery scores for voice answers and coding scores for coding questions |
| `quick_tip` | evaluation | tip turn | Practice mode only; 1–2 sentences, spoken |
| `router` | evaluation, counters | next edge | `follow_up` if vague & cap not hit; `next` otherwise; `end` when plan exhausted or user types "end" |
| `final_feedback` | transcript, evaluations, gap_report | `FinalReport` | Uses stronger model; cites specific answers; saved to `history.db` |

---

## 6. Project Structure

```
interview_simulator/
├── PLAN.md
├── README.md
├── pyproject.toml
├── .env.example                 # LLM_PROVIDER, ANTHROPIC/GOOGLE/TAVILY keys, model names
├── src/interview_simulator/
│   ├── __init__.py
│   ├── config.py                # settings via pydantic-settings
│   ├── models.py                # Pydantic schemas (Resume, Job, Evaluation, Report...)
│   ├── state.py                 # InterviewState
│   ├── llm.py                   # provider-agnostic model factory (Claude / Gemini)
│   ├── prompts/                 # one .md/.py prompt per node
│   ├── tools/
│   │   ├── document_loader.py   # PDF/DOCX/TXT → text
│   │   ├── jd_fetcher.py        # URL → text
│   │   ├── web_search.py        # Tavily wrapper (+ result cache)
│   │   ├── code_runner.py       # run candidate code vs tests (subprocess, timeout, limits)
│   │   └── harness/             # per-language drivers: python, java
│   ├── voice/
│   │   ├── stt.py               # mlx-whisper (default) / faster-whisper / Gemini transcription (English)
│   │   ├── tts.py               # edge-tts (fallback: macOS say) → MP3 for questions
│   │   └── delivery.py          # wpm, filler words, pauses from word timestamps
│   ├── nodes/
│   │   ├── parsing.py
│   │   ├── analysis.py
│   │   ├── research.py
│   │   ├── planning.py
│   │   ├── interview.py         # ask / await / evaluate / router
│   │   ├── coding.py            # generate & validate coding problems, coding evaluation
│   │   ├── hints.py             # give_hint, quick_tip
│   │   └── feedback.py
│   ├── history.py               # save/load past sessions & reports (SQLite)
│   ├── graph.py                 # builds & compiles StateGraph with checkpointer
│   └── cli.py                   # terminal interview runner (Phase 1)
├── app/
│   ├── streamlit_app.py         # entry point (st.navigation)
│   ├── ui_common.py             # charts, report & transcript rendering
│   └── views/
│       ├── interview.py         # setup → interview → report
│       └── history.py           # past interviews & score trends
├── data/samples/                # sample resume + JD for testing
└── tests/
    ├── test_parsing.py
    ├── test_router.py
    ├── test_code_runner.py
    ├── test_delivery.py
    └── test_graph_e2e.py
```

---

## 6a. Multi-Provider LLM Support (Claude + Gemini)

**Approach:** nodes never import a provider SDK directly. They call `get_llm(role)` from `llm.py`, which returns a LangChain `BaseChatModel`. Because both providers implement the same interface (`.invoke`, `.stream`, `.with_structured_output`), the graph is provider-agnostic.

```python
# llm.py
from langchain.chat_models import init_chat_model

def get_llm(role: Literal["interviewer", "evaluator", "feedback"]) -> BaseChatModel:
    provider = settings.llm_provider            # "anthropic" | "google_genai"
    model = settings.models[provider][role]
    return init_chat_model(model, model_provider=provider, temperature=settings.temps[role])
```

**`.env.example`**

```bash
LLM_PROVIDER=anthropic          # anthropic | google_genai
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=                 # from Google AI Studio
TAVILY_API_KEY=

# Role → model mapping (override as needed)
ANTHROPIC_INTERVIEWER_MODEL=claude-sonnet-5-5
ANTHROPIC_FEEDBACK_MODEL=claude-opus-5-5
GEMINI_INTERVIEWER_MODEL=<latest Gemini Flash model>
GEMINI_FEEDBACK_MODEL=<latest Gemini Pro model>
```

**Role mapping**

| Role | Used by | Claude | Gemini |
|---|---|---|---|
| `interviewer` | ask_question, follow-ups, parsing | Sonnet | Flash |
| `evaluator` | evaluate_answer, gap_analysis, planning | Sonnet | Flash / Pro |
| `feedback` | final_feedback | Opus | Pro |

**Things to handle explicitly**

- **Structured output differences:** test every Pydantic schema against both providers; avoid schema features Gemini handles poorly (deeply nested `Union`s, `Optional` with complex defaults). Keep schemas flat-ish and use `Literal` enums.
- **System prompts:** both support system messages via LangChain; keep prompts provider-neutral (no XML-tag-only conventions that one model follows worse).
- **Fallbacks:** optional `llm.with_fallbacks([other_provider_llm])` so a rate limit/outage on one provider switches to the other.
- **Native file input (optional):** both Claude and Gemini accept PDFs directly — can be used as a fallback when text extraction from a resume fails.
- **UI:** provider dropdown in Streamlit settings; show which model generated the report.
- **Tests:** parametrize integration tests over `["anthropic", "google_genai"]`; a provider is skipped if its API key is missing.

---

## 6b. Voice Input & Spoken-Answer Judging (English only)

**Voice is a required, primary input mode** — the interview UI defaults to the microphone. Typing is kept only as a fallback (mic unavailable / accessibility). Each spoken answer is judged on **content** (same rubric as typed answers) **and delivery** (how it was said).

### Speech-to-text (STT) options

| Option | Type | Pros | Cons | Use |
|---|---|---|---|---|
| **`mlx-whisper`** (`whisper-small.en` / `whisper-large-v3-turbo`, MLX builds) | Local, Apple Silicon GPU | Fast on M-series, accurate English, offline, private, **word-level timestamps** (needed for delivery metrics) | Mac-only; first model download | **Default (your Mac)** |
| `faster-whisper` (`small.en`) | Local, CPU | Cross-platform, word timestamps | CPU-only on Mac → slower | Fallback for non-Mac / deployment |
| **Gemini native audio** (send audio to Gemini Flash) | Cloud | No local model; can transcribe *and* comment on tone in one call | No reliable word timestamps; costs per call | Alternative, when `LLM_PROVIDER=google_genai` and local STT is not wanted |

> Claude does not take audio input, so with Claude the STT step always uses local Whisper → text → Claude. The STT layer is separate from the LLM provider (`STT_ENGINE=mlx_whisper | faster_whisper | gemini`).

### Voice pipeline

```
🎤 Browser mic (st.audio_input)  ──►  WAV bytes
        ▼
 transcribe (mlx-whisper, language="en", word_timestamps=True)
        ▼
 ┌──────────────────────────┬─────────────────────────────┐
 │ transcript text          │ delivery metrics (Python):  │
 │  → evaluate_answer       │  • words per minute         │
 │    (content rubric)      │  • filler words (um, uh,    │
 │                          │    like, you know, basically)│
 │                          │  • long pauses (> 2s)       │
 │                          │  • answer duration          │
 └──────────────────────────┴─────────────────────────────┘
        ▼
 AnswerEvaluation (content scores + delivery scores + notes)
```

- Delivery metrics are **computed deterministically in Python** from Whisper's word timestamps (not guessed by the LLM), then passed to the LLM so its feedback can quote them ("you said 'um' 14 times; pace was 185 wpm — aim for 130–160").
- Recording is capped at **3 minutes per answer** (configurable); coding questions use a separate **20-minute** timer for the editor.
- User sees the transcript and can **re-record** before submitting (avoids judging a bad recording).
- **Interviewer voice (MVP):** every question/follow-up is converted to speech with `edge-tts` (default `en-US-GuyNeural`; `en-US-JennyNeural` etc. selectable) and auto-played; text is shown too. Offline fallback: macOS `say`. Audio is cached per question so replays are instant.

### Delivery rubric (voice answers only)

| Dimension | Measured by | Healthy range |
|---|---|---|
| Pace | words per minute | ~130–160 wpm |
| Filler words | count / per minute | < 3 per minute |
| Pauses | count of gaps > 2s | few, not mid-sentence |
| Conciseness | answer duration vs. question type | ~1–2 min behavioral, ~2–3 min technical |
| Clarity / confidence | LLM read of transcript (hedging like "I think maybe", restarts) | low hedging |

---

## 6c. Software Engineer Focus (MVP target role)

The MVP is tuned for **Software Engineer** interviews (junior → senior, detected from the JD). Other roles can be added later by adding a new question-category config.

**Question mix (default 10 questions)**

| Category | Share | Examples |
|---|---|---|
| Resume / project deep-dive | ~25% | "Walk me through the architecture of <project>. What would you change?" |
| CS fundamentals (verbal) | ~15% | Data structures, complexity, concurrency, language-specific questions from the JD stack |
| **Live coding** (§6d) | ~10–20% (1–2 problems) | Problem tailored to JD stack, solved in editor + think-aloud |
| System design (verbal, scaled to seniority) | ~20% | "Design a URL shortener / rate limiter" — junior: smaller scope; senior: scale, trade-offs |
| Behavioral (STAR) | ~20% | Conflict, failure, ownership, tight deadline |
| Role/company-specific | ~5% | From web research for that company + role |

**Web research queries (examples):** `"<company> software engineer interview questions"`, `"<tech stack> interview questions <seniority>"`, `"<company> system design interview"`, `"<company> behavioral interview"`.

**SWE-specific rubric additions**
- *Technical accuracy*: correctness of concepts, complexity analysis.
- *Problem-solving approach*: clarifies requirements, states assumptions, discusses trade-offs.
- *System design*: requirements → high-level design → bottlenecks → scaling.
- *Ownership & impact*: concrete metrics in project answers.

**Seniority calibration:** `plan_interview` adjusts difficulty and system-design depth based on `JobProfile.seniority` and years of experience in `ResumeProfile`.

---

## 6d. Live Coding Questions

Technical portions of the interview include **1–2 live coding questions** (count scales with seniority). Coding applies to technical questions only — behavioral and resume questions stay voice-only.

**Flow**
1. Interviewer reads the problem aloud; problem statement + examples shown in UI.
2. Candidate **thinks aloud by voice** (approach, complexity) and writes code in an in-browser editor (`streamlit-ace`) in **Python or Java** (language picked in settings or per problem).
3. "Run" executes visible test cases; "Submit" runs visible + hidden tests.
4. Interviewer may ask a follow-up ("What's the complexity? Can you do better? What about edge case X?").

**Problem source:** LLM generates problems tailored to JD stack + seniority, seeded with common patterns found by web research (arrays/hashing, two pointers, BFS/DFS, DP…). Each problem is a structured `CodingProblem`: `title, statement, examples[], constraints, signature{name, params[{name, type}], return_type}` (language-neutral types: `int, long, double, bool, string, int[], string[], int[][], list<...>`), `visible_tests[]`, `hidden_tests[]` (JSON inputs/outputs), `starter_code{python, java}`, `reference_solution{python, java}`, `expected_complexity`. Both reference solutions are run against all tests at generation time — problems that fail in either language are regenerated.

**Multi-language harness:** tests are stored once as JSON; `tools/harness/` holds a template per language that wraps the candidate's function, parses JSON inputs, calls the function, and prints JSON results.
- *Python:* `solution.py` + `harness_py.py`, run with `python3`.
- *Java:* candidate writes `class Solution { ... }`; a generated `Main.java` driver (uses a tiny bundled JSON parser, no Maven) is compiled with `javac` once per submission, then run with `java` for all tests in a single JVM to avoid per-test startup cost. Compile errors are shown to the candidate like a real IDE.

**Code execution (local only):** `tools/code_runner.py` runs the candidate's code in a **subprocess** inside a temp dir with a timeout (e.g. 5s per test), memory cap (`resource.setrlimit`), and no network access expected. Acceptable because Phase 1–2 run only on your machine; must be replaced by a real sandbox (Docker / Firecracker / hosted runner) before any deployment.

**Coding evaluation (`CodingEvaluation`)**

| Dimension | Measured by |
|---|---|
| Correctness | tests passed / total (deterministic) |
| Efficiency | LLM compares to expected complexity + runtime on large hidden test |
| Code quality | LLM review: naming, structure, edge-case handling |
| Problem-solving process | voice think-aloud transcript: clarified requirements, discussed approach before coding |
| Communication | delivery metrics from §6b |
| Hints used | count of hints requested (noted, small score impact) |

---

## 6e. Hints & Feedback Modes

The user gets **both** in-interview hints **and** the detailed end-of-interview report.

| Type | When | What |
|---|---|---|
| **On-demand hint** | User clicks "I need a hint" while answering (any question; most useful for coding/system design) | Progressive: level 1 = gentle nudge, level 2 = direction, level 3 = key insight. Never gives the full answer. Hint count is recorded and mentioned in feedback |
| **Post-answer quick tip** | After each answer is evaluated (practice mode) | 1–2 sentences, spoken + shown: one thing done well, one thing to improve. Scores are still not shown until the end |
| **Final report** | End of interview | Full detailed feedback (§8) |

`InterviewConfig.feedback_mode`: `practice` (default — quick tips on) or `realistic` (no tips; on-demand hints still available but flagged).

---

## 7. Implementation Phases

### Phase 0 — Setup (½ day)
- [x] `uv init`, add deps: `langgraph langchain langchain-anthropic langchain-google-genai langchain-community tavily-python pypdf python-docx trafilatura pydantic-settings streamlit streamlit-ace mlx-whisper edge-tts`
- [x] `.env.example`, config loader, sample resume + JD in `data/samples/`, `check_env.py`, config tests
- [x] `ffmpeg` made **optional**: mic WAV is decoded in Python (`soundfile` + `scipy` resample to 16 kHz) and passed to Whisper as a numpy array
- [x] JDK verified (OpenJDK 24 present; ≥21 required)
- [ ] User adds `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY` to `.env` (git-ignored)

### Phase 1 — Core graph, CLI, local only (4–5 days)
- [x] Pydantic models + state
- [x] `llm.py` model factory supporting Claude + Gemini; smoke-test structured output on both
- [x] Resilience: retries with backoff + same-provider fallback chain (e.g. `gemini-2.5-flash` → `gemini-3.1-flash-lite`) for 503 "high demand" / 429 quota errors (`GEMINI_FALLBACK_MODELS` in `.env`). Observed during setup: newer Flash models often 503, Pro models unavailable on free quota
- [x] Document loader & JD fetcher
- [x] Parsing + gap analysis nodes
- [x] Web research node (Tavily) with dedupe & source tracking
- [x] Interview planner
- [x] Interview loop with `interrupt()` + SQLite checkpointer
- [x] Evaluator + router (follow-up cap = 2)
- [x] Hints: `give_hint` (progressive, on-demand) + `quick_tip` (practice mode)
- [x] Coding: problem generation + reference-solution validation (Python + Java), `code_runner.py` + per-language harness, coding evaluation
- [x] `history.py`: save completed sessions & reports to SQLite
- [x] Final feedback node
- [x] `voice/stt.py`, `voice/tts.py` + `voice/delivery.py` with unit tests on sample audio clips (`data/samples/audio/`)
- [x] `cli.py`: run full interview in terminal; print report as Markdown

**Milestone (done):** complete an interview end-to-end in the terminal on your Mac (incl. one coding question solved from a file, hints, final report saved to history).

**Phase 1 notes:**
- Shared `session.py` wrapper (start / answer / hint / end / pending / new_turns) used by CLI and, next, Streamlit.
- CLI voice is tested via `/audio <file>`; live mic recording comes with the Streamlit UI.
- STT is biased with JD/resume terms (`vocabulary`) to spell technical words correctly.
- Checkpoint serializer allow-lists our Pydantic models; pause with `/quit` or Ctrl+C, resume with `--resume-session`.
- Tests: 28 offline (fake LLM end-to-end graph, code runner Python/Java, voice metrics, CLI) + `RUN_LIVE=1` live test.

### Phase 2 — Streamlit UI, local only (3–4 days)
- [x] Upload resume / paste JD or URL
- [x] Config: LLM provider (Claude / Gemini), difficulty, # questions (default 10), answer time limit (default 3 min), feedback mode (practice / realistic), interviewer persona
- [x] Chat interface driven by graph `stream()` + `Command(resume=...)`
- [x] **Voice answers:** `st.audio_input` mic recorder → transcribe → show editable transcript → submit; text box as fallback
- [x] **Interviewer voice:** each question rendered with TTS and auto-played (`st.audio(autoplay=True)`), with replay button and voice selector
- [x] Settings: STT engine (local Whisper / Gemini), Whisper model size, interviewer voice
- [x] **Coding panel:** problem statement, language toggle (Python / Java) with starter code, `streamlit-ace` editor, Run / Submit buttons, test results, 20-min timer, mic for think-aloud
- [x] **"I need a hint"** button + spoken quick tips after each answer (practice mode)
- [x] Countdown timer + auto-stop at answer time limit
- [x] "End interview early" button
- [x] Report page: score cards, strengths, improvement areas, per-question breakdown, coding results, download as Markdown
- [ ] PDF export (deferred; Markdown download available)
- [x] **History page:** list of past interviews, reopen reports, score trends over time

**Phase 2 notes:**
- Pages: `app/views/interview.py` (setup → interview → report) and `app/views/history.py`, wired with `st.navigation` in `app/streamlit_app.py`.
- New interviewer turns are combined into one TTS clip that autoplays in the sidebar (stable across reruns, so typing in the editor doesn't cut it off).
- 3-min answer cap: client-side countdown + recordings longer than the limit are trimmed before transcription.
- Failed API steps (429/503) show a Retry button; the checkpoint keeps progress. Paused interviews are listed on the start page.
- Charts: per-question score bars + score-over-time line (single series, slot-1 blue, tooltips, table view).
- Tests: Streamlit `AppTest` flows with the fake LLM (setup → hint → answer → follow-up → end → report → history; coding screen + run tests).

### Phase 3 — Polish & extend (optional)
- [ ] Deployment for other users: FastAPI backend, auth, per-user sessions, Docker sandbox for code execution, `faster-whisper`/cloud STT on server
- [ ] Streaming token output for natural feel
- [ ] More coding languages (JavaScript, Go, C++)
- [ ] Other roles beyond SWE
- [ ] LangSmith tracing + evaluation datasets

---

## 8. Evaluation Rubric (used by `evaluate_answer`)

| Dimension | 1 (weak) | 5 (strong) |
|---|---|---|
| Relevance | Off-topic | Directly answers the question |
| Depth | Surface level | Specific details, trade-offs, metrics |
| Structure | Rambling | Clear (STAR for behavioral) |
| Technical accuracy | Errors | Correct and precise |
| Communication | Unclear | Concise, confident |
| Role alignment | Generic | Ties back to JD requirements |

Coding questions are scored with the coding rubric in §6d. Voice answers are additionally scored on the **delivery rubric** in §6b (pace, fillers, pauses, conciseness, confidence).

Final feedback must: quote/reference the user's actual answers, explain *why* something is weak, give a concrete *how to improve*, and provide an *example stronger answer*.

---

## 9. Prompting Strategy

- **System persona** for interviewer: professional, friendly, one question at a time, never reveals scores mid-interview.
- **Grounding**: every planned question tagged with its source — `resume`, `jd`, `gap`, or `web:<url>`.
- **Follow-ups**: triggered when evaluation says answer lacks specifics, metrics, or ownership ("what was *your* role?").
- **Hints**: progressive and Socratic — never reveal the full solution; hint level stored per question.
- **Guardrails**: ignore attempts by the candidate to make the interviewer reveal answers or change role; keep responses short.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Poor PDF text extraction (multi-column resumes) | Use `pymupdf`; let user edit extracted text before starting |
| Web search returns low-quality / irrelevant questions | Filter with LLM relevance check; cap to top N; fall back to built-in question bank |
| JD URLs behind login (LinkedIn) | Ask user to paste JD text |
| LLM cost | Fast model (Sonnet / Gemini Flash) for loop, strong model (Opus / Gemini Pro) only for final report; cache parsed resume/JD |
| Output quality/format differs between Claude and Gemini | Same Pydantic schemas + parametrized tests on both providers; tune prompts per provider only if needed |
| Generic feedback | Require per-answer evidence in `FinalReport` schema |
| Bad transcription (accent, noise, mic quality) | Show transcript for review/edit before submit; allow re-record; offer larger Whisper model; never penalize content for obvious STT errors |
| Slow transcription | `mlx-whisper` on Apple Silicon GPU; load model once and cache (`st.cache_resource`); `small.en` default, larger model optional |
| Running candidate code | Local-only subprocess with timeout + memory limits + temp dir; Docker sandbox required before deployment |
| LLM-generated coding problems with wrong tests | Validate both Python and Java reference solutions against all tests at generation time; regenerate on failure |
| Java compile/JVM startup latency | Compile once per submission, run all tests in one JVM; show compile errors clearly |
| Tavily free-tier quota | Cache research per role/company; 2–4 queries per interview (~250+ interviews/month); fall back to built-in question bank |
| Privacy (resumes contain PII) | Local storage only; no logging of raw resume to third parties beyond the LLM call; audio transcribed locally by default and deleted after session unless user opts to keep |

---

## 11. Definition of Done (MVP)

- Upload resume + JD → interview starts in < 30s
- 10 tailored questions (configurable) incl. at least 2 sourced from web research and 1–2 live coding problems with passing reference tests
- Adaptive follow-ups work
- Final report shows strengths, detailed improvement areas with example answers, and per-question scores
- Session can be paused and resumed (checkpointer); completed interviews appear on the History page
- On-demand hints and post-answer quick tips work; 3-min answer cap enforced
- Everything runs locally on your Mac
- Full interview works with both Claude and Gemini by switching `LLM_PROVIDER`
- Interviewer speaks every question aloud; user answers by voice (English); final report includes delivery feedback (pace, filler words, pauses) alongside content feedback

---

## 12. Decisions

- **LLM provider:** Claude + Gemini (factory makes adding OpenAI/Ollama later a config change)
- **Voice:** voice input required (primary mode), English only; local Whisper STT + delivery judging
- **TTS:** interviewer speaks questions aloud in MVP (`edge-tts`)
- **UI:** Streamlit
- **Target role:** Software Engineer (§6c)
- **Where it runs:** locally on your Mac in Phase 1–2; deployment is a Phase 3 consideration
- **Hardware:** Apple Silicon → `mlx-whisper` for STT
- **Coding:** live coding panel for technical questions (1–2 problems per interview), **Python and Java** (§6d)
- **Feedback:** on-demand hints + post-answer quick tips **and** detailed final report (§6e)
- **Length:** 10 questions, 3-min cap per spoken answer, 20-min timer per coding problem
- **Web research:** Tavily free tier; company name optional
- **History:** saved locally in SQLite with a History page
- **API keys:** user adds them to `.env` (git-ignored); `.env.example` documents them
- **Interviewer voice:** default `en-US-GuyNeural`, changeable in settings

No open questions — ready to start Phase 0.
