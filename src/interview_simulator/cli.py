"""Terminal interview runner (Phase 1).

Examples:
    uv run interview-sim --resume data/samples/resume.md --jd data/samples/job_description.md
    uv run interview-sim --resume cv.pdf --jd https://company.com/jobs/123 --company Acme --speak
    uv run interview-sim --resume-session <session-id>     # continue a paused interview
    uv run interview-sim --history                          # list past interviews

During the interview:
    type your answer, then press Enter on an empty line to submit (multi-line supported)
    /hint            ask for a hint            /end    finish now and get feedback
    /quit            pause (resume later with --resume-session <id>)
    /audio <file>    answer with a recorded audio file (WAV/AIFF/FLAC) to test voice analysis
    coding questions: edit the file shown, then /run (visible tests), /submit, /lang java|python
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import textwrap
from pathlib import Path

from interview_simulator.config import get_settings
from interview_simulator.models import CodingProblem, InterviewConfig

WIDTH = min(shutil.get_terminal_size((100, 20)).columns, 100)


def _print_wrapped(prefix: str, text: str) -> None:
    for i, para in enumerate(text.split("\n")):
        print(textwrap.fill(para, WIDTH, initial_indent=prefix if i == 0 else " " * len(prefix),
                            subsequent_indent=" " * len(prefix)) if para else "")


def _progress(events) -> None:
    for label in events:
        print(f"  … {label}", flush=True)


class PauseInterview(Exception):
    """Raised to pause the interview and exit (state is checkpointed)."""


class CLI:
    def __init__(self, session, speak: bool):
        self.session = session
        self.speak = speak
        self.language = "python"
        self.workspace = get_settings().data_path / "workspace" / session.thread_id
        self.workspace.mkdir(parents=True, exist_ok=True)

    # ---- output ------------------------------------------------------------------

    def show_new_turns(self) -> None:
        labels = {"intro": "Interviewer", "question": "Interviewer", "follow_up": "Interviewer (follow-up)",
                  "hint": "Hint", "tip": "Quick tip", "closing": "Interviewer"}
        for turn in self.session.new_turns():
            if turn.role != "interviewer":
                continue
            print()
            _print_wrapped(f"🎙  {labels.get(turn.kind, 'Interviewer')}: ", turn.text)
            if self.speak:
                from interview_simulator.voice import tts
                try:
                    tts.play(tts.synthesize(turn.text))
                except Exception as exc:  # noqa: BLE001
                    print(f"  (speech unavailable: {exc})")

    # ---- input -------------------------------------------------------------------

    @staticmethod
    def read_multiline(prompt: str = "You") -> str:
        print(f"\n✍  {prompt} (empty line to submit, /hint, /end, /quit, /audio <file>):")
        lines = []
        while True:
            try:
                line = input("   ")
            except EOFError:
                if not lines:
                    raise PauseInterview from None
                break
            if line.startswith("/") and not lines:
                return line.strip()
            if not line.strip():
                if lines:
                    break
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def code_file(self, qid: str) -> Path:
        ext = "py" if self.language == "python" else "java"
        return self.workspace / f"{qid}_solution.{ext}"

    def show_problem(self, problem: CodingProblem, qid: str) -> None:
        from interview_simulator.tools.code_runner import starter_code

        print("\n" + "─" * WIDTH)
        print(f"💻 {problem.title}\n")
        _print_wrapped("", problem.statement)
        for ex in problem.examples:
            _print_wrapped("   Example: ", ex)
        if problem.constraints:
            print("   Constraints: " + "; ".join(problem.constraints))
        for lang in ("python", "java"):
            self.language = lang
            path = self.code_file(qid)
            if not path.exists():
                path.write_text(starter_code(problem.signature, lang))
        self.language = "python"
        print("\n   Starter files (edit in any editor):")
        print(f"     Python: {self.code_file(qid)}")
        self.language = "java"
        print(f"     Java:   {self.code_file(qid)}")
        self.language = "python"
        print("   Commands: /run  /submit  /lang java|python  /hint  /end   (current: python)")
        print("─" * WIDTH)

    def handle_coding(self, pending: dict) -> None:
        from interview_simulator.tools.code_runner import run_code

        problem = CodingProblem(**pending["coding_problem"])
        qid = pending["question_id"]
        self.show_problem(problem, qid)
        while True:
            try:
                cmd = input(f"\n[{self.language}] > ").strip()
            except EOFError:
                raise PauseInterview from None
            if cmd == "/quit":
                raise PauseInterview
            if cmd.startswith("/lang"):
                lang = cmd.split()[-1].lower()
                if lang in ("python", "java"):
                    self.language = lang
                    print(f"   Now using {lang}: {self.code_file(qid)}")
            elif cmd == "/run":
                result = run_code(problem, self.code_file(qid).read_text(), self.language, include_hidden=False)
                self.print_run(result)
            elif cmd == "/hint":
                code = self.code_file(qid).read_text()
                _progress(self.session.hint(text="", code=code))
                self.show_new_turns()
            elif cmd == "/end":
                _progress(self.session.end())
                return
            elif cmd == "/submit":
                code = self.code_file(qid).read_text()
                explanation = self.read_multiline("Explain your approach and its time/space complexity")
                _progress(self.session.answer(explanation, code=code, language=self.language))
                return
            else:
                print("   Commands: /run  /submit  /lang java|python  /hint  /end  /quit")

    @staticmethod
    def print_run(result) -> None:
        if not result.compiled:
            print("   ❌ Compile error:\n" + textwrap.indent(result.compile_error or "", "      "))
            return
        for r in result.results:
            mark = "✅" if r.passed else "❌"
            detail = "" if r.passed else f" expected {r.expected_json}, got {r.output_json or r.error}"
            print(f"   {mark} test {r.index + 1}: {r.args_json}{detail}")
        print(f"   {result.passed}/{result.total} visible tests passed")

    def handle_answer(self, pending: dict) -> None:
        limit = pending.get("time_limit_sec")
        print(f"\n   (question {pending['index']}/{pending['total']} · suggested max {limit // 60} min)")
        text = self.read_multiline()
        if text == "/quit":
            raise PauseInterview
        if text == "/hint":
            _progress(self.session.hint())
        elif text == "/end":
            _progress(self.session.end())
        elif text.startswith("/audio"):
            path = text.split(maxsplit=1)[1] if " " in text else ""
            if not Path(path).expanduser().exists():
                print(f"   File not found: {path}")
                return
            from interview_simulator.session import transcribe_answer
            print("  … Transcribing")
            transcript, metrics = transcribe_answer(str(Path(path).expanduser()), self.session.vocabulary())
            print(f"   Transcript: {transcript}")
            print(f"   Delivery: {metrics.wpm} wpm, {metrics.filler_count} fillers, {metrics.long_pauses} long pauses")
            _progress(self.session.answer(transcript, mode="voice", delivery=metrics))
        elif text.startswith("/"):
            print("   Unknown command. Use /hint, /end, /quit or /audio <file>, or type your answer.")
        else:
            _progress(self.session.answer(text))

    # ---- main loop ---------------------------------------------------------------

    def loop(self) -> None:
        self.show_new_turns()
        while not self.session.done:
            pending = self.session.pending()
            if pending is None and self.session.needs_retry:
                print("  … A step failed earlier (e.g. API error); retrying")
                _progress(self.session.retry())
                self.show_new_turns()
                pending = self.session.pending()
                if self.session.done:
                    break
            if pending is None:
                print("Interview is not waiting for input (a step failed). Try again later with "
                      f"--resume-session {self.session.thread_id}")
                return
            if pending.get("coding_problem"):
                self.handle_coding(pending)
            else:
                self.handle_answer(pending)
            self.show_new_turns()

        report = self.session.report_markdown()
        out = get_settings().data_path / "reports" / f"{self.session.thread_id}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print("\n" + "═" * WIDTH + "\n" + report + "\n" + "═" * WIDTH)
        print(f"\nReport saved to {out}")


def show_history() -> None:
    from interview_simulator import history

    rows = history.list_sessions()
    if not rows:
        print("No past interviews yet.")
        return
    print(f"{'session':<14}{'date':<22}{'score':>6}  {'signal':<11}role")
    for r in rows:
        role = f"{r['job_title'] or ''}" + (f" @ {r['company']}" if r["company"] else "")
        print(f"{r['id']:<14}{r['created_at']:<22}{r['overall_score'] or 0:>6}  {r['hire_signal'] or '':<11}{role}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI mock interview (terminal)")
    parser.add_argument("--resume", help="Resume file (PDF/DOCX/TXT/MD)")
    parser.add_argument("--jd", help="Job description: file path, URL, or text")
    parser.add_argument("--company", help="Company name (optional, improves research)")
    parser.add_argument("--questions", type=int, default=None, help="Number of questions (default 10)")
    parser.add_argument("--coding", type=int, default=None, help="Number of coding questions (default auto)")
    parser.add_argument("--mode", choices=["practice", "realistic"], default=None)
    parser.add_argument("--provider", choices=["anthropic", "google_genai"], default=None)
    parser.add_argument("--speak", action="store_true", help="Read interviewer turns aloud")
    parser.add_argument("--resume-session", metavar="ID", help="Continue a paused interview")
    parser.add_argument("--history", action="store_true", help="List past interviews")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="  [%(levelname)s] %(message)s")
    for noisy in ("httpx", "google_genai", "langchain_google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.history:
        show_history()
        return 0

    settings = get_settings()
    if args.provider:
        settings.llm_provider = args.provider
    missing = [k for k in settings.missing_keys() if k != "TAVILY_API_KEY"]
    if missing:
        print(f"Missing API keys in .env: {', '.join(missing)}")
        return 1

    from interview_simulator.session import InterviewSession

    if args.resume_session:
        session = InterviewSession(thread_id=args.resume_session)
        if not session.started:
            print(f"No interview found with id {args.resume_session}")
            return 1
        if session.done:
            print(session.report_markdown())
            return 0
        session._seen_turns = max(0, len(session.values.get("transcript", [])) - 1)
        print(f"Resuming interview {session.thread_id}")
        try:
            CLI(session, args.speak).loop()
        except (KeyboardInterrupt, PauseInterview):
            print(f"\nPaused. Resume with: uv run interview-sim --resume-session {session.thread_id}")
        return 0

    if not args.resume or not args.jd:
        parser.error("--resume and --jd are required (or use --resume-session / --history)")

    from interview_simulator.tools.document_loader import load_document
    from interview_simulator.tools.jd_fetcher import resolve_jd

    resume_text = load_document(args.resume)
    jd_text = resolve_jd(args.jd)
    config = InterviewConfig(
        num_questions=args.questions or settings.num_questions,
        num_coding_questions=args.coding,
        answer_time_limit_sec=settings.answer_time_limit_sec,
        coding_time_limit_sec=settings.coding_time_limit_sec,
        feedback_mode=args.mode or settings.feedback_mode,
        max_follow_ups=settings.max_follow_ups,
        company=args.company,
    )
    session = InterviewSession()
    print(f"Interview session {session.thread_id}  (resume later with --resume-session {session.thread_id})")
    print(f"Provider: {settings.llm_provider} · Preparing your interview, this takes about a minute…")
    try:
        _progress(session.start(resume_text, jd_text, config))
        CLI(session, args.speak).loop()
    except (KeyboardInterrupt, PauseInterview):
        print(f"\nPaused. Resume with: uv run interview-sim --resume-session {session.thread_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
