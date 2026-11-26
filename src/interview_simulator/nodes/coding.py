"""Generate + validate coding problems, and evaluate coding answers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from interview_simulator.models import (
    CodingProblem,
    CodingReview,
    JobProfile,
    PlannedQuestion,
    RunResult,
)
from interview_simulator.nodes._util import ask_structured, log
from interview_simulator.prompts import templates as P
from interview_simulator.prompts.question_bank import FALLBACK_PROBLEMS
from interview_simulator.state import InterviewState
from interview_simulator.tools.code_runner import validate_problem

TOPICS = [
    "arrays and hashing", "sliding window or two pointers", "intervals or sorting",
    "stack or queue", "graph traversal (BFS/DFS on a grid or adjacency list)", "heap / top-k",
]
MAX_ATTEMPTS = 3


def coding_intro(problem: CodingProblem) -> str:
    return (
        f"Now let's do a coding exercise called {problem.title}. The full problem statement and "
        "examples are on your screen. You can use Python or Java. Please talk me through your "
        "approach before and while you code, and tell me the time and space complexity when you're done."
    )


def generate_problem(job: JobProfile, difficulty: str, topic: str, avoid: list[str]) -> CodingProblem | None:
    feedback = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = P.CODING_PROBLEM.format(
            seniority=job.seniority, job_title=job.title, difficulty=difficulty, topic=topic,
            tech_stack=", ".join(job.tech_stack) or "general", avoid=", ".join(avoid) or "none",
            feedback=feedback,
        )
        try:
            problem = ask_structured("evaluator", CodingProblem, prompt)
            errors = validate_problem(problem)
        except Exception as exc:  # noqa: BLE001
            errors = [f"generation error: {exc}"]
            problem = None
        if not errors:
            log.info("Coding problem %r validated (attempt %d)", problem.title, attempt)
            return problem
        log.warning("Coding problem attempt %d failed validation: %s", attempt, errors)
        feedback = (
            "\nYOUR PREVIOUS ATTEMPT FAILED VALIDATION. Fix these issues (or choose a simpler problem):\n- "
            + "\n- ".join(e[:600] for e in errors)
        )
    return None


def generate_coding_problems(state: InterviewState) -> dict:
    job, config = state["job"], state["config"]
    drafts = state["plan_drafts"]
    n = sum(1 for d in drafts if d.category == "coding")
    difficulty = config.difficulty if config.difficulty != "auto" else (
        "easy" if job.seniority in ("intern", "junior") else "medium"
    )
    problems: list[CodingProblem | None] = []
    if n:
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [
                pool.submit(generate_problem, job, difficulty, TOPICS[i % len(TOPICS)], [])
                for i in range(n)
            ]
            problems = [f.result() for f in futures]
    # Replace failures / duplicates with validated built-in problems
    used: set[str] = set()
    fallback = iter(FALLBACK_PROBLEMS)
    final: list[CodingProblem] = []
    for p in problems:
        if p is None or p.title in used:
            p = next((fp for fp in fallback if fp.title not in used), FALLBACK_PROBLEMS[0])
        used.add(p.title)
        final.append(p)

    plan: list[PlannedQuestion] = []
    coding_iter = iter(final)
    for i, d in enumerate(drafts, start=1):
        if d.category == "coding":
            problem = next(coding_iter)
            plan.append(PlannedQuestion(
                id=f"q{i}", text=coding_intro(problem), category="coding",
                target_skill=problem.title, source="generated",
                rubric_hints=[f"Optimal complexity: {problem.expected_time_complexity}",
                              "Clarifies requirements and edge cases", "Explains approach before coding"],
                coding_problem=problem,
            ))
        else:
            plan.append(PlannedQuestion(id=f"q{i}", **d.model_dump()))
    return {"plan": plan, "current_index": 0, "follow_up_count": 0, "hints_used": {},
            "pending_follow_up": None}


def review_coding(state: InterviewState, q: PlannedQuestion, code: str, language: str,
                  explanation: str, run: RunResult, follow_up_allowed: bool) -> CodingReview:
    problem = q.coding_problem
    job = state["job"]
    failures = "\n".join(
        f"- {'hidden' if r.hidden else 'visible'} test args={r.args_json} expected={r.expected_json} "
        f"got={r.output_json} error={r.error}"
        for r in run.results if not r.passed
    )[:2000]
    prompt = P.EVALUATE_CODING.format(
        seniority=job.seniority, job_title=job.title, title=problem.title, statement=problem.statement,
        time_c=problem.expected_time_complexity, space_c=problem.expected_space_complexity,
        language=language, code=code[:8000], passed=run.passed, total=run.total,
        compile_note="" if run.compiled else f" (DID NOT COMPILE: {(run.compile_error or '')[:800]})",
        failures=f"Failed tests:\n{failures}" if failures else "",
        explanation=explanation or "(no explanation given)",
        hints=state.get("hints_used", {}).get(q.id, 0),
        follow_up_rule="" if follow_up_allowed else " — NOT allowed now, always false",
        spoken_style=P.SPOKEN_STYLE,
    )
    return ask_structured("evaluator", CodingReview, prompt)
