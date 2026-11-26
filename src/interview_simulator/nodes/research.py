"""Web research: commonly asked interview questions for this role/company (Tavily)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from interview_simulator.models import JobProfile, ResearchedQuestions
from interview_simulator.nodes._util import ask_structured, log
from interview_simulator.prompts import templates as P
from interview_simulator.prompts.question_bank import COMMON_QUESTIONS
from interview_simulator.state import InterviewState
from interview_simulator.tools import web_search


def build_queries(job: JobProfile) -> list[str]:
    title = job.title or "software engineer"
    stack = " ".join(job.tech_stack[:3])
    queries = [
        f"{job.seniority} {title} interview questions",
        f"{stack} interview questions for {job.seniority} engineers" if stack else None,
        f"{job.seniority} software engineer system design interview questions",
    ]
    if job.company:
        queries = [
            f"{job.company} software engineer interview questions",
            f"{job.company} {title} interview experience",
        ] + queries[:2]
    return [q for q in queries if q][:4]


def web_research(state: InterviewState) -> dict:
    job = state["job"]
    queries = build_queries(job)
    with ThreadPoolExecutor(max_workers=4) as pool:
        batches = list(pool.map(web_search.search, queries))
    seen, results = set(), []
    for batch in batches:
        for r in batch:
            if r["url"] not in seen:
                seen.add(r["url"])
                results.append(r)
    if not results:
        log.info("No web results; using built-in question bank")
        return {"researched_questions": list(COMMON_QUESTIONS)}

    text = "\n\n".join(f"[{r['url']}] {r['title']}\n{r['content']}" for r in results[:15])
    prompt = P.RESEARCH_EXTRACT.format(
        job_title=job.title, seniority=job.seniority, company=job.company or "(not specified)",
        tech_stack=", ".join(job.tech_stack), results=text,
    )
    try:
        questions = ask_structured("evaluator", ResearchedQuestions, prompt).questions
    except Exception as exc:  # noqa: BLE001 - research is best-effort
        log.warning("Question extraction failed: %s", exc)
        questions = []
    if len(questions) < 5:
        questions += COMMON_QUESTIONS
    return {"researched_questions": questions}
