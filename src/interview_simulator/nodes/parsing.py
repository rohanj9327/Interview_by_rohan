"""Parse resume and job description into structured profiles (run in parallel)."""

from interview_simulator.models import JobProfile, ResumeProfile
from interview_simulator.nodes._util import ask_structured
from interview_simulator.prompts import templates as P
from interview_simulator.state import InterviewState


def parse_resume(state: InterviewState) -> dict:
    resume = ask_structured("evaluator", ResumeProfile, state["resume_text"], system=P.PARSE_RESUME)
    return {"resume": resume}


def parse_jd(state: InterviewState) -> dict:
    job = ask_structured("evaluator", JobProfile, state["jd_text"], system=P.PARSE_JD)
    company = state["config"].company
    if company:  # user-provided company name wins
        job.company = company
    return {"job": job}
