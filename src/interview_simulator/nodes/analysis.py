"""Compare resume with job requirements."""

from interview_simulator.models import GapReport
from interview_simulator.nodes._util import ask_structured, dump
from interview_simulator.prompts import templates as P
from interview_simulator.state import InterviewState


def gap_analysis(state: InterviewState) -> dict:
    prompt = P.GAP_ANALYSIS.format(resume=dump(state["resume"]), job=dump(state["job"]))
    return {"gap_report": ask_structured("evaluator", GapReport, prompt)}
