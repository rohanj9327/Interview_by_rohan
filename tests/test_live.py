"""Live checks against the configured LLM provider. Skipped unless RUN_LIVE=1.

    RUN_LIVE=1 uv run pytest tests/test_live.py -q
"""

import os

import pytest

from interview_simulator.config import get_settings
from interview_simulator.models import JobProfile, ResumeProfile
from interview_simulator.nodes._util import ask_structured
from interview_simulator.prompts import templates as P
from interview_simulator.tools.document_loader import load_document

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to call real APIs")


def test_parse_sample_resume_and_jd():
    if [k for k in get_settings().missing_keys() if k != "TAVILY_API_KEY"]:
        pytest.skip("LLM API key missing")
    resume = ask_structured("evaluator", ResumeProfile, load_document("data/samples/resume.md"), system=P.PARSE_RESUME)
    assert "Java" in resume.skills and resume.years_experience >= 3
    job = ask_structured("evaluator", JobProfile, load_document("data/samples/job_description.md"), system=P.PARSE_JD)
    assert job.seniority == "senior" and "Kafka" in " ".join(job.tech_stack)
