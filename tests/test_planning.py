from interview_simulator.models import InterviewConfig, JobProfile
from interview_simulator.nodes.planning import category_counts, num_coding
from interview_simulator.nodes.research import build_queries


def test_category_counts_sum_to_total():
    for total in range(3, 16):
        for coding in (1, 2):
            counts = category_counts(total, coding, has_company=True)
            assert sum(counts.values()) == total - coding
            assert "coding" not in counts


def test_default_mix_for_10_questions():
    counts = category_counts(10, 2, has_company=True)
    assert counts["resume"] >= 2 and counts["behavioral"] >= 2 and counts["system_design"] >= 1


def test_num_coding_by_seniority():
    cfg = InterviewConfig()
    assert num_coding(cfg, JobProfile(title="SWE", seniority="junior")) == 1
    assert num_coding(cfg, JobProfile(title="SWE", seniority="senior")) == 2
    assert num_coding(InterviewConfig(num_coding_questions=0), JobProfile(title="SWE")) == 0


def test_research_queries_use_company_when_known():
    q = build_queries(JobProfile(title="Backend Engineer", company="FinFlow", seniority="senior", tech_stack=["Java", "Kafka"]))
    assert any("FinFlow" in x for x in q) and len(q) <= 4
    q = build_queries(JobProfile(title="Backend Engineer", seniority="mid"))
    assert all("FinFlow" not in x for x in q)
