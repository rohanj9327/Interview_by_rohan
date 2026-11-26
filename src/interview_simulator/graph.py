"""Build the LangGraph interview graph."""

from __future__ import annotations

import sqlite3

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from interview_simulator.config import get_settings
from interview_simulator.nodes.analysis import gap_analysis
from interview_simulator.nodes.coding import generate_coding_problems
from interview_simulator.nodes.feedback import final_feedback, save_session
from interview_simulator.nodes.hints import give_hint
from interview_simulator.nodes.interview import (
    ask_question,
    await_answer,
    evaluate_answer,
    next_question,
    quick_tip,
    route_after_await,
    route_after_evaluate,
    route_after_next,
)
from interview_simulator.nodes.parsing import parse_jd, parse_resume
from interview_simulator.nodes.planning import plan_interview
from interview_simulator.nodes.research import web_research
from interview_simulator.state import InterviewState

# Human-friendly labels for progress display
NODE_LABELS = {
    "parse_resume": "Reading your resume",
    "parse_jd": "Reading the job description",
    "gap_analysis": "Comparing your profile with the role",
    "web_research": "Researching common interview questions",
    "plan_interview": "Planning your interview",
    "generate_coding_problems": "Preparing and testing coding problems",
    "evaluate_answer": "Evaluating your answer",
    "give_hint": "Thinking of a hint",
    "final_feedback": "Writing your feedback report",
}


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    g = StateGraph(InterviewState)
    g.add_node("parse_resume", parse_resume)
    g.add_node("parse_jd", parse_jd)
    g.add_node("gap_analysis", gap_analysis)
    g.add_node("web_research", web_research)
    g.add_node("plan_interview", plan_interview)
    g.add_node("generate_coding_problems", generate_coding_problems)
    g.add_node("ask_question", ask_question)
    g.add_node("await_answer", await_answer)
    g.add_node("give_hint", give_hint)
    g.add_node("evaluate_answer", evaluate_answer)
    g.add_node("quick_tip", quick_tip)
    g.add_node("next_question", next_question)
    g.add_node("final_feedback", final_feedback)
    g.add_node("save_session", save_session)

    # Preparation: resume + JD parsed in parallel, then joined
    g.add_edge(START, "parse_resume")
    g.add_edge(START, "parse_jd")
    g.add_edge(["parse_resume", "parse_jd"], "gap_analysis")
    g.add_edge("gap_analysis", "web_research")
    g.add_edge("web_research", "plan_interview")
    g.add_edge("plan_interview", "generate_coding_problems")
    g.add_edge("generate_coding_problems", "ask_question")

    # Interview loop
    g.add_edge("ask_question", "await_answer")
    g.add_conditional_edges("await_answer", route_after_await,
                            ["give_hint", "evaluate_answer", "final_feedback"])
    g.add_edge("give_hint", "await_answer")
    g.add_conditional_edges("evaluate_answer", route_after_evaluate,
                            ["ask_question", "quick_tip", "next_question"])
    g.add_edge("quick_tip", "next_question")
    g.add_conditional_edges("next_question", route_after_next, ["ask_question", "final_feedback"])

    # Wrap-up
    g.add_edge("final_feedback", "save_session")
    g.add_edge("save_session", END)
    return g.compile(checkpointer=checkpointer)


def serializer():
    """Checkpoint serializer that explicitly allows our Pydantic state models."""
    import inspect

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from pydantic import BaseModel

    from interview_simulator import models

    allowed = [
        ("interview_simulator.models", name)
        for name, obj in inspect.getmembers(models, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == models.__name__
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def sqlite_checkpointer():
    """Persistent checkpointer so interviews can be paused and resumed."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = get_settings().data_path / "checkpoints.db"
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn, serde=serializer())


def memory_checkpointer():
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver(serde=serializer())
