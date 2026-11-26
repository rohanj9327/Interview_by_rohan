"""Small helpers shared by nodes."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from interview_simulator import llm
from interview_simulator.config import Role

log = logging.getLogger("interview_simulator")


def dump(obj) -> str:
    """Compact JSON for prompts."""
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(exclude_none=True)
    if isinstance(obj, list):
        return json.dumps([o.model_dump(exclude_none=True) if isinstance(o, BaseModel) else o for o in obj])
    return json.dumps(obj, default=str)


def ask_structured(role: Role, schema: type[BaseModel], prompt: str, system: str | None = None):
    messages = ([SystemMessage(system)] if system else []) + [HumanMessage(prompt)]
    result = llm.structured(role, schema).invoke(messages)
    if result is None:
        raise RuntimeError(f"LLM returned no structured output for {schema.__name__}")
    return result
