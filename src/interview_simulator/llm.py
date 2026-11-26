"""Provider-agnostic LLM factory (Claude / Gemini) with retries and fallbacks.

Nodes never import a provider SDK directly; they call `structured()` or `chat()`.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from interview_simulator.config import Provider, Role, get_settings

TEMPERATURE: dict[Role, float] = {"interviewer": 0.7, "evaluator": 0.2, "feedback": 0.4}


def _make_model(model: str, provider: Provider, temperature: float) -> BaseChatModel:
    s = get_settings()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            api_key=s.anthropic_api_key,
            temperature=temperature,
            max_tokens=8192,
            timeout=s.llm_timeout_sec,
            max_retries=s.llm_max_retries,
        )
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=s.google_api_key,
        temperature=temperature,
        timeout=s.llm_timeout_sec,
        max_retries=s.llm_max_retries,
    )


def _models(role: Role, provider: Provider | None) -> list[BaseChatModel]:
    s = get_settings()
    provider = provider or s.llm_provider
    names = [s.model_for(role, provider)]
    names += [m for m in s.fallback_models_for(provider) if m not in names]
    return [_make_model(n, provider, TEMPERATURE[role]) for n in names]


def chat(role: Role, provider: Provider | None = None) -> Runnable:
    """Plain chat model (returns AIMessage) with fallbacks."""
    primary, *rest = _models(role, provider)
    return primary.with_fallbacks(rest) if rest else primary


def structured[T: BaseModel](role: Role, schema: type[T], provider: Provider | None = None) -> Runnable:
    """Model that returns an instance of `schema`, with fallbacks.

    `with_structured_output` must be applied to each model *before* `with_fallbacks`.
    """
    primary, *rest = [m.with_structured_output(schema) for m in _models(role, provider)]
    return primary.with_fallbacks(rest) if rest else primary


def model_name(role: Role, provider: Provider | None = None) -> str:
    s = get_settings()
    return s.model_for(role, provider or s.llm_provider)
