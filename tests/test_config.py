import pytest

from interview_simulator.config import Settings


def make(**kw) -> Settings:
    # _env_file=None: ignore the developer's real .env
    return Settings(_env_file=None, **kw)


def test_defaults():
    s = make()
    assert s.llm_provider == "anthropic"
    assert s.num_questions == 10
    assert s.answer_time_limit_sec == 180
    assert s.tts_voice == "en-US-GuyNeural"


def test_model_for_anthropic():
    s = make()
    assert s.model_for("feedback") == "claude-opus-5-5"
    assert s.model_for("interviewer") == "claude-sonnet-5-5"


def test_model_for_gemini_requires_config():
    s = make(llm_provider="google_genai")
    with pytest.raises(ValueError, match="GEMINI_INTERVIEWER_MODEL"):
        s.model_for("interviewer")
    s = make(llm_provider="google_genai", gemini_interviewer_model="gemini-x-flash")
    assert s.model_for("interviewer") == "gemini-x-flash"


def test_missing_keys():
    s = make(anthropic_api_key=None, tavily_api_key=None)
    assert set(s.missing_keys()) == {"ANTHROPIC_API_KEY", "TAVILY_API_KEY"}
    s = make(llm_provider="google_genai", google_api_key="k", tavily_api_key="t")
    assert s.missing_keys() == []


def test_data_path_created(tmp_path):
    s = make(data_dir=tmp_path / "runtime")
    assert s.data_path.exists()
