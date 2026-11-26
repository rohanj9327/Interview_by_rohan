"""Application settings loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

Provider = Literal["anthropic", "google_genai"]
Role = Literal["interviewer", "evaluator", "feedback"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider + keys
    llm_provider: Provider = "anthropic"
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    tavily_api_key: str | None = None

    # Role -> model mapping
    anthropic_interviewer_model: str = "claude-sonnet-5-5"
    anthropic_evaluator_model: str = "claude-sonnet-5-5"
    anthropic_feedback_model: str = "claude-opus-5-5"
    gemini_interviewer_model: str | None = None
    gemini_evaluator_model: str | None = None
    gemini_feedback_model: str | None = None
    # Comma-separated same-provider fallbacks, tried when the primary model fails (503/429)
    anthropic_fallback_models: str = ""
    gemini_fallback_models: str = "gemini-3.1-flash-lite"
    llm_timeout_sec: int = 90
    llm_max_retries: int = 2

    # Voice
    stt_engine: Literal["mlx_whisper", "faster_whisper", "gemini"] = "mlx_whisper"
    whisper_model: str = "mlx-community/whisper-small.en-mlx"
    tts_voice: str = "en-US-GuyNeural"

    # Interview defaults
    num_questions: int = Field(10, ge=1, le=30)
    answer_time_limit_sec: int = 180
    coding_time_limit_sec: int = 1200
    feedback_mode: Literal["practice", "realistic"] = "practice"
    max_follow_ups: int = 2

    # Coding
    code_run_timeout_sec: int = 5
    code_memory_limit_mb: int = 512

    # Storage
    data_dir: Path = Path("data/runtime")

    @property
    def data_path(self) -> Path:
        path = self.data_dir if self.data_dir.is_absolute() else PROJECT_ROOT / self.data_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def model_for(self, role: Role, provider: Provider | None = None) -> str:
        provider = provider or self.llm_provider
        prefix = "anthropic" if provider == "anthropic" else "gemini"
        model = getattr(self, f"{prefix}_{role}_model")
        if not model:
            raise ValueError(
                f"No model configured for provider={provider!r}, role={role!r}. "
                f"Set {prefix.upper()}_{role.upper()}_MODEL in .env."
            )
        return model

    def fallback_models_for(self, provider: Provider | None = None) -> list[str]:
        provider = provider or self.llm_provider
        raw = self.anthropic_fallback_models if provider == "anthropic" else self.gemini_fallback_models
        return [m.strip() for m in raw.split(",") if m.strip()]

    def missing_keys(self) -> list[str]:
        """API keys required by the current configuration that are not set."""
        missing = []
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if (self.llm_provider == "google_genai" or self.stt_engine == "gemini") and not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        return missing


@lru_cache
def get_settings() -> Settings:
    return Settings()
