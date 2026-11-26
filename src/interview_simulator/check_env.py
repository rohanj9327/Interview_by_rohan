"""Verify the local environment: Python packages, Java, optional ffmpeg, API keys.

Run:  uv run python -m interview_simulator.check_env
"""

import importlib
import platform
import shutil
import subprocess
import sys

from interview_simulator.config import get_settings

REQUIRED_MODULES = [
    "langgraph", "langchain", "langchain_anthropic", "langchain_google_genai",
    "tavily", "pymupdf", "docx", "trafilatura", "streamlit", "streamlit_ace",
    "edge_tts", "numpy", "soundfile", "scipy",
]


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _warn(msg: str) -> None:
    print(f"  [warn] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def main() -> int:
    failures = 0
    print(f"Python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")

    print("\nPython packages")
    modules = REQUIRED_MODULES + (["mlx_whisper"] if platform.machine() == "arm64" else [])
    for name in modules:
        try:
            importlib.import_module(name)
            _ok(name)
        except Exception as exc:  # noqa: BLE001
            _fail(f"{name}: {exc}")
            failures += 1

    print("\nSystem tools")
    for tool in ("java", "javac"):
        if shutil.which(tool):
            out = subprocess.run([tool, "-version"], capture_output=True, text=True, check=False)
            _ok(f"{tool}: {(out.stderr or out.stdout).splitlines()[0]}")
        else:
            _fail(f"{tool} not found (needed for Java coding questions)")
            failures += 1
    if shutil.which("ffmpeg"):
        _ok("ffmpeg (optional)")
    else:
        _warn("ffmpeg not found (optional; WAV audio is decoded in Python)")

    print("\nConfiguration (.env)")
    settings = get_settings()
    _ok(f"LLM_PROVIDER={settings.llm_provider}, STT_ENGINE={settings.stt_engine}")
    missing = settings.missing_keys()
    if missing:
        _warn("missing keys: " + ", ".join(missing) + " (add them to .env)")
    else:
        _ok("all required API keys present")
    for role in ("interviewer", "evaluator", "feedback"):
        try:
            _ok(f"{role} model: {settings.model_for(role)}")
        except ValueError as exc:
            _warn(str(exc))

    print("\nResult:", "OK" if failures == 0 else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
