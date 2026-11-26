"""Text-to-speech for interviewer questions: edge-tts (default) with macOS `say` fallback.

Audio is cached by (voice, text) so replays are instant.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import subprocess
from pathlib import Path

from interview_simulator.config import get_settings

log = logging.getLogger(__name__)

VOICES = {
    "en-US-GuyNeural": "US English, male",
    "en-US-JennyNeural": "US English, female",
    "en-US-AriaNeural": "US English, female",
    "en-GB-RyanNeural": "British English, male",
    "en-GB-SoniaNeural": "British English, female",
    "en-IN-PrabhatNeural": "Indian English, male",
    "en-IN-NeerjaNeural": "Indian English, female",
}


def _cache_dir() -> Path:
    path = get_settings().data_path / "tts"
    path.mkdir(parents=True, exist_ok=True)
    return path


async def _edge(text: str, voice: str, out: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice).save(str(out))


def _run_async(coro) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    # Already inside an event loop (e.g. some UI frameworks): run in a worker thread
    import threading

    err: list[BaseException] = []

    def runner():
        try:
            asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)

    t = threading.Thread(target=runner)
    t.start()
    t.join()
    if err:
        raise err[0]


def synthesize(text: str, voice: str | None = None) -> Path:
    """Return the path to an audio file (MP3 from edge-tts, or WAV from macOS say)."""
    voice = voice or get_settings().tts_voice
    key = hashlib.sha256(f"{voice}|{text}".encode()).hexdigest()[:24]
    mp3 = _cache_dir() / f"{key}.mp3"
    wav = _cache_dir() / f"{key}.wav"
    for cached in (mp3, wav):
        if cached.exists() and cached.stat().st_size > 0:
            return cached
    try:
        _run_async(_edge(text, voice, mp3))
        if mp3.exists() and mp3.stat().st_size > 0:
            return mp3
    except Exception as exc:  # noqa: BLE001
        log.warning("edge-tts failed (%s); falling back to macOS say", exc)
        mp3.unlink(missing_ok=True)
    if shutil.which("say"):
        subprocess.run(["say", "--file-format=WAVE", "--data-format=LEI16@22050", "-o", str(wav), text],
                       check=True, timeout=60)
        return wav
    raise RuntimeError("No TTS engine available (edge-tts failed and macOS `say` not found)")


def play(path: Path) -> None:
    """Play audio in the terminal (macOS afplay). Used by the CLI."""
    if shutil.which("afplay"):
        subprocess.run(["afplay", str(path)], check=False)
