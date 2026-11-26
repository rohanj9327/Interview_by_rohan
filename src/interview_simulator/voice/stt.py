"""Speech-to-text (English): mlx-whisper (default, Apple Silicon), faster-whisper, or Gemini."""

from __future__ import annotations

import io
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from interview_simulator.config import get_settings
from interview_simulator.voice.audio import TARGET_SR, decode_audio_bytes, duration_sec, load_audio

log = logging.getLogger(__name__)

# Whisper tends to drop disfluencies. Priming it with a filler-heavy prompt makes it keep
# "um", "uh", etc., which we need for delivery feedback.
FILLER_PROMPT = "Umm, let me think, uh... so, like, hmm. Okay, I mean, you know, here's what I, uh, did."


class Word(BaseModel):
    word: str
    start: float
    end: float


class Transcript(BaseModel):
    text: str
    words: list[Word] = Field(default_factory=list)
    duration_sec: float
    engine: str
    warnings: list[str] = Field(default_factory=list)


def _prompt(vocabulary: list[str] | None) -> str:
    if not vocabulary:
        return FILLER_PROMPT
    return FILLER_PROMPT + " Technical terms: " + ", ".join(vocabulary[:40]) + "."


def _mlx(audio: np.ndarray, vocabulary: list[str] | None = None) -> Transcript:
    import mlx_whisper

    s = get_settings()
    out = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=s.whisper_model,
        language="en",
        word_timestamps=True,
        initial_prompt=_prompt(vocabulary),
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
        no_speech_threshold=0.6,
        verbose=None,
    )
    words = [
        Word(word=w["word"].strip(), start=float(w["start"]), end=float(w["end"]))
        for seg in out.get("segments", [])
        for w in seg.get("words", [])
        if w.get("word", "").strip()
    ]
    return Transcript(text=out.get("text", "").strip(), words=words,
                      duration_sec=duration_sec(audio), engine="mlx_whisper")


@lru_cache(maxsize=1)
def _faster_model(name: str):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device="cpu", compute_type="int8")


def _faster(audio: np.ndarray, vocabulary: list[str] | None = None) -> Transcript:
    name = get_settings().whisper_model
    if "/" in name:  # an mlx repo id was configured; fall back to a faster-whisper name
        name = "small.en"
    segments, _ = _faster_model(name).transcribe(
        audio, language="en", word_timestamps=True, initial_prompt=_prompt(vocabulary),
        condition_on_previous_text=False,
    )
    words, texts = [], []
    for seg in segments:
        texts.append(seg.text)
        words += [Word(word=w.word.strip(), start=w.start, end=w.end) for w in (seg.words or []) if w.word.strip()]
    return Transcript(text="".join(texts).strip(), words=words,
                      duration_sec=duration_sec(audio), engine="faster_whisper")


def _gemini(audio: np.ndarray) -> Transcript:
    """Transcribe with Gemini. No word timestamps -> delivery metrics are approximate."""
    import base64

    import soundfile as sf
    from langchain_core.messages import HumanMessage

    from interview_simulator import llm

    buf = io.BytesIO()
    sf.write(buf, audio, TARGET_SR, format="WAV")
    msg = HumanMessage(content=[
        {"type": "text", "text": "Transcribe this English speech verbatim, keeping filler words "
                                 "(um, uh, like). Return only the transcript."},
        {"type": "media", "mime_type": "audio/wav", "data": base64.b64encode(buf.getvalue()).decode()},
    ])
    text = llm.chat("interviewer", provider="google_genai").invoke([msg]).content
    if isinstance(text, list):
        text = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
    return Transcript(text=str(text).strip(), words=[], duration_sec=duration_sec(audio), engine="gemini")


def transcribe(audio: np.ndarray | bytes | str | Path, vocabulary: list[str] | None = None) -> Transcript:
    """Transcribe English speech. Accepts a 16 kHz array, audio bytes, or a file path.

    `vocabulary`: technical terms (from the JD/resume) that bias Whisper toward correct spelling.

    Raises `quality.NoSpeechError` if the audio has no audible speech (instead of letting
    Whisper hallucinate). Looping transcripts are collapsed and flagged in `warnings`.
    """
    from interview_simulator.voice import quality

    if isinstance(audio, bytes):
        audio = decode_audio_bytes(audio)
    elif isinstance(audio, (str, Path)):
        audio = load_audio(audio)
    original_duration = duration_sec(audio)
    _save_debug(audio)
    audio, stats = quality.prepare(audio)
    log.info("Audio: %s", stats.model_dump())

    engine = get_settings().stt_engine
    if engine == "gemini":
        result = _gemini(audio)
    elif engine == "faster_whisper":
        result = _faster(audio, vocabulary)
    else:
        try:
            result = _mlx(audio, vocabulary)
        except ImportError:
            log.warning("mlx-whisper unavailable; falling back to faster-whisper")
            result = _faster(audio, vocabulary)

    result.duration_sec = original_duration
    text, looped = quality.collapse_repetition(result.text)
    if looped:
        result.warnings.append(
            "The transcript repeated itself, which usually means the audio was unclear. "
            "Please check the text or record again."
        )
        result.text = text
        result.words = _dedupe_words(result.words)
    if not result.text.strip():
        raise quality.NoSpeechError("No words were recognised. Please record again, closer to the microphone.")
    return result


def _dedupe_words(words: list[Word]) -> list[Word]:
    """Drop word timestamps after a loop is detected (they no longer match the text)."""
    return []


def _save_debug(audio: np.ndarray) -> None:
    """Keep the last recording for troubleshooting (data/runtime/debug/last_recording.wav)."""
    try:
        import soundfile as sf

        folder = get_settings().data_path / "debug"
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / "last_recording.wav", audio, TARGET_SR)
    except Exception as exc:  # noqa: BLE001 - debugging aid only
        log.debug("Could not save debug recording: %s", exc)
