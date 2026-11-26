"""Audio quality checks and transcript sanity checks.

Whisper "hallucinates" on silent or very quiet audio, typically looping one sentence
("You can find it on the web. You can find it on the web. ..."). We guard against that by
(1) refusing to transcribe audio with no audible speech, (2) normalising quiet audio and
trimming silence, and (3) detecting looping transcripts afterwards.
"""

from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel

from interview_simulator.voice.audio import TARGET_SR

FRAME_SEC = 0.03
SILENT_PEAK = 0.005  # below ~ -46 dBFS peak the recording is effectively silent
SPEECH_FRAME_RMS = 0.02  # after normalisation
MIN_SPEECH_SEC = 0.6
TARGET_PEAK = 0.9


class NoSpeechError(Exception):
    """The recording contains no audible speech."""


class AudioStats(BaseModel):
    duration_sec: float
    peak: float
    rms_dbfs: float
    speech_sec: float


def _frame_rms(audio: np.ndarray) -> np.ndarray:
    n = int(FRAME_SEC * TARGET_SR)
    usable = len(audio) // n * n
    if usable == 0:
        return np.array([float(np.sqrt(np.mean(audio**2)))]) if len(audio) else np.array([0.0])
    frames = audio[:usable].reshape(-1, n)
    return np.sqrt(np.mean(frames**2, axis=1))


def analyze(audio: np.ndarray) -> AudioStats:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    scale = TARGET_PEAK / peak if peak > 0 else 1.0
    speech_frames = int(np.sum(_frame_rms(audio * scale) > SPEECH_FRAME_RMS))
    return AudioStats(
        duration_sec=round(len(audio) / TARGET_SR, 2),
        peak=round(peak, 4),
        rms_dbfs=round(20 * np.log10(rms) if rms > 0 else -120.0, 1),
        speech_sec=round(speech_frames * FRAME_SEC, 2),
    )


def prepare(audio: np.ndarray) -> tuple[np.ndarray, AudioStats]:
    """Validate, normalise and trim audio before transcription. Raises NoSpeechError."""
    stats = analyze(audio)
    if stats.duration_sec < 0.5:
        raise NoSpeechError("The recording is too short.")
    if stats.peak < SILENT_PEAK:
        raise NoSpeechError(
            "The recording is silent. Your microphone may be muted, the wrong input device may be "
            "selected, or the browser may not have microphone permission."
        )
    if stats.speech_sec < MIN_SPEECH_SEC:
        raise NoSpeechError("No speech was detected in the recording. Please speak closer to the microphone.")

    audio = (audio * (TARGET_PEAK / stats.peak)).astype(np.float32)
    # Trim leading/trailing silence (keep 0.3 s padding)
    rms = _frame_rms(audio)
    voiced = np.where(rms > SPEECH_FRAME_RMS)[0]
    if len(voiced):
        n = int(FRAME_SEC * TARGET_SR)
        pad = int(0.3 * TARGET_SR)
        start = max(0, voiced[0] * n - pad)
        end = min(len(audio), (voiced[-1] + 1) * n + pad)
        audio = audio[start:end]
    return audio, stats


_SENTENCE = re.compile(r"[^.!?]+[.!?]*")


def collapse_repetition(text: str, min_repeats: int = 3) -> tuple[str, bool]:
    """Collapse runs of the same sentence repeated >= min_repeats times.

    Returns (cleaned_text, looped) where looped=True means the transcript was likely a
    hallucination loop and should be reviewed/re-recorded.
    """
    sentences = [s.strip() for s in _SENTENCE.findall(text) if s.strip()]
    if not sentences:
        return text.strip(), False
    out: list[str] = []
    looped = False
    run = 1
    for i, sentence in enumerate(sentences):
        norm = re.sub(r"\W+", " ", sentence.lower()).strip()
        prev = re.sub(r"\W+", " ", sentences[i - 1].lower()).strip() if i else None
        if norm and norm == prev:
            run += 1
            if run >= min_repeats:
                looped = True
            continue
        run = 1
        out.append(sentence)
    return " ".join(out), looped
