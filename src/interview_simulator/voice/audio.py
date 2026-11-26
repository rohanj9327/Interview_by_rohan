"""Decode audio to 16 kHz mono float32 numpy arrays without ffmpeg."""

from __future__ import annotations

import io
from math import gcd
from pathlib import Path

import numpy as np

TARGET_SR = 16_000


def _to_mono_16k(data: np.ndarray, sr: int) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != TARGET_SR:
        from scipy.signal import resample_poly

        g = gcd(sr, TARGET_SR)
        data = resample_poly(data, TARGET_SR // g, sr // g).astype(np.float32)
    return data


def decode_audio_bytes(data: bytes) -> np.ndarray:
    """Decode WAV/FLAC/OGG/AIFF bytes (e.g. from st.audio_input) to 16 kHz mono."""
    import soundfile as sf

    samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return _to_mono_16k(samples, sr)


def load_audio(path: str | Path) -> np.ndarray:
    return decode_audio_bytes(Path(path).read_bytes())


def duration_sec(audio: np.ndarray) -> float:
    return len(audio) / TARGET_SR
