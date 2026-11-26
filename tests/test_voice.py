import io

import numpy as np
import pytest
import soundfile as sf

from interview_simulator.voice import quality
from interview_simulator.voice.audio import TARGET_SR, decode_audio_bytes
from interview_simulator.voice.delivery import compute_metrics, count_fillers, pace_score
from interview_simulator.voice.stt import Transcript, Word


def test_decode_resamples_stereo_44k_to_mono_16k():
    sr = 44_100
    t = np.linspace(0, 1, sr, endpoint=False)
    stereo = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    audio = decode_audio_bytes(buf.getvalue())
    assert audio.ndim == 1 and abs(len(audio) - TARGET_SR) <= 1 and audio.dtype == np.float32


def test_count_fillers():
    c = count_fillers("Um, so I, uh, basically built it. You know, umm, it was kind of hard.")
    assert c["um"] == 2 and c["uh"] == 1 and c["basically"] == 1
    assert c["you know"] == 1 and c["kind of"] == 1


def test_metrics_from_word_timestamps():
    # 30 words over 12 s (=150 wpm) with one 3 s pause
    words, t = [], 0.0
    for i in range(30):
        if i == 15:
            t += 3.0
        words.append(Word(word="um" if i in (3, 20) else "word", start=t, end=t + 0.3))
        t += 0.3
    tr = Transcript(text=" ".join(w.word for w in words), words=words, duration_sec=t + 0.5, engine="test")
    m = compute_metrics(tr)
    assert m.word_count == 30 and m.filler_count == 2 and m.long_pauses == 1
    assert 140 < m.wpm < 160 and m.pace_score == 5


def test_pace_score_bands():
    assert pace_score(145) == 5 and pace_score(120) == 4 and pace_score(200) == 2 and pace_score(60) == 1


# ---- quality guard (prevents Whisper hallucinations on silent audio) --------------------




def test_silent_audio_is_rejected():
    with pytest.raises(quality.NoSpeechError, match="silent"):
        quality.prepare(np.zeros(TARGET_SR * 3, np.float32))


def test_short_audio_is_rejected():
    with pytest.raises(quality.NoSpeechError, match="too short"):
        quality.prepare(np.ones(TARGET_SR // 4, np.float32) * 0.5)


def test_quiet_speechlike_audio_is_normalised_and_trimmed():
    t = np.linspace(0, 1, TARGET_SR, endpoint=False)
    tone = (0.02 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio = np.concatenate([np.zeros(TARGET_SR * 2, np.float32), tone, np.zeros(TARGET_SR * 2, np.float32)])
    out, stats = quality.prepare(audio)
    assert stats.speech_sec >= 0.9
    assert np.max(np.abs(out)) == pytest.approx(quality.TARGET_PEAK, rel=0.01)
    assert len(out) < len(audio) / 2  # leading/trailing silence trimmed


def test_collapse_repetition():
    loop = " ".join(["You can find it on the web."] * 20)
    assert quality.collapse_repetition(loop) == ("You can find it on the web.", True)
    assert quality.collapse_repetition("A. B. B.") == ("A. B.", False)
