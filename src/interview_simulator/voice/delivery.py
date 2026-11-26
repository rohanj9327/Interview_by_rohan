"""Deterministic delivery metrics from word timestamps: pace, filler words, pauses."""

from __future__ import annotations

import re
from collections import Counter
from itertools import pairwise

from interview_simulator.models import DeliveryMetrics
from interview_simulator.voice.stt import Transcript

SINGLE_FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "mm", "basically", "literally"}
PHRASE_FILLERS = ["you know", "i mean", "kind of", "sort of"]
LONG_PAUSE_SEC = 2.0
IDEAL_WPM = (130, 160)


def _normalise(word: str) -> str:
    return re.sub(r"[^a-z']", "", word.lower())


def count_fillers(text: str) -> Counter:
    tokens = [_normalise(t) for t in text.split()]
    tokens = [t for t in tokens if t]
    counts = Counter(t for t in tokens if t in SINGLE_FILLERS)
    joined = " " + " ".join(tokens) + " "
    for phrase in PHRASE_FILLERS:
        n = joined.count(f" {phrase} ")
        if n:
            counts[phrase] += n
    # merge spelling variants
    for variant, base in (("umm", "um"), ("uhh", "uh"), ("uhm", "um")):
        if variant in counts:
            counts[base] += counts.pop(variant)
    return counts


def pace_score(wpm: float) -> int:
    lo, hi = IDEAL_WPM
    if lo <= wpm <= hi:
        return 5
    off = (lo - wpm) if wpm < lo else (wpm - hi)
    if off <= 15:
        return 4
    if off <= 35:
        return 3
    if off <= 60:
        return 2
    return 1


def fluency_score(fillers_per_min: float, long_pauses_per_min: float) -> int:
    penalty = fillers_per_min / 2 + long_pauses_per_min
    if penalty < 1:
        return 5
    if penalty < 2.5:
        return 4
    if penalty < 4:
        return 3
    if penalty < 6:
        return 2
    return 1


def compute_metrics(transcript: Transcript) -> DeliveryMetrics:
    words = transcript.words
    word_count = len(words) if words else len(transcript.text.split())
    if words:
        speaking = max(words[-1].end - words[0].start, 0.1)
        gaps = [b.start - a.end for a, b in pairwise(words)]
    else:  # no timestamps (e.g. Gemini STT): approximate with total duration
        speaking = max(transcript.duration_sec, 0.1)
        gaps = []
    long_gaps = [g for g in gaps if g >= LONG_PAUSE_SEC]
    minutes = speaking / 60
    fillers = count_fillers(transcript.text)
    filler_total = sum(fillers.values())
    wpm = word_count / minutes if minutes > 0 else 0.0
    fpm = filler_total / minutes if minutes > 0 else 0.0
    return DeliveryMetrics(
        duration_sec=round(transcript.duration_sec, 1),
        speaking_sec=round(speaking, 1),
        word_count=word_count,
        wpm=round(wpm, 1),
        filler_count=filler_total,
        filler_words=dict(fillers),
        fillers_per_min=round(fpm, 1),
        long_pauses=len(long_gaps),
        longest_pause_sec=round(max(gaps, default=0.0), 1),
        pace_score=pace_score(wpm),
        fluency_score=fluency_score(fpm, len(long_gaps) / minutes if minutes > 0 else 0),
    )
