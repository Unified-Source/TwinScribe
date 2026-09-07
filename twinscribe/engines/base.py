"""Records shared by every engine: words, segments, transcripts and diarization results, plus
the timing and thread-count helpers the engines have in common.

The records are frozen dataclasses so that a transcript, once produced, is an immutable
account of what an engine returned together with how long each stage took.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

MAX_DEFAULT_THREADS = 8


@dataclass(frozen=True)
class Word:
    """One word with its time span in seconds and the engine's confidence, if it has one."""

    text: str
    start: float
    end: float
    prob: float | None = None


@dataclass(frozen=True)
class Segment:
    """A contiguous stretch of transcript with its words and the engine's quality figures."""

    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()
    quality: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class Transcript:
    """What one engine produced for one file, with the timings recorded beside it.

    load_s covers model construction; transcribe_s covers the whole decode including full
    consumption of any lazy generator. audio_s is the duration the engine reported or the
    length of the waveform fed to it. extras carries engine-specific numeric bookkeeping;
    settings carries what the engine was asked to use (device, compute type, provider).
    """

    engine: str
    model: str
    preset: str
    segments: tuple[Segment, ...]
    audio_s: float
    load_s: float
    transcribe_s: float
    versions: dict[str, str] = field(default_factory=dict)
    extras: dict[str, float | int | None] = field(default_factory=dict)
    settings: dict[str, object] = field(default_factory=dict)

    @property
    def words(self) -> list[Word]:
        """Every word of every segment, ordered by start time (stable for equal starts)."""
        flat = [word for segment in self.segments for word in segment.words]
        return sorted(flat, key=lambda word: (word.start, word.end))

    @property
    def text(self) -> str:
        """Segment texts joined by single spaces; empty segments contribute nothing."""
        parts = [segment.text.strip() for segment in self.segments]
        return " ".join(part for part in parts if part)


@dataclass(frozen=True)
class SpeakerTurn:
    """One labelled stretch of speech from the diarizer."""

    start: float
    end: float
    label: str

    @property
    def duration(self) -> float:
        """Length of the turn in seconds, never negative."""
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class Diarization:
    """Speaker turns for one file, seconds per label, timings and library versions."""

    segments: tuple[SpeakerTurn, ...]
    seconds_per_label: dict[str, float]
    audio_s: float
    load_s: float
    diarize_s: float
    versions: dict[str, str] = field(default_factory=dict)
    settings: dict[str, float | int | None] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        """Labels in order of first appearance."""
        seen: list[str] = []
        for turn in self.segments:
            if turn.label not in seen:
                seen.append(turn.label)
        return seen


class Stopwatch:
    """Wall-clock timer used as a context manager; seconds is valid once the block exits."""

    def __init__(self) -> None:
        self._start: float | None = None
        self.seconds: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._start is not None:
            self.seconds = time.perf_counter() - self._start
        self._start = None


def default_threads(threads: int | None = None) -> int:
    """Thread count for an engine: the argument when given, else min(8, core count).

    Eight is the ceiling because the engines stop scaling well past that on laptop parts and
    leaving cores free keeps the machine usable while a batch runs.
    """
    if threads is None:
        return max(1, min(MAX_DEFAULT_THREADS, os.cpu_count() or 1))
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")
    return int(threads)


def elapsed_since(start: float) -> float:
    """Seconds elapsed since a time.perf_counter() reading."""
    return time.perf_counter() - start
