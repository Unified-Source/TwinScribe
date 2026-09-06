"""Test-only stand-ins for the engine record types.

Frozen dataclasses equivalent to Word, Segment and Transcript as specified for the engines
package, used by the review tests only while that package is absent. Tests import the real
types first and fall back to these.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    prob: float | None = None


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    words: tuple[Word, ...]
    quality: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class Transcript:
    engine: str
    model: str
    preset: str
    segments: tuple[Segment, ...]
    audio_s: float
    load_s: float = 0.0
    transcribe_s: float = 0.0
    versions: dict[str, str] = field(default_factory=dict)
    extras: dict[str, float | int | None] = field(default_factory=dict)

    @property
    def words(self) -> list[Word]:
        flattened = [word for segment in self.segments for word in segment.words]
        return sorted(flattened, key=lambda word: (word.start, word.end))

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())
