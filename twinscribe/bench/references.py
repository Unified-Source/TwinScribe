"""References built from corpus annotations: which words were said, by whom, and when the
annotation knows it. Each builder takes the annotation as text or parsed JSON and returns a
Reference; the audio arithmetic the items need (channel mixing, window cutting) is here too so
that a test can exercise it on arrays.
"""

from __future__ import annotations

import re
import wave
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from twinscribe.engines.base import Word

WORD_TIMES_ANNOTATED = "annotated"
WORD_TIMES_SPREAD = "spread"
WORD_TIMES_NONE = "none"

HVB_ROLES: tuple[str, ...] = ("agent", "caller")
LIBRISPEECH_SPEAKER = "reader"

_PCM_MIN = -32768
_PCM_MAX = 32767


@dataclass(frozen=True)
class RefWord:
    """One reference word with its speaker and, when the annotation has them, its times."""

    text: str
    speaker: str
    start: float | None = None
    end: float | None = None


@dataclass(frozen=True)
class RefSegment:
    """One stretch of one speaker's speech with the text said in it."""

    start: float
    end: float
    speaker: str
    text: str


@dataclass(frozen=True)
class Reference:
    """What an item's annotation says: speakers, segments, words, and how good the word times
    are: annotated by the corpus, spread evenly inside each segment, or absent."""

    item: str
    corpus: str
    audio_s: float
    speakers: tuple[str, ...]
    segments: tuple[RefSegment, ...]
    words: tuple[RefWord, ...]
    word_times: str
    notes: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

    def timed_words(self) -> list[Word]:
        """The reference words as engine words, for the review evaluation; empty without times."""
        if self.word_times == WORD_TIMES_NONE:
            return []
        return [
            Word(text=w.text, start=float(w.start), end=float(w.end), prob=None)
            for w in self.words
            if w.start is not None and w.end is not None
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "corpus": self.corpus,
            "audio_s": self.audio_s,
            "speakers": list(self.speakers),
            "segments": [{"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text} for s in self.segments],
            "words": [{"text": w.text, "speaker": w.speaker, "start": w.start, "end": w.end} for w in self.words],
            "word_times": self.word_times,
            "notes": list(self.notes),
        }


def spread_words(segment: RefSegment) -> list[RefWord]:
    """The segment's words with times spread evenly inside its span."""
    tokens = segment.text.split()
    if not tokens:
        return []
    step = (segment.end - segment.start) / len(tokens)
    return [
        RefWord(text=token, speaker=segment.speaker, start=segment.start + i * step, end=segment.start + (i + 1) * step)
        for i, token in enumerate(tokens)
    ]


# --- HarperValleyBank ---------------------------------------------------------------------


def hvb_reference(item: str, transcript: Sequence[Mapping[str, Any]], audio_s: float, corpus: str = "harper-valley-bank") -> Reference:
    """A reference from a call's transcript file: the human transcript of every segment with
    the segment's offset and duration in the recording, spoken by its role."""
    segments: list[RefSegment] = []
    for seg in sorted(transcript, key=lambda s: (int(s["offset_ms"]), int(s.get("index", 0)))):
        text = str(seg.get("human_transcript") or "").strip()
        if not text:
            continue
        start = int(seg["offset_ms"]) / 1000.0
        end = start + int(seg["duration_ms"]) / 1000.0
        segments.append(RefSegment(start=start, end=end, speaker=str(seg["speaker_role"]), text=text))
    words = [word for segment in segments for word in spread_words(segment)]
    speakers = tuple(role for role in HVB_ROLES if any(s.speaker == role for s in segments))
    return Reference(
        item=item,
        corpus=corpus,
        audio_s=float(audio_s),
        speakers=speakers,
        segments=tuple(segments),
        words=tuple(words),
        word_times=WORD_TIMES_SPREAD,
        notes=("human transcripts with segment times; word times spread evenly inside each segment",),
    )


def hvb_delays(transcript: Sequence[Mapping[str, Any]]) -> dict[str, tuple[float, ...]]:
    """Per role, the distinct values of offset minus start over its segments, in seconds.

    offset is a segment's position in the recording and start its position in the role's own
    channel; the difference is how much later that channel began than the recording. A single
    value per role is the expected shape.
    """
    delays: dict[str, tuple[float, ...]] = {}
    for role in HVB_ROLES:
        values = sorted({(int(s["offset_ms"]) - int(s["start_ms"])) / 1000.0 for s in transcript if s["speaker_role"] == role})
        if values:
            delays[role] = tuple(values)
    return delays


def hvb_alignment_notes(
    delays: Mapping[str, Sequence[float]], channel_difference_s: float, tolerance_s: float = 0.05
) -> list[str]:
    """Notes for a call whose transcript delays disagree with its channel lengths.

    channel_difference_s is the caller channel length minus the agent channel length: the
    agent's expected delay when positive, the caller's when negative. A value within the
    tolerance of the expected delay agrees; every other value is reported.
    """
    expected = {"agent": max(0.0, channel_difference_s), "caller": max(0.0, -channel_difference_s)}
    notes: list[str] = []
    for role, values in delays.items():
        off = [v for v in values if abs(v - expected.get(role, 0.0)) > tolerance_s]
        if off:
            notes.append(
                f"{role} channel delay in the transcript is {', '.join(f'{v:.3f}' for v in off)} s "
                f"against {expected.get(role, 0.0):.3f} s from the channel lengths; segment times may be off"
            )
    return notes


def mix_channels(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Sum two 16-bit channels of one call so that both end together.

    The shorter channel is padded with silence at the front, which is where the later party's
    recording began; the sum is clipped to the 16-bit range.
    """
    a = np.asarray(first, dtype=np.int32).ravel()
    b = np.asarray(second, dtype=np.int32).ravel()
    n = max(len(a), len(b))
    a = np.concatenate([np.zeros(n - len(a), dtype=np.int32), a])
    b = np.concatenate([np.zeros(n - len(b), dtype=np.int32), b])
    return np.clip(a + b, _PCM_MIN, _PCM_MAX).astype(np.int16)


def read_pcm16(path: str | Path) -> tuple[np.ndarray, int]:
    """A mono 16-bit PCM WAV at any rate as int16 samples, with its rate."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono 16-bit PCM, got {handle.getnchannels()} channels, {handle.getsampwidth()} bytes")
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").copy(), rate


def write_pcm16(path: str | Path, samples: np.ndarray, rate: int) -> None:
    """Write int16 samples as a mono 16-bit PCM WAV at the given rate."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(np.asarray(samples, dtype="<i2").tobytes())


# --- AMI ------------------------------------------------------------------------------


def _xml_root(source: bytes | str) -> ET.Element:
    """Parse XML given as bytes (the declared encoding is honoured) or text."""
    if isinstance(source, str):
        match = re.search(r'encoding="([^"]+)"', source[:200])
        source = source.encode(match.group(1) if match else "utf-8")
    return ET.fromstring(source)


def ami_speakers(meetings_xml: bytes | str, meeting: str) -> dict[str, str]:
    """Agent letter to global speaker name for one meeting, from the corpus meeting map."""
    names: dict[str, str] = {}
    for element in _xml_root(meetings_xml).iter("meeting"):
        if element.get("observation") != meeting:
            continue
        for speaker in element.iter("speaker"):
            agent = speaker.get("nxt_agent")
            name = speaker.get("global_name")
            if agent and name:
                names[agent] = name
    return names


def ami_words(words_xml: bytes | str, speaker: str) -> list[RefWord]:
    """Timed words of one speaker from a words file; punctuation entries and anything that is
    not a word element (vocal sounds, gaps) are left out."""
    words: list[RefWord] = []
    for element in _xml_root(words_xml).iter("w"):
        if element.get("punc") == "true":
            continue
        start = element.get("starttime")
        end = element.get("endtime")
        text = (element.text or "").strip()
        if start is None or end is None or not text:
            continue
        words.append(RefWord(text=text, speaker=speaker, start=float(start), end=float(end)))
    words.sort(key=lambda w: (w.start, w.end))
    return words


def ami_segments(segments_xml: bytes | str, speaker: str) -> list[RefSegment]:
    """One speaker's transcriber segments (start and end only; the text is filled from the
    words inside each span when the reference is assembled)."""
    segments: list[RefSegment] = []
    for element in _xml_root(segments_xml).iter("segment"):
        start = element.get("transcriber_start")
        end = element.get("transcriber_end")
        if start is None or end is None:
            continue
        segments.append(RefSegment(start=float(start), end=float(end), speaker=speaker, text=""))
    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def cut_reference(
    words: Sequence[RefWord], segments: Sequence[RefSegment], window: tuple[float, float]
) -> tuple[list[RefWord], list[RefSegment]]:
    """Words wholly inside the window and segments clipped to it, both shifted to start at 0."""
    w0, w1 = float(window[0]), float(window[1])
    if w1 <= w0:
        raise ValueError(f"window must be increasing, got {window}")
    kept_words = [
        RefWord(text=w.text, speaker=w.speaker, start=w.start - w0, end=w.end - w0)
        for w in words
        if w.start is not None and w.end is not None and w.start >= w0 and w.end <= w1
    ]
    kept_segments = []
    for s in segments:
        start, end = max(s.start, w0), min(s.end, w1)
        if end > start:
            kept_segments.append(RefSegment(start=start - w0, end=end - w0, speaker=s.speaker, text=s.text))
    return kept_words, kept_segments


def ami_reference(
    item: str,
    words_by_speaker: Mapping[str, Sequence[RefWord]],
    segments_by_speaker: Mapping[str, Sequence[RefSegment]],
    audio_s: float,
    window: tuple[float, float] | None = None,
    corpus: str = "ami-es2002a",
) -> Reference:
    """A reference from the words and segments of every speaker, optionally cut to a window.

    Each segment's text is the speaker's words whose start lies inside its span; words that
    fall in no segment still count in the reference text. Words are ordered by start time,
    so the reference text interleaves speakers as they spoke.
    """
    words = sorted((w for group in words_by_speaker.values() for w in group), key=lambda w: (float(w.start or 0.0), float(w.end or 0.0)))
    segments = sorted((s for group in segments_by_speaker.values() for s in group), key=lambda s: (s.start, s.end))
    notes = ["word times from the corpus's forced alignment; segments from the transcriber"]
    if window is not None:
        words, segments = cut_reference(words, segments, window)
        notes.append(f"window {window[0]:.0f}-{window[1]:.0f} s of the meeting, shifted to start at 0")
    filled: list[RefSegment] = []
    for s in segments:
        inside = [w.text for w in words if w.speaker == s.speaker and w.start is not None and s.start <= w.start < s.end]
        filled.append(RefSegment(start=s.start, end=s.end, speaker=s.speaker, text=" ".join(inside)))
    speakers = tuple(sorted({w.speaker for w in words} | {s.speaker for s in filled}))
    return Reference(
        item=item,
        corpus=corpus,
        audio_s=float(audio_s),
        speakers=speakers,
        segments=tuple(filled),
        words=tuple(words),
        word_times=WORD_TIMES_ANNOTATED,
        notes=tuple(notes),
    )


def cut_window(samples: np.ndarray, rate: int, window: tuple[float, float]) -> np.ndarray:
    """The samples inside [start, end) seconds; the end is clamped to the signal."""
    w0, w1 = float(window[0]), float(window[1])
    if w1 <= w0 or w0 < 0.0:
        raise ValueError(f"window must be increasing and non-negative, got {window}")
    start = int(round(w0 * rate))
    end = min(len(samples), int(round(w1 * rate)))
    if start >= len(samples):
        raise ValueError(f"window starts at {w0} s, beyond the {len(samples) / rate:.1f} s signal")
    return np.asarray(samples)[start:end]


# --- LibriSpeech --------------------------------------------------------------------------


def librispeech_transcripts(trans_txt: str) -> dict[str, str]:
    """Utterance id to text from a chapter transcript file (one "<id> TEXT" line each)."""
    out: dict[str, str] = {}
    for line in trans_txt.splitlines():
        line = line.strip()
        if not line:
            continue
        utterance, _, text = line.partition(" ")
        out[utterance] = text.strip()
    return out


def librispeech_reference(item: str, utterance: str, text: str, audio_s: float, corpus: str = "librispeech-test-other") -> Reference:
    """A single-reader reference with no word times; the text is lower-cased as the corpus
    writes it in capitals."""
    segment = RefSegment(start=0.0, end=float(audio_s), speaker=LIBRISPEECH_SPEAKER, text=text.strip().lower())
    words = tuple(RefWord(text=token, speaker=LIBRISPEECH_SPEAKER) for token in segment.text.split())
    return Reference(
        item=item,
        corpus=corpus,
        audio_s=float(audio_s),
        speakers=(LIBRISPEECH_SPEAKER,),
        segments=(segment,),
        words=words,
        word_times=WORD_TIMES_NONE,
        notes=(f"utterance {utterance}; one reader, no word times",),
    )
