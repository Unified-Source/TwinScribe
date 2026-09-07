"""Subtitle rendering: SubRip (.srt) and WebVTT (.vtt) cues built from the transcript lines,
so that the transcript plays against the recording line by line in any player that loads a
subtitle file beside the media.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from twinscribe.labelling import UNLABELLED_NAME
from twinscribe.outputs.transcript_doc import speaker_names

MAX_CUE_CHARS = 84
MAX_ROW_CHARS = 42
MAX_CUE_SECONDS = 7.0
MIN_CUE_SECONDS = 1.0


@dataclass(frozen=True)
class Cue:
    """One subtitle: its time span and its text, with an explicit newline between rows."""

    start: float
    end: float
    text: str


def _timestamp(seconds: float, separator: str) -> str:
    millis = max(0, int(round(seconds * 1000.0)))
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def srt_timestamp(seconds: float) -> str:
    """HH:MM:SS,mmm."""
    return _timestamp(seconds, ",")


def vtt_timestamp(seconds: float) -> str:
    """HH:MM:SS.mmm."""
    return _timestamp(seconds, ".")


def split_rows(text: str, max_row_chars: int = MAX_ROW_CHARS) -> str:
    """Break a long cue into two rows at the space nearest the middle."""
    if len(text) <= max_row_chars:
        return text
    middle = len(text) // 2
    candidates = [i for i, ch in enumerate(text) if ch == " "]
    if not candidates:
        return text
    cut = min(candidates, key=lambda i: abs(i - middle))
    return text[:cut].rstrip() + "\n" + text[cut + 1 :].lstrip()


def _chunk_fits(chunk: Sequence[dict], prefix_len: int, max_chars: int, max_seconds: float) -> bool:
    length = prefix_len + sum(len(str(w["w"]).strip()) + 1 for w in chunk) - 1
    span = float(chunk[-1]["e"]) - float(chunk[0]["s"])
    return length <= max_chars and span <= max_seconds


def _balanced_chunks(words: Sequence[dict], prefix_len: int, max_chars: int, max_seconds: float) -> list[list[dict]]:
    """Halve a run of words at the boundary nearest its middle (by characters) until every
    chunk fits; a single word that does not fit is kept whole."""
    chunk = list(words)
    if len(chunk) <= 1 or _chunk_fits(chunk, prefix_len, max_chars, max_seconds):
        return [chunk]
    total = sum(len(str(w["w"]).strip()) + 1 for w in chunk)
    running = 0
    best_index = 1
    best_distance = float("inf")
    for index in range(1, len(chunk)):
        running += len(str(chunk[index - 1]["w"]).strip()) + 1
        distance = abs(running - total / 2.0)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    left, right = chunk[:best_index], chunk[best_index:]
    return _balanced_chunks(left, prefix_len, max_chars, max_seconds) + _balanced_chunks(
        right, prefix_len, max_chars, max_seconds
    )


def build_cues(
    doc: Mapping[str, Any],
    max_chars: int = MAX_CUE_CHARS,
    max_seconds: float = MAX_CUE_SECONDS,
    min_seconds: float = MIN_CUE_SECONDS,
) -> list[Cue]:
    """Cues from the transcript lines.

    A line that exceeds max_chars of text or max_seconds of time is halved at the word
    boundary nearest its middle, and the halves again until every chunk fits, so that the
    chunks of one line are of similar size rather than a full cue followed by a one-word tail.
    Every cue is prefixed with the name of the speaker so that a viewer who joins mid-way knows
    who is speaking. A cue shorter than min_seconds is lengthened, up to the start of the next
    cue.
    """
    if max_chars < 8 or max_seconds <= 0.0 or min_seconds < 0.0:
        raise ValueError("max_chars must be at least 8, max_seconds positive, min_seconds not negative")
    names = speaker_names(doc)
    cues: list[Cue] = []
    for line in doc.get("lines", []):
        label = line.get("speaker")
        name = names.get(label, label) if label is not None else UNLABELLED_NAME
        prefix = f"{name}: "
        words = [w for w in line.get("words", []) if str(w.get("w", "")).strip()]
        if not words:
            continue
        for chunk in _balanced_chunks(words, len(prefix), max_chars, max_seconds):
            text = prefix + " ".join(str(w["w"]).strip() for w in chunk)
            cues.append(Cue(float(chunk[0]["s"]), float(chunk[-1]["e"]), split_rows(text)))

    adjusted: list[Cue] = []
    for position, cue in enumerate(cues):
        end = max(cue.end, cue.start + min_seconds)
        if position + 1 < len(cues):
            next_start = cues[position + 1].start
            end = min(end, next_start) if next_start > cue.end else cue.end
        adjusted.append(Cue(cue.start, end, cue.text))
    return adjusted


def render_srt(cues: Sequence[Cue]) -> str:
    """SubRip text: index, time line, text, blank line."""
    blocks = [
        f"{index}\n{srt_timestamp(cue.start)} --> {srt_timestamp(cue.end)}\n{cue.text}\n"
        for index, cue in enumerate(cues, start=1)
    ]
    return "\n".join(blocks)


def render_vtt(cues: Sequence[Cue]) -> str:
    """WebVTT text."""
    blocks = ["WEBVTT\n"] + [
        f"{vtt_timestamp(cue.start)} --> {vtt_timestamp(cue.end)}\n{cue.text}\n" for cue in cues
    ]
    return "\n".join(blocks)
