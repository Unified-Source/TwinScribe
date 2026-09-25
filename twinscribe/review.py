"""Two-engine review list.

A mark is raised wherever the published transcript has no word for a while and the detector
engine, whose text is never published, placed words there. The marks are what a person
listens to. This module builds the marks, scores them against a reference when one exists,
and produces the review-set document the verification screen reads.
"""

from __future__ import annotations

import os
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from twinscribe.runrecord import write_json_atomic
from twinscribe.text import tokenize_norm

if TYPE_CHECKING:
    from twinscribe.engines.base import Transcript, Word

REVIEW_SCHEMA = "twinscribe.review.v1"

# Tolerance for the span-length comparison, so that a span whose end points come from
# decimal word times (for example 2.0 - 1.2) still counts as exactly 0.8 s long.
_LENGTH_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Mark:
    """One place to listen to.

    start and end are the padded play window, clamped to the audio; span_start and span_end
    are the publisher's silent span; detector_words and detector_text describe what the
    detector engine placed inside that span.
    """

    start: float
    end: float
    span_start: float
    span_end: float
    detector_words: int
    detector_text: str


@dataclass(frozen=True)
class ReviewEvaluation:
    """How the marks fared against a reference: precision, recall and listening cost."""

    marks: int
    marks_on_speech: int
    precision: float
    dropped_words: int
    dropped_covered: int
    recall: float
    audio_to_review_s: float
    audio_to_review_fraction: float


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """Strict overlap: the intervals share a positive-length intersection.

    Intervals that only touch do not overlap. A zero-length interval overlaps another only
    when it lies strictly inside it; two zero-length intervals never overlap.
    """
    return a_start < b_end and b_start < a_end


class _Intervals:
    """Intervals sorted by start with a running maximum of ends, for overlap queries.

    Intervals with start < b lie in a prefix of the sorted order; among them, those with
    end > a lie in a suffix once the running maximum of ends is used as the lower bound. The
    overlapping intervals are therefore a contiguous window, found with two bisections.
    """

    def __init__(self, intervals: Iterable[tuple[float, float]]) -> None:
        items = sorted(
            (float(start), float(end), index) for index, (start, end) in enumerate(intervals)
        )
        self._starts = [start for start, _, _ in items]
        self._ends = [end for _, end, _ in items]
        self._indices = [index for _, _, index in items]
        self._max_ends: list[float] = []
        running = float("-inf")
        for end in self._ends:
            running = max(running, end)
            self._max_ends.append(running)

    def overlapping(self, a: float, b: float) -> list[int]:
        """Original indices of the intervals overlapping [a, b], in order of start."""
        upper = bisect_left(self._starts, b)
        lower = bisect_right(self._max_ends, a)
        return [
            self._indices[position]
            for position in range(lower, upper)
            if _overlaps(self._starts[position], self._ends[position], a, b)
        ]


def _merged(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of intervals as sorted, disjoint intervals; touching intervals merge."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _silent_spans(published: Iterable[Word], audio_s: float) -> list[tuple[float, float]]:
    """Maximal spans of [0, audio_s] with no published word coverage, in time order."""
    covered = _merged((max(0.0, w.start), min(audio_s, w.end)) for w in published)
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in covered:
        if start > cursor:
            spans.append((cursor, start))
        cursor = max(cursor, end)
    if audio_s > cursor:
        spans.append((cursor, audio_s))
    return spans


def checking_windows(
    published: Iterable[Word],
    audio_s: float,
    min_silence_s: float = 0.8,
    margin_s: float = 0.5,
) -> list[tuple[float, float]]:
    """The spans a checker need decode when it checks only where the publisher fell silent:
    every published-silent span of at least min_silence_s, widened by margin_s on both sides
    for context, clamped to [0, audio_s], and merged where they touch or overlap. Leading and
    trailing silence count, as in build_review."""
    if min_silence_s < 0.0 or margin_s < 0.0:
        raise ValueError("min_silence_s and margin_s must not be negative")
    if audio_s <= 0.0:
        return []
    windows = [
        (max(0.0, start - margin_s), min(audio_s, end + margin_s))
        for start, end in _silent_spans(published, audio_s)
        if end - start >= min_silence_s
    ]
    merged: list[tuple[float, float]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _normalised(word: Word) -> str:
    """The word's text as the normaliser reads it, for comparing the two engines' words."""
    return " ".join(tokenize_norm(word.text))


def echoes_removed(
    inside: Sequence[Word],
    before: Sequence[Word],
    after: Sequence[Word],
    detector_before: Sequence[Word] = (),
    detector_after: Sequence[Word] = (),
) -> list[Word]:
    """The detector words of a span without those that only repeat the published words
    bordering it.

    The two engines time a word differently by a fraction of a second, so the detector's copy
    of the word just before a gap, or just after it, can fall inside the gap; such a copy is
    an echo of speech the transcript already has, not missed speech. The longest run at the
    start of the span that matches, word for word after normalisation, the published words
    ending before the span is dropped, and so is the longest run at the end that matches the
    published words starting after it. A run the detector also heard outside the span, where
    the publisher has it, is not an echo but a repetition and stays: `detector_before` and
    `detector_after` are the detector's words ending before and starting after the span.
    """
    words = list(inside)
    norms = [_normalised(w) for w in words]
    outside_before = [_normalised(w) for w in detector_before]
    outside_after = [_normalised(w) for w in detector_after]
    head = 0
    for count in range(min(len(words), len(before)), 0, -1):
        candidate = norms[:count]
        if all(candidate) and candidate == [_normalised(w) for w in before[-count:]]:
            if outside_before[-count:] != candidate:
                head = count
            break
    words, norms = words[head:], norms[head:]
    tail = 0
    for count in range(min(len(words), len(after)), 0, -1):
        candidate = norms[len(norms) - count:]
        if all(candidate) and candidate == [_normalised(w) for w in after[:count]]:
            if outside_after[:count] != candidate:
                tail = count
            break
    return words[: len(words) - tail]


def build_review(
    published: list[Word],
    detector: list[Word],
    audio_s: float,
    min_silence_s: float = 0.8,
    min_detector_words: int = 2,
    pad_s: float = 0.4,
) -> list[Mark]:
    """Marks for every maximal published-silent span of at least min_silence_s that holds
    at least min_detector_words detector words; leading and trailing silence count.

    A detector word falls in a span when its interval overlaps the span; detector words that
    only echo the published words bordering the span do not count (echoes_removed), and the
    mark's hint carries the words that remain. The play window is the span padded by pad_s
    on both sides and clamped to [0, audio_s].
    """
    if min_silence_s < 0.0 or pad_s < 0.0 or min_detector_words < 0:
        raise ValueError("min_silence_s, pad_s and min_detector_words must not be negative")
    if audio_s <= 0.0:
        return []
    detector_sorted = sorted(detector, key=lambda w: (w.start, w.end))
    detector_index = _Intervals((w.start, w.end) for w in detector_sorted)
    by_end = sorted(published, key=lambda w: (w.end, w.start))
    ends = [w.end for w in by_end]
    by_start = sorted(published, key=lambda w: (w.start, w.end))
    starts = [w.start for w in by_start]
    detector_by_end = sorted(detector, key=lambda w: (w.end, w.start))
    detector_ends = [w.end for w in detector_by_end]
    detector_starts = [w.start for w in detector_sorted]
    marks: list[Mark] = []
    for span_start, span_end in _silent_spans(published, audio_s):
        if (span_end - span_start) + _LENGTH_TOLERANCE < min_silence_s:
            continue
        inside = [detector_sorted[i] for i in detector_index.overlapping(span_start, span_end)]
        if len(inside) < min_detector_words:
            continue
        count = len(inside)
        last_before = bisect_right(ends, span_start + _LENGTH_TOLERANCE)
        first_after = bisect_left(starts, span_end - _LENGTH_TOLERANCE)
        detector_last_before = bisect_right(detector_ends, span_start + _LENGTH_TOLERANCE)
        detector_first_after = bisect_left(detector_starts, span_end - _LENGTH_TOLERANCE)
        inside = echoes_removed(
            inside,
            by_end[max(0, last_before - count):last_before],
            by_start[first_after:first_after + count],
            detector_by_end[max(0, detector_last_before - count):detector_last_before],
            detector_sorted[detector_first_after:detector_first_after + count],
        )
        if len(inside) < min_detector_words:
            continue
        text = " ".join(piece for piece in (w.text.strip() for w in inside) if piece)
        marks.append(
            Mark(
                start=max(0.0, span_start - pad_s),
                end=min(audio_s, span_end + pad_s),
                span_start=span_start,
                span_end=span_end,
                detector_words=len(inside),
                detector_text=text,
            )
        )
    return marks


def evaluate_review(
    marks: Sequence[Mark],
    published: list[Word],
    reference: list[Word],
    audio_s: float,
) -> ReviewEvaluation:
    """Score marks against a reference.

    A dropped word is a reference word no published word overlaps. A mark is on speech when
    any reference word overlaps its span; a dropped word is covered when any mark's span
    overlaps it. Precision and recall are 0.0 when their denominators are zero. The audio to
    review is the union of the padded play windows.
    """
    published_index = _Intervals((w.start, w.end) for w in published)
    reference_index = _Intervals((w.start, w.end) for w in reference)
    mark_index = _Intervals((m.span_start, m.span_end) for m in marks)

    dropped = [w for w in reference if not published_index.overlapping(w.start, w.end)]
    on_speech = sum(1 for m in marks if reference_index.overlapping(m.span_start, m.span_end))
    covered = sum(1 for w in dropped if mark_index.overlapping(w.start, w.end))

    review_s = sum(end - start for start, end in _merged((m.start, m.end) for m in marks))
    return ReviewEvaluation(
        marks=len(marks),
        marks_on_speech=on_speech,
        precision=on_speech / len(marks) if marks else 0.0,
        dropped_words=len(dropped),
        dropped_covered=covered,
        recall=covered / len(dropped) if dropped else 0.0,
        audio_to_review_s=review_s,
        audio_to_review_fraction=review_s / audio_s if audio_s > 0.0 else 0.0,
    )


def _engine_facts(transcript: Transcript) -> dict[str, str]:
    return {
        "engine": str(transcript.engine),
        "model": str(transcript.model),
        "preset": str(transcript.preset),
    }


def _mark_entry(mark: Mark) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "start": float(mark.start),
        "end": float(mark.end),
        "span_start": float(mark.span_start),
        "span_end": float(mark.span_end),
        "detector_words": int(mark.detector_words),
        "detector_text": mark.detector_text,
        "reference_words": None,
        "reference_speakers": None,
    }
    return entry


def review_set(
    published: Transcript,
    detector: Transcript,
    audio_path: str,
    marks: list[Mark],
    reference: list[Word] | None = None,
    reference_speakers: list[str] | None = None,
    parts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The document the verification screen reads.

    `parts`, for a recording written in parts, lists each file's audio reference and its
    offset in seconds, so that the screen can rebuild the joined audio when its copy is gone.

    reference_words (count in the span), reference_speakers (distinct labels in the span, in
    order of first appearance) and evaluation are filled only when a reference is supplied;
    reference_speakers stays None when no labels are supplied. reference_speakers, when
    given, is parallel to reference.
    """
    if reference_speakers is not None:
        if reference is None:
            raise ValueError("reference_speakers requires reference")
        if len(reference_speakers) != len(reference):
            raise ValueError(
                f"reference_speakers has {len(reference_speakers)} labels for "
                f"{len(reference)} reference words"
            )
    published_words = list(published.words)
    audio_s = float(published.audio_s)
    doc: dict[str, Any] = {
        "schema": REVIEW_SCHEMA,
        "audio": os.fspath(audio_path),
        "duration_s": audio_s,
        "publisher": _engine_facts(published),
        "detector": _engine_facts(detector),
        "transcript": [
            {"s": float(w.start), "e": float(w.end), "w": str(w.text)} for w in published_words
        ],
        "marks": [_mark_entry(mark) for mark in marks],
        "evaluation": None,
    }
    if parts:
        doc["parts"] = [{"audio": str(part["audio"]), "offset_s": float(part["offset_s"])} for part in parts]
    if reference is not None:
        reference_index = _Intervals((w.start, w.end) for w in reference)
        for entry, mark in zip(doc["marks"], marks):
            inside = reference_index.overlapping(mark.span_start, mark.span_end)
            entry["reference_words"] = len(inside)
            if reference_speakers is not None:
                labels: list[str] = []
                for index in inside:
                    label = str(reference_speakers[index])
                    if label not in labels:
                        labels.append(label)
                entry["reference_speakers"] = labels
        doc["evaluation"] = asdict(evaluate_review(marks, published_words, reference, audio_s))
    return doc


def write_review_set(doc: dict[str, Any], path: str | os.PathLike[str]) -> None:
    """Write the review-set document as JSON atomically (temporary file, then rename)."""
    write_json_atomic(doc, path)
