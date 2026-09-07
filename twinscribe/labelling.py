"""Speaker attribution and line building.

The published engine yields words with times and the diarizer yields labelled turns; neither
knows about the other. This module gives every word the label of the turn it overlaps most,
fills the words no turn covers from their neighbours, absorbs a one-word flicker inside an
utterance where a turn boundary jittered against a word boundary, groups consecutive words of
one speaker into lines for reading, and counts words per speaker, which is the cheapest way to
notice a participant the labelling lost.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from twinscribe.engines.base import Segment, SpeakerTurn, Word

UNLABELLED_NAME = "Unknown speaker"
DEFAULT_MAX_ATTACH_S = 1.0
DEFAULT_LINE_GAP_S = 1.5
DEFAULT_LINE_WORDS = 60
# A run of words with a label of its own inside an utterance is a flicker, not a speaker,
# when it is shorter than both of these.
DEFAULT_MIN_RUN_WORDS = 2
DEFAULT_MIN_RUN_S = 0.6


@dataclass(frozen=True)
class Line:
    """Consecutive words of one speaker, read as one line of transcript."""

    start: float
    end: float
    speaker: str | None
    words: tuple[Word, ...]

    @property
    def text(self) -> str:
        return " ".join(piece for piece in (w.text.strip() for w in self.words) if piece)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class SpeakerCount:
    """Words and seconds attributed to one label; label is None for unlabelled words."""

    label: str | None
    words: int
    seconds: float


class _TurnIndex:
    """Turns sorted by start with a running maximum of ends, for overlap queries."""

    def __init__(self, turns: Iterable[SpeakerTurn]) -> None:
        self._turns = sorted(turns, key=lambda t: (t.start, t.end, t.label))
        self._starts = [t.start for t in self._turns]
        self._max_ends: list[float] = []
        running = float("-inf")
        for turn in self._turns:
            running = max(running, turn.end)
            self._max_ends.append(running)

    def __len__(self) -> int:
        return len(self._turns)

    def overlapping(self, a: float, b: float) -> list[SpeakerTurn]:
        """Turns sharing a positive-length intersection with [a, b], in start order."""
        upper = bisect_left(self._starts, b)
        lower = bisect_right(self._max_ends, a)
        return [t for t in self._turns[lower:upper] if t.start < b and t.end > a]

    def nearest(self, a: float, b: float) -> tuple[SpeakerTurn, float] | None:
        """The turn closest to [a, b] and its distance; None when there are no turns."""
        best: tuple[SpeakerTurn, float] | None = None
        for turn in self._turns:
            distance = max(0.0, turn.start - b, a - turn.end)
            if best is None or distance < best[1]:
                best = (turn, distance)
        return best


def label_words(
    words: Sequence[Word],
    turns: Sequence[SpeakerTurn],
    max_attach_s: float = DEFAULT_MAX_ATTACH_S,
) -> list[str | None]:
    """One label per word, or None where no label can be justified.

    A word takes the label of the turn it overlaps most (the earlier turn on a tie). A word no
    turn overlaps takes the label of the nearest turn when that turn is within max_attach_s,
    which covers the word-boundary jitter between the two engines. Words still unlabelled
    inherit the label of the previous labelled word, or of the next one at the start of the
    file, so a labelled recording never shows a stray unknown speaker for one word; only a
    recording with no turns at all stays unlabelled throughout.
    """
    if max_attach_s < 0.0:
        raise ValueError("max_attach_s must not be negative")
    index = _TurnIndex(turns)
    labels: list[str | None] = []
    for word in words:
        chosen: str | None = None
        if len(index):
            candidates = index.overlapping(word.start, word.end)
            if candidates:
                best_overlap = -1.0
                for turn in candidates:
                    overlap = min(word.end, turn.end) - max(word.start, turn.start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        chosen = turn.label
            else:
                near = index.nearest(word.start, word.end)
                if near is not None and near[1] <= max_attach_s:
                    chosen = near[0].label
        labels.append(chosen)

    previous: str | None = None
    for position, label in enumerate(labels):
        if label is None:
            labels[position] = previous
        else:
            previous = label
    following: str | None = None
    for position in range(len(labels) - 1, -1, -1):
        if labels[position] is None:
            labels[position] = following
        else:
            following = labels[position]
    return labels


def _utterance_groups(words: Sequence[Word], segments: Iterable[Segment]) -> list[list[int]]:
    """Word indices per utterance, by the utterance whose span holds the word's middle."""
    spans = sorted((float(s.start), float(s.end)) for s in segments)
    groups: list[list[int]] = [[] for _ in spans]
    for index, word in enumerate(words):
        middle = 0.5 * (word.start + word.end)
        position = bisect_right([s for s, _ in spans], middle) - 1
        if position >= 0 and middle <= spans[position][1]:
            groups[position].append(index)
    return [group for group in groups if group]


def smooth_labels(
    words: Sequence[Word],
    labels: Sequence[str | None],
    segments: Iterable[Segment],
    min_run_words: int = DEFAULT_MIN_RUN_WORDS,
    min_run_s: float = DEFAULT_MIN_RUN_S,
) -> tuple[list[str | None], int]:
    """Absorb label flicker inside the utterances of the published engine.

    Within one utterance, a run of words carrying one label that is shorter than min_run_words
    and than min_run_s, with the same other label on both sides of it, takes that other label:
    a turn boundary jittering against a word boundary is not a change of speaker. Runs at
    either edge of an utterance are left alone, so a short interjection at the start or the end
    of a sentence keeps its label; so are runs that cross utterances. Returns the labels and
    the count of words relabelled.
    """
    if len(words) != len(labels):
        raise ValueError(f"{len(words)} words but {len(labels)} labels")
    if min_run_words < 1 or min_run_s < 0.0:
        raise ValueError("min_run_words must be at least 1 and min_run_s not negative")
    result = list(labels)
    changed = 0
    for group in _utterance_groups(words, segments):
        while True:
            runs: list[tuple[int, int]] = []                 # (first, last) positions within the group
            for position, index in enumerate(group):
                if runs and result[group[runs[-1][0]]] == result[index]:
                    runs[-1] = (runs[-1][0], position)
                else:
                    runs.append((position, position))
            relabelled = False
            for run_index in range(1, len(runs) - 1):
                first, last = runs[run_index]
                before = result[group[runs[run_index - 1][1]]]
                after = result[group[runs[run_index + 1][0]]]
                count = last - first + 1
                span = words[group[last]].end - words[group[first]].start
                if before == after and count < min_run_words and span < min_run_s:
                    for position in range(first, last + 1):
                        result[group[position]] = before
                    changed += count
                    relabelled = True
                    break
            if not relabelled:
                break
    return result, changed


def build_lines(
    words: Sequence[Word],
    labels: Sequence[str | None],
    max_gap_s: float = DEFAULT_LINE_GAP_S,
    max_words: int = DEFAULT_LINE_WORDS,
) -> list[Line]:
    """Group words into lines: a new line starts when the speaker changes, when the pause
    before a word exceeds max_gap_s, or when a line has reached max_words."""
    if len(words) != len(labels):
        raise ValueError(f"{len(words)} words but {len(labels)} labels")
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    lines: list[Line] = []
    current: list[Word] = []
    current_label: str | None = None

    def close() -> None:
        if current:
            lines.append(
                Line(
                    start=current[0].start,
                    end=max(w.end for w in current),
                    speaker=current_label,
                    words=tuple(current),
                )
            )

    for word, label in zip(words, labels):
        if current:
            gap = word.start - current[-1].end
            if label != current_label or gap > max_gap_s or len(current) >= max_words:
                close()
                current = []
        if not current:
            current_label = label
        current.append(word)
    close()
    return lines


def summarise_speakers(lines: Iterable[Line]) -> list[SpeakerCount]:
    """Words and seconds per label in order of first appearance; unlabelled words last."""
    order: list[str | None] = []
    words: dict[str | None, int] = {}
    seconds: dict[str | None, float] = {}
    for line in lines:
        if line.speaker not in words:
            order.append(line.speaker)
            words[line.speaker] = 0
            seconds[line.speaker] = 0.0
        words[line.speaker] += len(line.words)
        seconds[line.speaker] += line.duration
    labelled = [label for label in order if label is not None]
    ordered: list[str | None] = labelled + ([None] if None in words else [])
    return [SpeakerCount(label=label, words=words[label], seconds=seconds[label]) for label in ordered]


def default_names(labels: Iterable[str | None]) -> dict[str, str]:
    """Display names by order of first appearance: Speaker 1, Speaker 2, and so on."""
    names: dict[str, str] = {}
    for label in labels:
        if label is not None and label not in names:
            names[label] = f"Speaker {len(names) + 1}"
    return names


def display_name(label: str | None, names: dict[str, str]) -> str:
    """The display name for a label; the unlabelled name for None; the label itself when no
    name has been assigned."""
    if label is None:
        return UNLABELLED_NAME
    return names.get(label, label)
