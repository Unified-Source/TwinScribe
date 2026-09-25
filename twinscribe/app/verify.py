"""The verification screen: work through the review list with the audio at each mark.

A review set (schema `twinscribe.review.v1`) lists the spans where the published engine
heard nothing and the second engine heard speech. This screen plays each span, shows the
published transcript either side of the gap, and takes for every mark either that nothing
was said or the words that were, typed in place over what the second engine heard, with the
speaker they belong to. Every resolution is written at once to `<review_set stem>.session.json`
beside the review set and, when the transcript document sits beside it, into the transcript
itself: the listener's words go into the line where the gap is, marked as the listener's, and
the text, Word and subtitle files are written again.

Entry point: `python -m twinscribe.app.verify <review_set.json> [--dark] [--shot out.png]`.

Screenshots (`--shot`) must be taken on the platform's real backend. Under
`QT_QPA_PLATFORM=offscreen` there is no font database and every glyph renders as a box, so
a screenshot taken there shows nothing useful.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QFont, QKeyEvent, QPalette
from PySide6.QtMultimedia import QAudioDevice, QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from twinscribe.app.playable import PlayableCopy, copy_is_current, play_dir, playable_copy_path
from twinscribe.pipeline import Parts
from twinscribe.amend import listener_line_for, resolutions_from_document
from twinscribe.app.app_icon import app_icon
from twinscribe.app.theme import Theme, apply_theme, theme_for
from twinscribe.app.timeline import Timeline, format_mss
from twinscribe.labelling import UNLABELLED_NAME
from twinscribe.outputs.transcript_doc import load_document, speaker_names
from twinscribe.runrecord import write_json_atomic

REVIEW_SCHEMA = "twinscribe.review.v1"
SESSION_SCHEMA = "twinscribe.review-session.v1"
CONTEXT_WORDS = 14
GAP_MARKER = "[ GAP ]"
NUDGE_S = 5.0
SHOT_DELAY_MS = 800
TICK = "✓"  # check mark shown before a resolved row
ROW_NOTE_CHARS = 24
STATUS_NOTE_CHARS = 60
CAPTION_TAIL = "the kept words go into the transcript at the gap, marked as the listener's."
SPEAKER_INFERRED = "Speaker: as the lines around the gap suggest"
SPEAKER_NONE = "No speaker"

STATUS_OPEN = "open"
STATUS_NOTHING = "nothing"
STATUS_TEXT = "text"


# ----- review set --------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptWord:
    """One published word with its interval in seconds."""

    s: float
    e: float
    w: str


@dataclass(frozen=True)
class ReviewMark:
    """One mark of the review list as the screen needs it.

    `start` and `end` are what to play (the silent span padded); `span_start` and
    `span_end` are the publisher's silent span. `reference_words` and
    `reference_speakers` are present only in test mode.
    """

    start: float
    end: float
    span_start: float
    span_end: float
    detector_words: int
    detector_text: str
    reference_words: int | None = None
    reference_speakers: tuple[str, ...] | None = None

    @property
    def span_length(self) -> float:
        return max(0.0, self.span_end - self.span_start)

    @property
    def play_length(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class ReviewSet:
    """A loaded review set document."""

    audio: str
    duration_s: float
    publisher: str
    detector: str
    transcript: tuple[TranscriptWord, ...]
    marks: tuple[ReviewMark, ...]
    evaluation: dict | None
    parts: tuple[tuple[str, float], ...] = ()

    @property
    def has_reference(self) -> bool:
        """True when the document carries reference information (test mode)."""
        if self.evaluation is not None:
            return True
        return any(m.reference_words is not None for m in self.marks)


@dataclass
class Resolution:
    """What a person recorded for one mark: `status` is open, nothing or text; `speaker` is
    the label the words belong to, an empty string for none, or None to take the speaker the
    lines around the gap suggest."""

    start: float
    end: float
    status: str = STATUS_OPEN
    note: str = ""
    speaker: str | None = None

    @property
    def resolved(self) -> bool:
        return self.status != STATUS_OPEN


def engine_label(record: object) -> str:
    """Join an engine record's engine and model names; "unknown engine" when empty."""
    if not isinstance(record, dict):
        return "unknown engine"
    parts = [str(record.get(key) or "").strip() for key in ("engine", "model")]
    text = " ".join(p for p in parts if p)
    return text or "unknown engine"


def _mark_from_dict(raw: dict) -> ReviewMark:
    speakers = raw.get("reference_speakers")
    ref_words = raw.get("reference_words")
    return ReviewMark(
        start=float(raw["start"]),
        end=float(raw["end"]),
        span_start=float(raw.get("span_start", raw["start"])),
        span_end=float(raw.get("span_end", raw["end"])),
        detector_words=int(raw.get("detector_words", 0)),
        detector_text=str(raw.get("detector_text", "")),
        reference_words=None if ref_words is None else int(ref_words),
        reference_speakers=None if speakers is None else tuple(str(s) for s in speakers),
    )


def review_set_from_dict(doc: dict) -> ReviewSet:
    """Build a ReviewSet from a parsed review-set document; ValueError on a bad schema."""
    schema = doc.get("schema")
    if schema != REVIEW_SCHEMA:
        raise ValueError(f"unexpected review set schema {schema!r}, wanted {REVIEW_SCHEMA!r}")
    transcript = tuple(
        TranscriptWord(float(w["s"]), float(w["e"]), str(w["w"])) for w in doc.get("transcript", [])
    )
    marks = tuple(_mark_from_dict(m) for m in doc.get("marks", []))
    evaluation = doc.get("evaluation")
    return ReviewSet(
        audio=str(doc.get("audio", "")),
        duration_s=float(doc.get("duration_s", 0.0)),
        publisher=engine_label(doc.get("publisher")),
        detector=engine_label(doc.get("detector")),
        transcript=transcript,
        marks=marks,
        evaluation=evaluation if isinstance(evaluation, dict) else None,
        parts=tuple(
            (str(part["audio"]), float(part["offset_s"]))
            for part in (doc.get("parts") or []) if isinstance(part, dict) and "audio" in part
        ),
    )


def load_review_set(path: Path) -> ReviewSet:
    """Read and validate a review set JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict):
        raise ValueError("review set is not a JSON object")
    return review_set_from_dict(doc)


def resolve_audio_path(review_path: Path, audio: str) -> Path:
    """The audio path as given, or relative to the review set's folder when relative."""
    candidate = Path(audio)
    if candidate.is_absolute():
        return candidate
    return review_path.parent / candidate


# ----- session -----------------------------------------------------------------------------


def session_path_for(review_path: Path) -> Path:
    """`<stem>.session.json` beside the review set."""
    return review_path.with_name(f"{review_path.stem}.session.json")


def transcript_path_for(review_path: Path) -> Path:
    """The transcript document beside a review set, named by the same stem."""
    name = review_path.name
    stem = name[: -len(".review.json")] if name.endswith(".review.json") else review_path.stem
    return review_path.with_name(f"{stem}.transcript.json")


def fresh_resolutions(marks: Sequence[ReviewMark]) -> list[Resolution]:
    """One open resolution per mark."""
    return [Resolution(start=m.start, end=m.end) for m in marks]


def session_document(resolutions: Sequence[Resolution]) -> dict:
    """The session JSON document for a list of resolutions."""
    return {"schema": SESSION_SCHEMA, "marks": [asdict(r) for r in resolutions]}


def write_session(path: Path, resolutions: Sequence[Resolution]) -> None:
    """Write the session atomically: a temporary file in the same folder, then a rename."""
    write_json_atomic(session_document(resolutions), path)


def read_session(path: Path, marks: Sequence[ReviewMark]) -> list[Resolution] | None:
    """Load an earlier session for these marks, or None when absent or not matching.

    A session matches when it has one entry per mark and every entry's start and end agree
    with the mark's within a millisecond. Anything else is ignored so a stale file from a
    different review set can never be attached to this one.
    """
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("schema") != SESSION_SCHEMA:
        return None
    entries = doc.get("marks")
    if not isinstance(entries, list) or len(entries) != len(marks):
        return None
    out: list[Resolution] = []
    for entry, mark in zip(entries, marks):
        try:
            start = float(entry["start"])
            end = float(entry["end"])
            status = str(entry.get("status", STATUS_OPEN))
            note = str(entry.get("note", ""))
        except (KeyError, TypeError, ValueError):
            return None
        if abs(start - mark.start) > 1e-3 or abs(end - mark.end) > 1e-3:
            return None
        if status not in (STATUS_OPEN, STATUS_NOTHING, STATUS_TEXT):
            status = STATUS_OPEN
        speaker = entry.get("speaker")
        out.append(Resolution(start=mark.start, end=mark.end, status=status, note=note,
                              speaker=None if speaker is None else str(speaker)))
    return out


def resolutions_from_transcript(transcript_path: Path, marks: Sequence[ReviewMark]) -> list[Resolution] | None:
    """Resolutions seeded from what the transcript document beside the review set records,
    for a pass whose session file is gone; None when there is no such document, it cannot
    be read, or its marks do not match."""
    if not transcript_path.is_file():
        return None
    try:
        doc = load_document(transcript_path)
    except (OSError, ValueError):
        return None
    recorded = resolutions_from_document(doc)
    if len(recorded) != len(marks):
        return None
    out: list[Resolution] = []
    for entry, mark in zip(recorded, marks):
        status = str(entry.get("status", STATUS_OPEN))
        if status not in (STATUS_OPEN, STATUS_NOTHING, STATUS_TEXT):
            status = STATUS_OPEN
        out.append(Resolution(start=mark.start, end=mark.end, status=status, note=str(entry.get("note", "")),
                              speaker=entry.get("speaker")))
    return out


def document_speakers(transcript_path: Path) -> list[tuple[str, str]] | None:
    """(label, name) of the transcript document's speakers beside the review set, in the
    document's order; None when there is no document."""
    if not transcript_path.is_file():
        return None
    try:
        doc = load_document(transcript_path)
    except (OSError, ValueError):
        return None
    names = speaker_names(doc)
    return [(str(entry["label"]), names.get(str(entry["label"]), str(entry["label"])))
            for entry in doc.get("speakers", []) if entry.get("label") is not None]


def document_words(transcript_path: Path) -> list[tuple[float, str | None, str]] | None:
    """(start, speaker label, word) of the engine lines of the transcript document beside
    the review set, in time order; None when there is no document."""
    if not transcript_path.is_file():
        return None
    try:
        doc = load_document(transcript_path)
    except (OSError, ValueError):
        return None
    words: list[tuple[float, str | None, str]] = []
    for line in doc.get("lines", []):
        if line.get("src") == "listener":
            continue
        label = line.get("speaker")
        for word in line.get("words", []):
            if word.get("src") == "listener":
                continue
            text = str(word.get("w", "")).strip()
            if text:
                words.append((float(word.get("s", 0.0)), None if label is None else str(label), text))
    words.sort(key=lambda entry: entry[0])
    return words


# ----- formatting ----------------------------------------------------------------------


def format_mss_t(seconds: float) -> str:
    """Format seconds as m:ss.t with one decimal (65.34 -> "1:05.3")."""
    tenths = max(0, int(round(seconds * 10.0)))
    minutes, rest = divmod(tenths, 600)
    whole, tenth = divmod(rest, 10)
    return f"{minutes}:{whole:02d}.{tenth}"


def shorten(text: str, limit: int) -> str:
    """One line of at most `limit` characters: whitespace collapsed, cut with an ellipsis."""
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: max(1, limit - 3)].rstrip() + "..."


def context_around_gap(words: Sequence[TranscriptWord], span_start: float, span_end: float,
                       each_side: int = CONTEXT_WORDS) -> str:
    """Up to `each_side` published words before and after a gap, with the gap marker between.

    A word belongs before the gap when it starts before the span starts; every other word
    is after it. An ellipsis is shown on a side whose words were cut to `each_side`.
    """
    before = [w.w for w in words if w.s < span_start]
    after = [w.w for w in words if w.s >= span_start]
    parts: list[str] = []
    if len(before) > each_side:
        parts.append("...")
    parts.extend(before[-each_side:] if each_side > 0 else [])
    parts.append(GAP_MARKER)
    parts.extend(after[:each_side])
    if len(after) > each_side:
        parts.append("...")
    return " ".join(parts)


def context_with_speakers(words: Sequence[tuple[float, str | None, str]], names: dict[str, str],
                          span_start: float, each_side: int = CONTEXT_WORDS) -> str:
    """As context_around_gap, from a transcript document's words with their speakers: each
    side opens with its speaker's name, and the name is repeated where the speaker changes,
    so a listener sees who was talking on either side of the gap."""
    before = [w for w in words if w[0] < span_start]
    after = [w for w in words if w[0] >= span_start]
    parts: list[str] = []

    def add(side: list[tuple[float, str | None, str]]) -> None:
        last: object = object()
        for _, label, text in side:
            if label != last:
                parts.append((names.get(label, label) if label is not None else UNLABELLED_NAME) + ":")
                last = label
            parts.append(text)

    if len(before) > each_side:
        parts.append("...")
    add(before[-each_side:] if each_side > 0 else [])
    parts.append(GAP_MARKER)
    add(after[:each_side])
    if len(after) > each_side:
        parts.append("...")
    return " ".join(parts)


def coverage_fraction(marks: Sequence[ReviewMark], duration_s: float) -> float:
    """Share of the recording inside the union of the marks' play spans; 0.0 for no audio."""
    if duration_s <= 0.0 or not marks:
        return 0.0
    spans = sorted((max(0.0, m.start), min(duration_s, m.end)) for m in marks)
    total = 0.0
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += max(0.0, cur_end - cur_start)
            cur_start, cur_end = start, end
    total += max(0.0, cur_end - cur_start)
    return min(1.0, total / duration_s)


def describe_reference(mark: ReviewMark) -> tuple[str, bool]:
    """Text for the test-only panel and whether words were really spoken in the span."""
    if mark.reference_words is None:
        return "No reference information for this mark.", False
    if mark.reference_words <= 0:
        return "False alarm: no reference words fall in this span.", False
    speakers = ", ".join(mark.reference_speakers or ()) or "unknown"
    noun = "word" if mark.reference_words == 1 else "words"
    return (f"{mark.reference_words} reference {noun} fall in this span; "
            f"speakers: {speakers}."), True


def summary_text(review: ReviewSet) -> str:
    """The summary line: mark count, coverage, and the evaluation when present."""
    n = len(review.marks)
    noun = "mark" if n == 1 else "marks"
    share = coverage_fraction(review.marks, review.duration_s) * 100.0
    text = f"{n} {noun} covering {share:.1f}% of the recording"
    ev = review.evaluation
    if ev:
        on_speech = ev.get("marks_on_speech")
        dropped = ev.get("dropped_words")
        covered = ev.get("dropped_covered")
        if on_speech is not None:
            text += f"; {on_speech} of {n} on real speech"
        if dropped is not None and covered is not None:
            recall = 100.0 * covered / dropped if dropped else 0.0
            text += f"; marks cover {recall:.0f}% of dropped words ({covered} of {dropped})"
    return text


def progress_text(done: int, total: int) -> str:
    """How far the pass has come: "2 of 40 checked"."""
    return f"{done} of {total} checked"


def header_text(review: ReviewSet, audio_path: Path) -> str:
    """The header line: file name, length, and both engines."""
    return (f"{audio_path.name}   |   {format_mss(review.duration_s)}   |   published by {review.publisher}"
            f"   |   checked against {review.detector}, which is never published")


def hint_text(mark: ReviewMark, resolution: Resolution | None = None) -> str:
    """The caption over the words box: what the second engine heard in the span, as the
    starting point, and what becomes of the words that are kept; for a span recorded as
    silent, that it was."""
    if resolution is not None and resolution.status == STATUS_NOTHING:
        return "Recorded as nothing said. Type words and keep them to change that, or press O to reopen the mark."
    if mark.detector_text:
        return (f'Starts from what the second engine heard: "{mark.detector_text}". '
                f"Edit it to what is actually said; {CAPTION_TAIL}")
    return f"The second engine recorded no text here. Type what is said; {CAPTION_TAIL}"


def mark_row_text(index: int, mark: ReviewMark, resolution: Resolution | None = None) -> str:
    """List row: tick prefix once resolved, index, start as m:ss.t, span length, then the
    detector word count while the mark is open, else what the listener recorded."""
    resolved = resolution is not None and resolution.resolved
    prefix = TICK if resolved else " "
    if resolved and resolution.status == STATUS_NOTHING:
        tail = "nothing said"
    elif resolved:
        tail = f'"{shorten(resolution.note, ROW_NOTE_CHARS)}"'
    else:
        noun = "word" if mark.detector_words == 1 else "words"
        tail = f"{mark.detector_words} {noun}"
    return (f"{prefix} {index + 1:>3}   {format_mss_t(mark.start):>8}   "
            f"{mark.span_length:.1f} s   {tail}")


FOOTER_TEXT = ("Space play or pause   |   J next   |   K previous   |   Enter play span   |   "
               "N nothing was said   |   T type what was said   |   Ctrl+Enter keep the words   |   "
               "O reopen   |   Esc back to the list   |   Left and Right nudge 5 s")


# ----- the window ----------------------------------------------------------------------


class VerifyWindow(QMainWindow):
    """Main window of the verification screen.

    `audio_device`, `volume` and `rate` are the output device, volume and speed to play with,
    so the screen sounds as the player bar that opened it. `transcript_changed` carries the
    revised transcript document after every write and `mark_written` the index of the mark
    just decided, so the window that opened the screen can show the listener's lines while
    the screen is open.
    """

    transcript_changed = Signal(object)
    mark_written = Signal(int)

    def __init__(self, review: ReviewSet, review_path: Path, theme: Theme | None = None,
                 parent: QWidget | None = None, author: str = "",
                 audio_device: QAudioDevice | None = None, volume: float = 1.0, rate: float = 1.0) -> None:
        super().__init__(parent)
        self.review = review
        self.review_path = Path(review_path)
        self.author = author
        self.audio_device = audio_device
        self.volume = float(volume)
        self.rate = float(rate)
        self.session_path = session_path_for(self.review_path)
        self.theme = theme if theme is not None else theme_for(False)
        self.setWindowIcon(app_icon(self.theme.dark))
        self.audio_path = resolve_audio_path(self.review_path, review.audio)
        self.audio_available = self.audio_path.is_file()

        # The transcript document beside the review set supplies the speakers and, when the
        # session file is gone, the decisions already written into it.
        self._speakers = document_speakers(self.transcript_path()) or []
        self._names = {label: name for label, name in self._speakers}
        self._document_words = document_words(self.transcript_path())

        self.resolutions: list[Resolution] = fresh_resolutions(review.marks)
        self._resumed = False
        self._resumed_from = ""
        self._stale_session = False
        earlier = read_session(self.session_path, review.marks)
        if earlier is None and self.session_path.is_file():
            self._stale_session = True
        if earlier is None:
            earlier = resolutions_from_transcript(self.transcript_path(), review.marks)
            if earlier is not None and any(r.resolved for r in earlier):
                self._resumed_from = "the transcript document"
            else:
                earlier = None
        elif any(r.resolved for r in earlier):
            self._resumed_from = "the session file"
        if earlier is not None:
            self.resolutions = earlier
            self._resumed = True

        self._position_s = 0.0
        self._stop_at_s: float | None = None
        self._auto_paused = False
        self._current: int | None = None
        self._session_error: str | None = None
        # Selecting a mark plays its span, except the first selection when the screen opens.
        self._play_on_select = False

        self._build_widgets()
        self._build_player()
        self._populate()

        if review.marks:
            first_open = next((i for i, r in enumerate(self.resolutions) if not r.resolved), 0)
            self.mark_list.setCurrentRow(first_open)
        self._play_on_select = True
        self._refresh_title()
        self._refresh_summary()
        self._initial_status()

    # ----- construction ---------------------------------------------------------------

    def _build_widgets(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 8)
        root.setSpacing(8)

        self.header_label = QLabel(header_text(self.review, self.audio_path), central)
        self.header_label.setWordWrap(True)
        root.addWidget(self.header_label)

        self.summary_label = QLabel(summary_text(self.review), central)
        self.summary_label.setWordWrap(True)
        root.addWidget(self.summary_label)

        self.timeline = Timeline(central)
        self.timeline.set_colours(self.theme.accent, self.theme.success, self.theme.track,
                                  self.theme.outline)
        self.timeline.set_duration(self.review.duration_s)
        self.timeline.seek_requested.connect(self._on_timeline_seek)
        root.addWidget(self.timeline)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, central)
        root.addWidget(self.splitter, 1)

        self.mark_list = QListWidget(self.splitter)
        mono = QFont(self.mark_list.font())
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setFamily("Consolas")
        self.mark_list.setFont(mono)
        self.mark_list.currentRowChanged.connect(self._on_current_row_changed)
        self.splitter.addWidget(self.mark_list)

        right = QWidget(self.splitter)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(8)

        self.span_label = QLabel("", right)
        bold = QFont(self.span_label.font())
        bold.setBold(True)
        bold.setPointSize(bold.pointSize() + 2)
        self.span_label.setFont(bold)
        right_layout.addWidget(self.span_label)

        context_box = QGroupBox("What the published transcript has here", right)
        context_layout = QVBoxLayout(context_box)
        self.context_panel = QPlainTextEdit(context_box)
        self.context_panel.setReadOnly(True)
        self.context_panel.setMinimumHeight(70)
        context_layout.addWidget(self.context_panel)
        right_layout.addWidget(context_box, 2)

        words_box = QGroupBox("What was said here, in the listener's words", right)
        words_layout = QVBoxLayout(words_box)
        self.hint_label = QLabel("", words_box)
        self.hint_label.setObjectName("muted")
        self.hint_label.setWordWrap(True)
        # Muted through the palette as well as the stylesheet, so the caption reads as
        # secondary when the screen runs on its own, without the application stylesheet.
        caption = QPalette(self.hint_label.palette())
        caption.setColor(QPalette.ColorRole.WindowText, self.theme.muted)
        self.hint_label.setPalette(caption)
        words_layout.addWidget(self.hint_label)
        self.words_edit = QPlainTextEdit(words_box)
        self.words_edit.setPlaceholderText("Type what was said, or press N when nothing was said.")
        self.words_edit.setMinimumHeight(60)
        self.words_edit.setTabChangesFocus(True)
        words_layout.addWidget(self.words_edit)
        speaker_row = QHBoxLayout()
        speaker_row.addWidget(QLabel("Spoken by", words_box))
        self.speaker_box = QComboBox(words_box)
        self.speaker_box.setToolTip(
            "Whose words these are; by default the speaker the lines around the gap suggest, "
            "which a gap between two speakers leaves undecided"
        )
        self.speaker_box.addItem(SPEAKER_INFERRED, None)
        for label, name in self._speakers:
            self.speaker_box.addItem(name, label)
        self.speaker_box.addItem(SPEAKER_NONE, "")
        speaker_row.addWidget(self.speaker_box, 1)
        words_layout.addLayout(speaker_row)
        if not self._speakers:
            self.speaker_box.hide()
        right_layout.addWidget(words_box, 2)

        self.reference_panel: QGroupBox | None = None
        self.reference_label: QLabel | None = None
        if self.review.has_reference:
            self.reference_panel = QGroupBox(
                "Test only: reference information, not shown in normal use", right)
            ref_layout = QVBoxLayout(self.reference_panel)
            self.reference_label = QLabel("", self.reference_panel)
            self.reference_label.setWordWrap(True)
            ref_layout.addWidget(self.reference_label)
            right_layout.addWidget(self.reference_panel)

        buttons = QHBoxLayout()
        self.play_button = QPushButton("Play this span (Enter)", right)
        self.nothing_button = QPushButton("Nothing was said (N)", right)
        self.keep_button = QPushButton("Keep these words (Ctrl+Enter)", right)
        self.keep_button.setToolTip(
            "Record the words above as what was said in this span; they go into the transcript "
            "at once, into the line where the gap is, marked as the listener's"
        )
        self.reopen_button = QPushButton("Reopen (O)", right)
        self.reopen_button.setToolTip(
            "Set this mark back to open; the listener's words for it leave the transcript"
        )
        for button in (self.play_button, self.nothing_button, self.keep_button, self.reopen_button):
            button.setAutoDefault(False)
            button.setDefault(False)
            buttons.addWidget(button)
        self.play_button.clicked.connect(self.play_span)
        self.nothing_button.clicked.connect(self.resolve_nothing)
        self.keep_button.clicked.connect(self.keep_words)
        self.reopen_button.clicked.connect(self.reopen)
        right_layout.addLayout(buttons)

        self.status_label = QLabel("", right)
        self.status_label.setWordWrap(True)
        right_layout.addWidget(self.status_label)

        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([340, 700])

        self.footer_label = QLabel(FOOTER_TEXT, central)
        self.footer_label.setWordWrap(True)
        root.addWidget(self.footer_label)

        self.setCentralWidget(central)

        # Children that take focus consume keys before the window sees them (a list view
        # turns letters into a keyboard search, a text panel scrolls on Space), so the
        # window filters key presses on each of them. The words box is filtered too, but
        # gives up only Ctrl+Enter and Esc: every other key types into it. A button keeps
        # Space and Enter, which press it.
        for widget in (self.mark_list, self.context_panel, self.words_edit, self.play_button,
                       self.nothing_button, self.keep_button, self.reopen_button, self.timeline):
            widget.installEventFilter(self)
        self.mark_list.setFocus()

    def _build_player(self) -> None:
        self.player: QMediaPlayer | None = None
        self.audio_output: QAudioOutput | None = None
        self._player_error: str | None = None
        self._copy_started = False
        self._copy: PlayableCopy | None = None
        try:
            self.player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            if self.audio_device is not None:
                self.audio_output.setDevice(self.audio_device)
            self.audio_output.setVolume(min(1.0, max(0.0, self.volume)))
            self.player.setAudioOutput(self.audio_output)
            self.player.setPlaybackRate(self.rate)
            self.player.positionChanged.connect(self._on_position_changed)
            self.player.errorOccurred.connect(self._on_player_error)
            self.player.playbackStateChanged.connect(self._on_playback_state_changed)
            self.player.mediaStatusChanged.connect(self._on_media_status_changed)
            if self.audio_available:
                # A copy made earlier for a recording the player cannot read is played instead.
                copy = playable_copy_path(self.audio_path)
                source = copy if copy_is_current(self.audio_path, copy) else self.audio_path
                self.player.setSource(QUrl.fromLocalFile(str(source)))
        except Exception as exc:  # noqa: BLE001 (any multimedia failure disables playback)
            self.player = None
            self.audio_output = None
            self._player_error = f"{type(exc).__name__}: {exc}"
            self.audio_available = False

    def _populate(self) -> None:
        self.timeline.set_marks([(m.start, m.end) for m in self.review.marks])
        self.mark_list.clear()
        for index, mark in enumerate(self.review.marks):
            self.mark_list.addItem(QListWidgetItem(mark_row_text(index, mark, self.resolutions[index])))
            self.timeline.set_resolved(index, self.resolutions[index].resolved)
        if not self.review.marks:
            self.span_label.setText("No marks: the published transcript has no silent spans to check.")
            self.context_panel.setPlainText("")
            self.hint_label.setText("")
            self.words_edit.setPlainText("")
            for widget in (self.play_button, self.nothing_button, self.keep_button,
                           self.reopen_button, self.words_edit, self.speaker_box):
                widget.setEnabled(False)

    def _initial_status(self) -> None:
        parts: list[str] = []
        if self._player_error is not None:
            parts.append(f"Playback is disabled: {self._player_error}.")
        elif not self.audio_available:
            parts.append(f"Audio file not found: {self.audio_path.name}. "
                         "Playback controls do nothing; resolutions are still recorded.")
        else:
            parts.append(f"Audio loaded: {self.audio_path.name}.")
        if self._stale_session:
            parts.append("An earlier session file did not match these marks and was set aside.")
        if self._resumed and self.done_count() > 0:
            parts.append(f"Resumed {self.done_count()} of {len(self.resolutions)} earlier decisions "
                         f"from {self._resumed_from}.")
        if self.review.marks and not self.transcript_path().is_file():
            parts.append("No transcript document beside the review set: decisions go to the "
                         "session file only.")
        self.set_status(" ".join(parts))

    # ----- properties -----------------------------------------------------------------

    @property
    def current_index(self) -> int | None:
        """Index of the current mark, or None when there are no marks."""
        return self._current

    @property
    def current_mark(self) -> ReviewMark | None:
        if self._current is None:
            return None
        return self.review.marks[self._current]

    @property
    def position_s(self) -> float:
        """Playhead position in seconds, tracked even without audio."""
        return self._position_s

    def done_count(self) -> int:
        """Number of resolved marks."""
        return sum(1 for r in self.resolutions if r.resolved)

    def is_playing(self) -> bool:
        return (self.player is not None
                and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState)

    def set_status(self, text: str) -> None:
        """Set the status line."""
        self.status_label.setText(text)

    # ----- selection ------------------------------------------------------------------

    def select_mark(self, index: int) -> None:
        """Make the mark at `index` current (clamped); selecting seeks to its start and, once
        the screen has opened, plays its span."""
        if not self.review.marks:
            return
        index = min(max(0, index), len(self.review.marks) - 1)
        if self.mark_list.currentRow() == index:
            self._show_mark(index)
        else:
            self.mark_list.setCurrentRow(index)

    def select_next(self) -> None:
        """J: the next mark."""
        if self._current is not None:
            self.select_mark(self._current + 1)

    def select_previous(self) -> None:
        """K: the previous mark."""
        if self._current is not None:
            self.select_mark(self._current - 1)

    def _on_current_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.review.marks):
            self._current = None
            self.timeline.set_current(None)
            return
        self._show_mark(row, play=self._play_on_select)

    def _show_mark(self, index: int, play: bool = False) -> None:
        mark = self.review.marks[index]
        self._current = index
        self.timeline.set_current(index)
        self.span_label.setText(
            f"{format_mss_t(mark.start)} to {format_mss_t(mark.end)} ({mark.play_length:.1f} seconds)")
        if self._document_words:
            self.context_panel.setPlainText(
                context_with_speakers(self._document_words, self._names, mark.span_start))
        else:
            self.context_panel.setPlainText(
                context_around_gap(self.review.transcript, mark.span_start, mark.span_end))
        self.hint_label.setText(hint_text(mark, self.resolutions[index]))
        self._fill_editor(index)
        if self.reference_label is not None:
            text, spoken = describe_reference(mark)
            self.reference_label.setText(text)
            palette = QPalette(self.reference_label.palette())
            colour = self.theme.warning if spoken else self.palette().color(QPalette.ColorRole.WindowText)
            palette.setColor(QPalette.ColorRole.WindowText, colour)
            self.reference_label.setPalette(palette)
            font = QFont(self.reference_label.font())
            font.setBold(spoken)
            self.reference_label.setFont(font)
        self._stop_at_s = None
        self._seek(mark.start)
        if play and self.player is not None and self.audio_available:
            self.play_span(announce=False)

    def _fill_editor(self, index: int) -> None:
        """The words box for a mark: the listener's words when there are any, nothing for a
        span resolved as silent, else what the second engine heard, to be edited; the speaker
        box shows the speaker the words were given."""
        resolution = self.resolutions[index]
        if resolution.status == STATUS_TEXT:
            text = resolution.note
        elif resolution.status == STATUS_NOTHING:
            text = ""
        else:
            text = self.review.marks[index].detector_text
        self.words_edit.setPlainText(text)
        if self.words_edit.hasFocus():
            self.words_edit.selectAll()
        position = self.speaker_box.findData(resolution.speaker) if resolution.speaker is not None else 0
        self.speaker_box.setCurrentIndex(position if position >= 0 else 0)

    # ----- playback -------------------------------------------------------------------

    def _seek(self, seconds: float) -> None:
        seconds = min(max(0.0, seconds), self.review.duration_s) if self.review.duration_s > 0 else max(0.0, seconds)
        self._position_s = seconds
        self.timeline.set_playhead(seconds)
        if self.player is not None and self.audio_available:
            self.player.setPosition(int(round(seconds * 1000.0)))

    def seek(self, seconds: float) -> None:
        """Move the playhead; a plain seek cancels any pending stop at a span end."""
        self._stop_at_s = None
        self._seek(seconds)

    def _on_timeline_seek(self, seconds: float) -> None:
        self.seek(seconds)
        self.set_status(f"Playhead at {format_mss_t(seconds)}.")

    def nudge(self, delta_s: float) -> None:
        """Move the playhead by `delta_s` seconds (Left and Right keys)."""
        self.seek(self._position_s + delta_s)
        self.set_status(f"Playhead at {format_mss_t(self._position_s)}.")

    def toggle_play(self) -> None:
        """Space: play or pause; free playback does not stop at a span end."""
        if not self._playback_possible():
            return
        assert self.player is not None
        self._stop_at_s = None
        if self.is_playing():
            self.player.pause()
        else:
            self.player.play()

    def play_span(self, announce: bool = True) -> None:
        """Enter: seek to the current mark's start, play, pause at its end. With `announce`
        the status line says so; a span played on selection leaves the status line alone."""
        mark = self.current_mark
        if mark is None:
            return
        self._seek(mark.start)
        if not self._playback_possible():
            return
        assert self.player is not None
        self._stop_at_s = mark.end
        self.player.play()
        if announce:
            self.set_status(f"Playing {format_mss_t(mark.start)} to {format_mss_t(mark.end)}.")

    def _playback_possible(self) -> bool:
        if self.player is None or not self.audio_available:
            if self._player_error is not None:
                self.set_status(f"Playback is disabled: {self._player_error}.")
            else:
                self.set_status(f"Audio file not found: {self.audio_path.name}. Nothing to play.")
            return False
        return True

    def _on_position_changed(self, position_ms: int) -> None:
        seconds = position_ms / 1000.0
        self._position_s = seconds
        self.timeline.set_playhead(seconds)
        if self._stop_at_s is not None and seconds >= self._stop_at_s:
            self._stop_at_s = None
            if self.player is not None:
                # The pause at a span end is the screen's own; it leaves the status line, and
                # what it says about the last decision, alone.
                self._auto_paused = True
                self.player.pause()

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._auto_paused = False
            return
        if self._auto_paused:
            self._auto_paused = False
            return
        if self._stop_at_s is None and self.current_mark is not None and self.audio_available:
            self.set_status(f"Paused at {format_mss_t(self._position_s)}.")

    def _on_player_error(self, error: QMediaPlayer.Error, message: str) -> None:
        if error == QMediaPlayer.Error.NoError:
            return
        if self._start_playable_copy(message or str(error)):
            return
        self._player_error = message or str(error)
        self.audio_available = False
        self.set_status(f"Playback is disabled: {self._player_error}.")

    def _on_media_status_changed(self, status) -> None:
        if self.player is None or status != QMediaPlayer.MediaStatus.LoadedMedia:
            return
        if self.player.hasVideo() and not self.player.hasAudio():
            # A picture the player shows without a sound it can decode: play a decoded copy.
            self._start_playable_copy("no sound the player can decode")

    # ----- a playable copy ------------------------------------------------------------------

    def _start_playable_copy(self, reason: str) -> bool:
        """Decode the recording to a copy the player can read, once; returns whether a copy
        was started. A copy that fails in the player is not copied again."""
        if self.player is None or self._copy_started or self.audio_path.parent == play_dir():
            return False
        self._copy_started = True
        parts = None
        if self.review.parts:
            parts = Parts(
                tuple(resolve_audio_path(self.review_path, audio) for audio, _ in self.review.parts),
                tuple(offset for _, offset in self.review.parts),
            )
        self._copy = PlayableCopy(self.audio_path, parts, self)
        self._copy.ready.connect(self._on_playable_copy_ready)
        self._copy.failed.connect(self._on_playable_copy_failed)
        self.set_status(f"The player cannot read this recording ({reason}); decoding a copy to play.")
        self._copy.start()
        return True

    def _on_playable_copy_ready(self, copy: object) -> None:
        if self.player is None:
            return
        self._player_error = None
        self.audio_available = True
        self.player.setSource(QUrl.fromLocalFile(str(copy)))
        self.set_status("Playing a decoded copy of the recording.")

    def _on_playable_copy_failed(self, reason: str) -> None:
        self._player_error = reason
        self.audio_available = False
        self.set_status(f"Playback is disabled: {reason}.")

    # ----- resolving ------------------------------------------------------------------

    def resolve_nothing(self) -> None:
        """N: record that nothing was said in the current span, then advance."""
        if self._current is None:
            return
        self._resolve(self._current, STATUS_NOTHING, "")

    def edit_words(self) -> None:
        """T: put the cursor in the words box with its text selected, to type over it."""
        if self._current is None:
            return
        self.words_edit.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.words_edit.selectAll()
        self.set_status("Type what was said, then Ctrl+Enter or Keep these words; Esc goes back to the list.")

    def keep_words(self) -> None:
        """Ctrl+Enter: record the words in the box as what was said in the current span, for
        the speaker chosen in the box, then advance."""
        if self._current is None:
            return
        note = self.words_edit.toPlainText().strip()
        if not note:
            self.set_status("Nothing typed: type what was said, or press N when nothing was said.")
            return
        chosen = self.speaker_box.currentData() if self.speaker_box.isVisible() or self._speakers else None
        self._resolve(self._current, STATUS_TEXT, note, speaker=chosen)

    def reopen(self) -> None:
        """O: set the current mark back to open; the listener's words for it leave the
        transcript."""
        if self._current is None:
            return
        if not self.resolutions[self._current].resolved:
            self.set_status(f"Mark {self._current + 1} is open already.")
            return
        self._resolve(self._current, STATUS_OPEN, "", advance=False)

    def _resolve(self, index: int, status: str, note: str, advance: bool = True,
                 speaker: str | None = None) -> None:
        resolution = self.resolutions[index]
        resolution.status = status
        resolution.note = note
        resolution.speaker = speaker if status == STATUS_TEXT else None
        self._refresh_row(index)
        self._write_session()
        revised, written = self._apply()
        self._refresh_title()
        self._refresh_summary()
        if status == STATUS_NOTHING:
            what = f"Mark {index + 1}: nothing was said."
        elif status == STATUS_TEXT:
            what = f'Mark {index + 1}: "{shorten(note, STATUS_NOTE_CHARS)}"{self._spoken_by(revised, index)}.'
        else:
            what = f"Mark {index + 1} is open again."
        if self._session_error is not None:
            written = f"Could not write the session file: {self._session_error}. {written}"
        self.set_status(f"{what} {written}")
        if revised is not None:
            self.mark_written.emit(index)
        # After a decision the keys named in the footer act again, wherever the cursor was.
        if self.words_edit.hasFocus():
            self.mark_list.setFocus(Qt.FocusReason.OtherFocusReason)
        if advance and index + 1 < len(self.review.marks):
            self.select_mark(index + 1)
        else:
            self.hint_label.setText(hint_text(self.review.marks[index], self.resolutions[index]))
            self._fill_editor(index)

    def _spoken_by(self, revised: dict | None, index: int) -> str:
        """", as Speaker 2" or ", without a speaker" for the words just written."""
        if revised is None:
            return ""
        line = listener_line_for(revised, index)
        if line is None:
            return ""
        label = line.get("speaker")
        if label is None:
            return ", without a speaker"
        return f", as {self._names.get(str(label), str(label))}"

    def _refresh_row(self, index: int) -> None:
        item = self.mark_list.item(index)
        if item is not None:
            item.setText(mark_row_text(index, self.review.marks[index], self.resolutions[index]))
        self.timeline.set_resolved(index, self.resolutions[index].resolved)

    def transcript_path(self) -> Path:
        """The transcript document beside the review set, named by the same stem."""
        return transcript_path_for(self.review_path)

    def _apply(self) -> tuple[dict | None, str]:
        """Put the session's resolutions into the transcript document beside the review set
        and render its text, Word and subtitle outputs again. Returns the revised document,
        or None, and a sentence saying what happened; an output that could not be written
        (a Word document open in Word) is named, and the document still counts as written."""
        from twinscribe.amend import apply_session
        from twinscribe.outputs import describe_failures, render_outputs
        from twinscribe.pipeline import output_paths

        target = self.transcript_path()
        if not target.is_file():
            return None, (f"No transcript document beside the review set ({target.name}); "
                          "the session file keeps the decisions.")
        try:
            revised = apply_session(target, self.session_path, author=self.author, render=False)
        except (OSError, ValueError) as exc:
            return None, f"Could not write the transcript: {exc}"
        self.transcript_changed.emit(revised)
        stem = target.name[: -len(".transcript.json")]
        paths = output_paths(target.parent / stem, target.parent)
        failures = render_outputs(revised, paths.text, paths.docx, paths.subtitles, author=self.author)
        applied = revised.get("review_applied") or {}
        with_words = int(applied.get("text", 0))
        silent = int(applied.get("nothing", 0))
        open_count = int(applied.get("open", 0))
        noun = "span" if with_words == 1 else "spans"
        sentence = (f"Transcript written: {with_words} {noun} with the listener's words, {silent} silent, "
                    f"{open_count} still open; ")
        if failures:
            sentence += describe_failures(failures) + "; the transcript is written and the file is written again on the next decision."
        else:
            sentence += "the text, Word and subtitle files are written again."
        return revised, sentence

    def apply_to_transcript(self) -> dict | None:
        """Write the session and the transcript again from the resolutions held; the status
        line says what happened. Returns the revised document, or None."""
        if not self.review.marks:
            self.set_status("Nothing to apply: the review list is empty.")
            return None
        self._write_session()
        revised, sentence = self._apply()
        self.set_status(sentence)
        return revised

    def _write_session(self) -> None:
        try:
            write_session(self.session_path, self.resolutions)
            self._session_error = None
        except OSError as exc:
            self._session_error = str(exc)
            self.set_status(f"Could not write the session file: {exc}")

    def _refresh_title(self) -> None:
        total = len(self.review.marks)
        self.setWindowTitle(f"{self.done_count()} of {total} done  |  TwinScribe verify  |  "
                            f"{self.audio_path.name}")

    def _refresh_summary(self) -> None:
        text = summary_text(self.review)
        if self.review.marks:
            text += f"   |   {progress_text(self.done_count(), len(self.review.marks))}"
        self.summary_label.setText(text)

    # ----- keys -----------------------------------------------------------------------

    def handle_key(self, key: int, modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier) -> bool:
        """Apply one of the screen's keys; True when the key was one of them."""
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.keep_words()
            return True
        if modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
            return False
        if key == Qt.Key.Key_Space:
            self.toggle_play()
        elif key == Qt.Key.Key_J:
            self.select_next()
        elif key == Qt.Key.Key_K:
            self.select_previous()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.play_span()
        elif key == Qt.Key.Key_N:
            self.resolve_nothing()
        elif key == Qt.Key.Key_T:
            self.edit_words()
        elif key == Qt.Key.Key_O:
            self.reopen()
        elif key == Qt.Key.Key_Left:
            self.nudge(-NUDGE_S)
        elif key == Qt.Key.Key_Right:
            self.nudge(NUDGE_S)
        else:
            return False
        return True

    def _editor_key(self, key: int, modifiers: Qt.KeyboardModifier) -> bool:
        """The keys the words box gives up: Ctrl+Enter keeps the words, Esc returns to the
        list; every other key types into the box."""
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.keep_words()
            return True
        if key == Qt.Key.Key_Escape:
            self.mark_list.setFocus(Qt.FocusReason.ShortcutFocusReason)
            self.set_status("Back at the list; the words in the box are kept only with Ctrl+Enter or the button.")
            return True
        return False

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt virtual)
        if self.handle_key(event.key(), event.modifiers()):
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt virtual)
        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            if watched is self.words_edit:
                return self._editor_key(event.key(), event.modifiers())
            if isinstance(watched, QAbstractButton) and event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter) \
                    and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                return super().eventFilter(watched, event)
            if self.handle_key(event.key(), event.modifiers()):
                return True
        return super().eventFilter(watched, event)

    # ----- lifecycle ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt virtual)
        if self.player is not None:
            self._stop_at_s = None
            self.player.stop()
            self.player.setSource(QUrl())
        # Written on close only when there is something to keep, so opening a review set
        # and closing it (or a screenshot run) leaves no empty session file behind.
        if self.review.marks and (self.done_count() > 0 or self.session_path.exists()):
            self._write_session()
        super().closeEvent(event)


# ----- entry point ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Command line: review set path, optional dark theme, optional screenshot."""
    parser = argparse.ArgumentParser(
        prog="python -m twinscribe.app.verify",
        description="Work through a twinscribe review list with the audio at each mark.")
    parser.add_argument("review_set", type=Path, help="review set JSON (twinscribe.review.v1)")
    parser.add_argument("--dark", action="store_true", help="use the dark palette")
    parser.add_argument("--shot", type=Path, default=None, metavar="OUT_PNG",
                        help="render the window, save a screenshot after a short delay, exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the screen; returns the process exit status."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    review_path: Path = args.review_set
    try:
        review = load_review_set(review_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"cannot read review set {review_path}: {exc}", file=sys.stderr)
        return 2

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    theme = apply_theme(app, dark=bool(args.dark))
    window = VerifyWindow(review, review_path, theme)
    window.resize(1120, 740)
    window.show()

    if args.shot is not None:
        shot_path: Path = args.shot

        def take_shot() -> None:
            saved = window.grab().save(str(shot_path))
            if not saved:
                print(f"could not save screenshot to {shot_path}", file=sys.stderr)
            window.close()
            app.quit()

        QTimer.singleShot(SHOT_DELAY_MS, take_shot)

    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
