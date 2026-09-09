"""The verification screen: work through the review list with the audio at each mark.

A review set (schema `twinscribe.review.v1`) lists the spans where the published engine
heard nothing and the second engine heard speech. This screen plays each span, shows the
published transcript either side of the gap and what the second engine heard, and records
for every mark either that nothing was said or what was said. Resolutions go to
`<review_set stem>.session.json` beside the review set, written on every resolution and on
close.

Entry point: `python -m twinscribe.app.verify <review_set.json> [--dark] [--shot out.png]`.

Screenshots (`--shot`) must be taken on the platform's real backend. Under
`QT_QPA_PLATFORM=offscreen` there is no font database and every glyph renders as a box, so
a screenshot taken there shows nothing useful.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, QUrl
from PySide6.QtGui import QCloseEvent, QFont, QKeyEvent, QPalette
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from twinscribe.app.app_icon import app_icon
from twinscribe.app.theme import Theme, apply_theme, theme_for
from twinscribe.app.timeline import Timeline, format_mss

REVIEW_SCHEMA = "twinscribe.review.v1"
SESSION_SCHEMA = "twinscribe.review-session.v1"
CONTEXT_WORDS = 14
GAP_MARKER = "[ GAP ]"
NUDGE_S = 5.0
SHOT_DELAY_MS = 800
TICK = "\u2713"  # check mark shown before a resolved row

STATUS_OPEN = "open"
STATUS_NOTHING = "nothing"
STATUS_TEXT = "text"

TextPrompt = Callable[[str], "str | None"]


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

    @property
    def has_reference(self) -> bool:
        """True when the document carries reference information (test mode)."""
        if self.evaluation is not None:
            return True
        return any(m.reference_words is not None for m in self.marks)


@dataclass
class Resolution:
    """What a person recorded for one mark; `status` is open, nothing or text."""

    start: float
    end: float
    status: str = STATUS_OPEN
    note: str = ""

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


def fresh_resolutions(marks: Sequence[ReviewMark]) -> list[Resolution]:
    """One open resolution per mark."""
    return [Resolution(start=m.start, end=m.end) for m in marks]


def session_document(resolutions: Sequence[Resolution]) -> dict:
    """The session JSON document for a list of resolutions."""
    return {"schema": SESSION_SCHEMA, "marks": [asdict(r) for r in resolutions]}


def write_session(path: Path, resolutions: Sequence[Resolution]) -> None:
    """Write the session atomically: a temporary file in the same folder, then a rename."""
    payload = json.dumps(session_document(resolutions), indent=2, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


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
        out.append(Resolution(start=mark.start, end=mark.end, status=status, note=note))
    return out


# ----- formatting ----------------------------------------------------------------------


def format_mss_t(seconds: float) -> str:
    """Format seconds as m:ss.t with one decimal (65.34 -> "1:05.3")."""
    tenths = max(0, int(round(seconds * 10.0)))
    minutes, rest = divmod(tenths, 600)
    whole, tenth = divmod(rest, 10)
    return f"{minutes}:{whole:02d}.{tenth}"


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


def header_text(review: ReviewSet, audio_path: Path) -> str:
    """The header line: file name, minutes, and both engines."""
    minutes = review.duration_s / 60.0
    return (f"{audio_path.name}   |   {minutes:.1f} min   |   published by {review.publisher}"
            f"   |   checked against {review.detector}, which is never published")


def mark_row_text(index: int, mark: ReviewMark, resolved: bool) -> str:
    """List row: tick prefix, index, start as m:ss.t, span length, detector words."""
    prefix = TICK if resolved else " "
    noun = "word" if mark.detector_words == 1 else "words"
    return (f"{prefix} {index + 1:>3}   {format_mss_t(mark.start):>8}   "
            f"{mark.span_length:.1f} s   {mark.detector_words} {noun}")


FOOTER_TEXT = ("Space play or pause   |   J next   |   K previous   |   Enter play span   |   "
               "N nothing was said   |   T type what was said   |   Left and Right nudge 5 s")


# ----- the window ----------------------------------------------------------------------


class VerifyWindow(QMainWindow):
    """Main window of the verification screen.

    `text_prompt` is the function used by the T action; it takes the earlier note and
    returns the typed text, or None when cancelled. Tests replace it to avoid a dialog.
    """

    def __init__(self, review: ReviewSet, review_path: Path, theme: Theme | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.review = review
        self.review_path = Path(review_path)
        self.session_path = session_path_for(self.review_path)
        self.theme = theme if theme is not None else theme_for(False)
        self.setWindowIcon(app_icon(self.theme.dark))
        self.audio_path = resolve_audio_path(self.review_path, review.audio)
        self.audio_available = self.audio_path.is_file()
        self.text_prompt: TextPrompt = self._default_text_prompt

        self.resolutions: list[Resolution] = fresh_resolutions(review.marks)
        self._resumed = False
        earlier = read_session(self.session_path, review.marks)
        if earlier is not None:
            self.resolutions = earlier
            self._resumed = True

        self._position_s = 0.0
        self._stop_at_s: float | None = None
        self._current: int | None = None
        self._session_error: str | None = None

        self._build_widgets()
        self._build_player()
        self._populate()

        if review.marks:
            self.mark_list.setCurrentRow(0)
        self._refresh_title()
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

        hint_box = QGroupBox("What the second engine heard, as a hint of what to listen for", right)
        hint_layout = QVBoxLayout(hint_box)
        self.hint_panel = QPlainTextEdit(hint_box)
        self.hint_panel.setReadOnly(True)
        self.hint_panel.setMinimumHeight(60)
        hint_layout.addWidget(self.hint_panel)
        right_layout.addWidget(hint_box, 2)

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
        self.text_button = QPushButton("Type what was said (T)", right)
        for button in (self.play_button, self.nothing_button, self.text_button):
            button.setAutoDefault(False)
            button.setDefault(False)
            buttons.addWidget(button)
        self.play_button.clicked.connect(self.play_span)
        self.nothing_button.clicked.connect(self.resolve_nothing)
        self.text_button.clicked.connect(self.resolve_text)
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
        # window filters key presses on each of them.
        for widget in (self.mark_list, self.context_panel, self.hint_panel, self.play_button,
                       self.nothing_button, self.text_button, self.timeline):
            widget.installEventFilter(self)
        self.mark_list.setFocus()

    def _build_player(self) -> None:
        self.player: QMediaPlayer | None = None
        self.audio_output: QAudioOutput | None = None
        self._player_error: str | None = None
        try:
            self.player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.player.setAudioOutput(self.audio_output)
            self.player.positionChanged.connect(self._on_position_changed)
            self.player.errorOccurred.connect(self._on_player_error)
            self.player.playbackStateChanged.connect(self._on_playback_state_changed)
            if self.audio_available:
                self.player.setSource(QUrl.fromLocalFile(str(self.audio_path)))
        except Exception as exc:  # noqa: BLE001 (any multimedia failure disables playback)
            self.player = None
            self.audio_output = None
            self._player_error = f"{type(exc).__name__}: {exc}"
            self.audio_available = False

    def _populate(self) -> None:
        self.timeline.set_marks([(m.start, m.end) for m in self.review.marks])
        self.mark_list.clear()
        for index, mark in enumerate(self.review.marks):
            item = QListWidgetItem(mark_row_text(index, mark, self.resolutions[index].resolved))
            self.mark_list.addItem(item)
            self.timeline.set_resolved(index, self.resolutions[index].resolved)
        if not self.review.marks:
            self.span_label.setText("No marks: the published transcript has no silent spans to check.")
            self.context_panel.setPlainText("")
            self.hint_panel.setPlainText("")
            for button in (self.play_button, self.nothing_button, self.text_button):
                button.setEnabled(False)

    def _initial_status(self) -> None:
        parts: list[str] = []
        if self._player_error is not None:
            parts.append(f"Playback is disabled: {self._player_error}.")
        elif not self.audio_available:
            parts.append(f"Audio file not found: {self.audio_path.name}. "
                         "Playback controls do nothing; resolutions are still recorded.")
        else:
            parts.append(f"Audio loaded: {self.audio_path.name}.")
        if self._resumed and self.done_count() > 0:
            parts.append(f"Resumed {self.done_count()} of {len(self.resolutions)} earlier resolutions.")
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
        """Make the mark at `index` current (clamped); selecting seeks to its start."""
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
        self._show_mark(row)

    def _show_mark(self, index: int) -> None:
        mark = self.review.marks[index]
        self._current = index
        self.timeline.set_current(index)
        self.span_label.setText(
            f"{format_mss_t(mark.start)} to {format_mss_t(mark.end)} ({mark.play_length:.1f} seconds)")
        self.context_panel.setPlainText(
            context_around_gap(self.review.transcript, mark.span_start, mark.span_end))
        self.hint_panel.setPlainText(mark.detector_text or "(the second engine recorded no text)")
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

    def play_span(self) -> None:
        """Enter: seek to the current mark's start, play, pause at its end."""
        mark = self.current_mark
        if mark is None:
            return
        self._seek(mark.start)
        if not self._playback_possible():
            return
        assert self.player is not None
        self._stop_at_s = mark.end
        self.player.play()
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
                self.player.pause()

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            return
        if self._stop_at_s is None and self.current_mark is not None and self.audio_available:
            self.set_status(f"Paused at {format_mss_t(self._position_s)}.")

    def _on_player_error(self, error: QMediaPlayer.Error, message: str) -> None:
        if error == QMediaPlayer.Error.NoError:
            return
        self._player_error = message or str(error)
        self.audio_available = False
        self.set_status(f"Playback is disabled: {self._player_error}.")

    # ----- resolving ------------------------------------------------------------------

    def resolve_nothing(self) -> None:
        """N: record that nothing was said in the current span, then advance."""
        if self._current is None:
            return
        self._resolve(self._current, STATUS_NOTHING, "")

    def resolve_text(self) -> None:
        """T: prompt for what was said (pre-filled with any earlier note), then advance."""
        if self._current is None:
            return
        earlier = self.resolutions[self._current].note
        typed = self.text_prompt(earlier)
        if typed is None:
            self.set_status("Cancelled; the mark is unchanged.")
            return
        self._resolve(self._current, STATUS_TEXT, typed)

    def _default_text_prompt(self, earlier: str) -> str | None:
        typed, accepted = QInputDialog.getText(
            self, "Type what was said", "What was said in this span:",
            QLineEdit.EchoMode.Normal, earlier)
        return typed if accepted else None

    def _resolve(self, index: int, status: str, note: str) -> None:
        resolution = self.resolutions[index]
        resolution.status = status
        resolution.note = note
        item = self.mark_list.item(index)
        if item is not None:
            item.setText(mark_row_text(index, self.review.marks[index], True))
        self.timeline.set_resolved(index, True)
        self._write_session()
        self._refresh_title()
        if status == STATUS_NOTHING:
            self.set_status(f"Mark {index + 1}: nothing was said.")
        else:
            self.set_status(f"Mark {index + 1}: \"{note}\".")
        if index + 1 < len(self.review.marks):
            self.select_mark(index + 1)

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

    # ----- keys -----------------------------------------------------------------------

    def handle_key(self, key: int, modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier) -> bool:
        """Apply one of the screen's keys; True when the key was one of them."""
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
            self.resolve_text()
        elif key == Qt.Key.Key_Left:
            self.nudge(-NUDGE_S)
        elif key == Qt.Key.Key_Right:
            self.nudge(NUDGE_S)
        else:
            return False
        return True

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt virtual)
        if self.handle_key(event.key(), event.modifiers()):
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt virtual)
        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
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
