"""The job card: what the detail pane shows for a recording that is queued or being worked on.

A stage strip with the current stage lit, a progress bar, the message of the moment, the time
elapsed, a smoothed estimate of the time left, a heartbeat that keeps moving between engine
reports together with the time since the last one, the acceleration plan in use, and the
lines the published engine has produced so far, provisional and without speaker labels. The
detector's text never appears here, as nowhere else.

The provisional lines can be read while the engine is still writing them. The list follows
the newest line only while the reader is at the bottom; scrolling up, or clicking a line,
holds the view where it is and a button offers the way back to the latest line, with the
count of lines that arrived meanwhile. Clicking a line moves the playhead to it and
double-clicking plays from it, so a passage can be checked against the audio before the
transcript is finished; the line under the playhead is highlighted as the audio plays.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from twinscribe.app.theme import Theme, theme_for, with_alpha
from twinscribe.outputs.transcript_doc import clock
from twinscribe.pipeline import STAGES, Progress

STAGE_LABELS: dict[str, str] = {
    "digest": "Reading",
    "decode": "Decoding",
    "publisher": "Transcribing",
    "detector": "Checking",
    "scenes": "Non-speech",
    "speakers": "Speakers",
    "outputs": "Writing",
}
HEARTBEAT_MS = 600
STALE_AFTER_S = 20.0
PROVISIONAL_TITLE = "What the published engine has heard so far (provisional; speakers are labelled at the end)"
PROVISIONAL_HINT = "Click a line to move the playhead there; double-click to play from it"
LATEST_TITLE = "Jump to latest"
START_ROLE = int(Qt.ItemDataRole.UserRole) + 1


def format_span(seconds: float) -> str:
    """m:ss, or h:mm:ss from an hour."""
    return clock(max(0.0, seconds), tenths=False)


def describe_remaining(eta_s: float | None) -> str:
    """A hedged phrase for the time left; the estimate is from the pace so far."""
    if eta_s is None:
        return "estimating the time left"
    if eta_s < 10.0:
        return "almost done"
    if eta_s < 90.0:
        return f"about {int(round(eta_s / 10.0)) * 10} s left"
    minutes = int(round(eta_s / 60.0))
    return f"roughly {minutes} min left"


def latest_button_text(unseen: int) -> str:
    """The label of the way back to the newest line, naming how many arrived unseen."""
    if unseen <= 0:
        return LATEST_TITLE
    noun = "new line" if unseen == 1 else "new lines"
    return f"{LATEST_TITLE} ({unseen} {noun})"


class JobStatusCard(QFrame):
    """The card for one queued or running recording.

    `seek_requested` carries the start time of a clicked line; `play_requested` the start time
    of a double-clicked line, which the owner should play from.
    """

    seek_requested = Signal(float)
    play_requested = Signal(float)

    def __init__(self, parent: QWidget | None = None, theme: Theme | None = None) -> None:
        super().__init__(parent)
        self._theme = theme if theme is not None else theme_for(False)
        self._stage_index: int | None = None
        self._last_report_at: float | None = None
        self._started_at: float | None = None
        self._running = False
        self._beat = False
        self._elapsed_s = 0.0
        self._eta_s: float | None = None
        self._message = ""
        self._partials: list[tuple[float, float, str]] = []
        self._following_latest = True
        self._follow_playback = True
        self._unseen = 0
        self._adjusting = False
        self._current_line: int | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 12)
        root.setSpacing(8)

        head = QHBoxLayout()
        self.heartbeat_label = QLabel("●", self)
        self.heartbeat_label.setFixedWidth(16)
        self.title_label = QLabel("", self)
        self.title_label.setObjectName("title")
        self.position_label = QLabel("", self)
        self.position_label.setObjectName("muted")
        head.addWidget(self.heartbeat_label)
        head.addWidget(self.title_label)
        head.addStretch(1)
        head.addWidget(self.position_label)
        root.addLayout(head)

        self.stage_label = QLabel("", self)
        self.stage_label.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(self.stage_label)

        bar_row = QHBoxLayout()
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setTextVisible(False)
        self.percent_label = QLabel("0%", self)
        self.percent_label.setFixedWidth(44)
        self.percent_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bar_row.addWidget(self.progress_bar, 1)
        bar_row.addWidget(self.percent_label)
        root.addLayout(bar_row)

        self.message_label = QLabel("", self)
        self.timing_label = QLabel("", self)
        self.timing_label.setObjectName("muted")
        root.addWidget(self.message_label)
        root.addWidget(self.timing_label)

        self.plan_label = QLabel("", self)
        self.plan_label.setObjectName("muted")
        self.plan_label.setWordWrap(True)
        root.addWidget(self.plan_label)

        self.provisional_title = QLabel(PROVISIONAL_TITLE, self)
        self.provisional_title.setObjectName("section")
        root.addSpacing(6)
        root.addWidget(self.provisional_title)

        self.provisional = QListWidget(self)
        self.provisional.setFrameShape(QFrame.Shape.NoFrame)
        self.provisional.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.provisional.setWordWrap(True)
        self.provisional.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.provisional.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.provisional.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.provisional.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.provisional.setSpacing(2)
        font = QFont(self.provisional.font())
        font.setPointSizeF(11.0)
        self.provisional.setFont(font)
        self.provisional.itemClicked.connect(self._on_item_clicked)
        self.provisional.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.provisional.verticalScrollBar().valueChanged.connect(self._on_scrolled)
        root.addWidget(self.provisional, 1)

        foot = QHBoxLayout()
        self.hint_label = QLabel(PROVISIONAL_HINT, self)
        self.hint_label.setObjectName("muted")
        self.latest_button = QPushButton(LATEST_TITLE, self)
        self.latest_button.setObjectName("flat")
        self.latest_button.setToolTip("Follow the newest line again")
        self.latest_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.latest_button.clicked.connect(lambda _checked=False: self.jump_to_latest())
        self.latest_button.hide()
        foot.addWidget(self.hint_label)
        foot.addStretch(1)
        foot.addWidget(self.latest_button)
        self.foot_widget = QWidget(self)
        self.foot_widget.setLayout(foot)
        foot.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.foot_widget)

        self._timer = QTimer(self)
        self._timer.setInterval(HEARTBEAT_MS)
        self._timer.timeout.connect(self._tick)
        self.set_theme(self._theme)

    # ----- appearance ----------------------------------------------------------------------

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self._refresh_stage_strip()
        self._refresh_heartbeat()
        self._paint_current_line()

    def _refresh_stage_strip(self) -> None:
        parts: list[str] = []
        for index, (name, _) in enumerate(STAGES):
            if self._stage_index is None:
                colour, weight = self._theme.muted.name(), "normal"
            elif index < self._stage_index:
                colour, weight = self._theme.success.name(), "normal"
            elif index == self._stage_index:
                colour, weight = self._theme.accent.name(), "600"
            else:
                colour, weight = self._theme.muted.name(), "normal"
            parts.append(f'<span style="color:{colour}; font-weight:{weight};">&#9679; {STAGE_LABELS[name]}</span>')
        self.stage_label.setText("&nbsp;&nbsp;&nbsp;".join(parts))

    def _refresh_heartbeat(self) -> None:
        if not self._running:
            colour = self._theme.muted
        else:
            colour = self._theme.accent if self._beat else self._theme.muted
        self.heartbeat_label.setStyleSheet(f"color: {colour.name()}; font-size: 14pt;")

    def _refresh_latest_button(self) -> None:
        self.latest_button.setText(latest_button_text(self._unseen))
        self.latest_button.setVisible(not self._following_latest and self.provisional.count() > 0)

    # ----- state -------------------------------------------------------------------------

    def show_queued(self, name: str, position: int, total: int) -> None:
        """A recording that is waiting for its turn in the batch."""
        self._running = False
        self._stage_index = None
        self._started_at = None
        self._last_report_at = None
        self._timer.stop()
        self.title_label.setText(name)
        self.position_label.setText(f"{position} of {total}")
        self.progress_bar.setValue(0)
        self.percent_label.setText("")
        self.message_label.setText("Queued; waiting for the recordings before it")
        self.timing_label.setText("")
        self.plan_label.setText("")
        self.provisional_title.hide()
        self.provisional.hide()
        self.foot_widget.hide()
        self._refresh_stage_strip()
        self._refresh_heartbeat()

    def show_running(self, name: str, position: int, total: int, plan_lines: Sequence[str] = ()) -> None:
        """A recording that is being worked on; progress reports follow."""
        self._running = True
        self._stage_index = 0
        self._started_at = time.monotonic()
        self._last_report_at = None
        self._elapsed_s = 0.0
        self._eta_s = None
        self._message = "Starting"
        self.title_label.setText(name)
        self.position_label.setText(f"{position} of {total}")
        self.progress_bar.setValue(0)
        self.percent_label.setText("0%")
        self.message_label.setText(self._message)
        self.plan_label.setText("   |   ".join(plan_lines))
        self.provisional_title.show()
        self.provisional.show()
        self.foot_widget.show()
        self._clear_partials()
        self._refresh_stage_strip()
        self._refresh_timing()
        self._refresh_heartbeat()
        self._timer.start()

    def update_progress(self, report: Progress) -> None:
        """Apply one progress report."""
        names = [name for name, _ in STAGES]
        if report.stage in names:
            self._stage_index = names.index(report.stage)
        self.progress_bar.setValue(int(round(1000 * min(1.0, max(0.0, report.fraction)))))
        self.percent_label.setText(f"{int(round(100 * report.fraction))}%")
        self._message = report.message
        self.message_label.setText(report.message)
        self._elapsed_s = float(report.elapsed_s)
        self._eta_s = report.eta_s
        self._last_report_at = time.monotonic()
        self._refresh_stage_strip()
        self._refresh_timing()

    def stop(self) -> None:
        """Stop the heartbeat; the card is about to be replaced."""
        self._running = False
        self._timer.stop()
        self._refresh_heartbeat()

    # ----- provisional lines ---------------------------------------------------------------

    def _clear_partials(self) -> None:
        self._adjusting = True
        self.provisional.clear()
        self._adjusting = False
        self._partials = []
        self._current_line = None
        self._unseen = 0
        self._following_latest = True
        self._refresh_latest_button()

    def add_partial(self, start: float, end: float, text: str) -> None:
        """Append one provisional line from the published engine.

        The list scrolls to it only while the reader is at the bottom; otherwise the line is
        counted as unseen and the button back to the latest line says so.
        """
        clean = " ".join(str(text).split())
        if not clean:
            return
        self._partials.append((float(start), float(end), clean))
        item = QListWidgetItem(f"[{clock(start)}]  {clean}")
        item.setData(START_ROLE, float(start))
        item.setToolTip(f"{clock(start)} to {clock(end)}")
        self.provisional.addItem(item)
        if self._following_latest:
            self._scroll_to_bottom()
        else:
            self._unseen += 1
            self._refresh_latest_button()

    def set_partials(self, lines: Sequence[tuple[float, float, str]]) -> None:
        """Replace the provisional lines (when a recording is selected again mid-run)."""
        self._clear_partials()
        for start, end, text in lines:
            self.add_partial(start, end, text)

    def partial_count(self) -> int:
        return len(self._partials)

    def line_start(self, index: int) -> float:
        """The start time of the provisional line at `index`."""
        return self._partials[index][0]

    def line_text(self, index: int) -> str:
        return self._partials[index][2]

    def following_latest(self) -> bool:
        """True while the list follows the newest line."""
        return self._following_latest

    def unseen_count(self) -> int:
        """Lines that arrived while the reader was elsewhere in the list."""
        return self._unseen

    def jump_to_latest(self) -> None:
        """Follow the newest line again."""
        self._following_latest = True
        self._unseen = 0
        self._scroll_to_bottom()
        self._refresh_latest_button()

    def hold_position(self) -> None:
        """Stop following the newest line; the reader is looking at something."""
        if self._following_latest:
            self._following_latest = False
            self._refresh_latest_button()

    def _scroll_to_bottom(self) -> None:
        self._adjusting = True
        try:
            self.provisional.scrollToBottom()
        finally:
            self._adjusting = False

    def _on_scrolled(self, value: int) -> None:
        if self._adjusting or self.provisional.count() == 0:
            return
        at_bottom = value >= self.provisional.verticalScrollBar().maximum()
        if at_bottom and not self._following_latest:
            self.jump_to_latest()
        elif not at_bottom and self._following_latest:
            self.hold_position()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        self.hold_position()
        start = item.data(START_ROLE)
        if start is not None:
            self.seek_requested.emit(float(start))

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        self.hold_position()
        start = item.data(START_ROLE)
        if start is not None:
            self.play_requested.emit(float(start))

    # ----- the playhead --------------------------------------------------------------------

    def set_follow(self, follow: bool) -> None:
        """Whether the line under the playhead is kept in view while the list is held."""
        self._follow_playback = bool(follow)

    def line_at(self, seconds: float) -> int | None:
        """Index of the last provisional line that starts at or before `seconds`, or None."""
        found: int | None = None
        for index, (start, _, _) in enumerate(self._partials):
            if start <= seconds:
                if found is None or start >= self._partials[found][0]:
                    found = index
        return found

    def current_line(self) -> int | None:
        return self._current_line

    def set_position(self, seconds: float) -> None:
        """Highlight the line under the playhead; keep it in view while the list is held."""
        line = self.line_at(seconds)
        if line == self._current_line:
            return
        previous = self._current_line
        self._current_line = line
        if previous is not None and previous < self.provisional.count():
            self.provisional.item(previous).setBackground(QBrush())
        self._paint_current_line()
        if line is not None and self._follow_playback and not self._following_latest:
            self._adjusting = True
            try:
                self.provisional.scrollToItem(self.provisional.item(line), QAbstractItemView.ScrollHint.PositionAtCenter)
            finally:
                self._adjusting = False

    def _paint_current_line(self) -> None:
        if self._current_line is None or self._current_line >= self.provisional.count():
            return
        tint: QColor = with_alpha(self._theme.accent, 34 if not self._theme.dark else 60)
        self.provisional.item(self._current_line).setBackground(QBrush(tint))

    # ----- ticking -------------------------------------------------------------------------

    def _tick(self) -> None:
        self._beat = not self._beat
        self._refresh_heartbeat()
        self._refresh_timing()

    def _refresh_timing(self) -> None:
        if not self._running or self._started_at is None:
            self.timing_label.setText("")
            return
        elapsed = max(self._elapsed_s, time.monotonic() - self._started_at)
        parts = [f"Elapsed {format_span(elapsed)}", describe_remaining(self._eta_s)]
        if self._last_report_at is not None:
            since = time.monotonic() - self._last_report_at
            if since >= 3.0:
                parts.append(f"last report {int(since)} s ago" + (", still working" if since >= STALE_AFTER_S else ""))
        self.timing_label.setText("   |   ".join(parts))

    # ----- for tests ---------------------------------------------------------------------

    def stage_index(self) -> int | None:
        return self._stage_index

    def message(self) -> str:
        return self._message

    def running(self) -> bool:
        return self._running
