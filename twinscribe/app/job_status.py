"""The job card: what the detail pane shows for a recording that is queued or being worked on.

A stage strip with the current stage lit, a progress bar, the message of the moment, the time
elapsed, a smoothed estimate of the time left, a heartbeat that keeps moving between engine
reports together with the time since the last one, the acceleration plan in use, and the
lines the published engine has produced so far, provisional and without speaker labels. The
detector's text never appears here, as nowhere else.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QVBoxLayout, QWidget

from twinscribe.app.theme import Theme, theme_for
from twinscribe.outputs.transcript_doc import clock
from twinscribe.pipeline import STAGES, Progress

STAGE_LABELS: dict[str, str] = {
    "digest": "Reading",
    "decode": "Decoding",
    "publisher": "Transcribing",
    "detector": "Checking",
    "speakers": "Speakers",
    "outputs": "Writing",
}
HEARTBEAT_MS = 600
STALE_AFTER_S = 20.0
PROVISIONAL_TITLE = "What the published engine has heard so far (provisional; speakers are labelled at the end)"


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


class JobStatusCard(QFrame):
    """The card for one queued or running recording."""

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
        self.provisional = QPlainTextEdit(self)
        self.provisional.setReadOnly(True)
        self.provisional.setFrameShape(QFrame.Shape.NoFrame)
        font = QFont(self.provisional.font())
        font.setPointSizeF(11.0)
        self.provisional.setFont(font)
        root.addWidget(self.provisional, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(HEARTBEAT_MS)
        self._timer.timeout.connect(self._tick)
        self.set_theme(self._theme)

    # ----- appearance ----------------------------------------------------------------------

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self._refresh_stage_strip()
        self._refresh_heartbeat()

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
        self.provisional.clear()
        self._partials = []
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

    def add_partial(self, start: float, end: float, text: str) -> None:
        """Append one provisional line from the published engine."""
        clean = " ".join(str(text).split())
        if not clean:
            return
        self._partials.append((float(start), float(end), clean))
        self.provisional.appendPlainText(f"[{clock(start)}]  {clean}")
        self.provisional.moveCursor(QTextCursor.MoveOperation.End)

    def set_partials(self, lines: Sequence[tuple[float, float, str]]) -> None:
        """Replace the provisional lines (when a recording is selected again mid-run)."""
        self.provisional.clear()
        self._partials = []
        for start, end, text in lines:
            self.add_partial(start, end, text)

    def partial_count(self) -> int:
        return len(self._partials)

    def stop(self) -> None:
        """Stop the heartbeat; the card is about to be replaced."""
        self._running = False
        self._timer.stop()
        self._refresh_heartbeat()

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
