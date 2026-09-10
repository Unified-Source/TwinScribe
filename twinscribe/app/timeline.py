"""Timeline bar for the twinscribe screens.

One bar stands for the whole recording. When a waveform overview is set it is drawn inside
the bar, mirrored about the centre line. Every mark is drawn on it in the accent colour,
resolved marks in the success colour, the current mark is outlined, and a playhead line shows
the playback position. Clicking or dragging on the bar asks the owner to seek.
"""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

MIN_HEIGHT_PX = 70
LABEL_STRIP_PX = 18
SIDE_MARGIN_PX = 8
MIN_MARK_WIDTH_PX = 2.0


def format_mss(seconds: float) -> str:
    """Format seconds as m:ss for the end labels (65.0 -> "1:05")."""
    whole = max(0, int(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


class Timeline(QWidget):
    """Bar for the whole recording with marks, the current mark, a playhead and, when set, a
    waveform overview.

    `seek_requested` carries the time in seconds under a click or drag on the bar.
    """

    seek_requested = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._duration_s = 0.0
        self._marks: list[tuple[float, float]] = []
        self._resolved: list[bool] = []
        self._current: int | None = None
        self._playhead_s = 0.0
        self._peaks: list[int] | None = None
        self._peak_scale = 100
        self._accent = QColor("#2b63c9")
        self._success = QColor("#2a8a4f")
        self._track = QColor("#d9d9d9")
        self._outline = QColor("#1c1c1c")
        self._peak_colour = QColor("#9a9aa0")
        self._show_labels = True
        self.setMinimumHeight(MIN_HEIGHT_PX)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ----- state -----------------------------------------------------------------------

    def set_colours(self, accent: QColor, success: QColor, track: QColor | None = None,
                    outline: QColor | None = None, peaks: QColor | None = None) -> None:
        """Set the state colours; `track`, `outline` and `peaks` keep their values when None."""
        self._accent = QColor(accent)
        self._success = QColor(success)
        if track is not None:
            self._track = QColor(track)
        if outline is not None:
            self._outline = QColor(outline)
        if peaks is not None:
            self._peak_colour = QColor(peaks)
        self.update()

    def set_duration(self, seconds: float) -> None:
        """Set the recording length that the bar stands for."""
        self._duration_s = max(0.0, float(seconds))
        self.update()

    def duration(self) -> float:
        """Recording length in seconds."""
        return self._duration_s

    def set_marks(self, marks: Sequence[tuple[float, float]], resolved: Sequence[bool] | None = None) -> None:
        """Replace the marks with (start, end) pairs in seconds; `resolved`, one flag per
        mark, says which are drawn as resolved, else all become unresolved."""
        self._marks = [(float(s), float(e)) for s, e in marks]
        if resolved is not None and len(resolved) == len(self._marks):
            self._resolved = [bool(flag) for flag in resolved]
        else:
            self._resolved = [False] * len(self._marks)
        if self._current is not None and self._current >= len(self._marks):
            self._current = None
        self.update()

    def set_resolved(self, index: int, resolved: bool) -> None:
        """Flag one mark as resolved or not."""
        if 0 <= index < len(self._resolved):
            self._resolved[index] = bool(resolved)
            self.update()

    def set_current(self, index: int | None) -> None:
        """Outline the mark at `index`, or none."""
        if index is not None and not (0 <= index < len(self._marks)):
            index = None
        self._current = index
        self.update()

    def current(self) -> int | None:
        """Index of the outlined mark, or None."""
        return self._current

    def set_playhead(self, seconds: float) -> None:
        """Move the playhead line to `seconds`, clamped to the recording once its length is known.

        Before the player has reported a duration the value is kept as given, so a seek made
        while a recording is still loading is not flattened to zero.
        """
        value = max(0.0, float(seconds))
        self._playhead_s = min(value, self._duration_s) if self._duration_s > 0.0 else value
        self.update()

    def playhead(self) -> float:
        """Playhead position in seconds."""
        return self._playhead_s

    def set_peaks(self, peaks: Sequence[int] | None, scale: int = 100) -> None:
        """Set the waveform overview (peak levels per bin on a 0 to `scale` range) or clear it."""
        self._peaks = None if peaks is None else [int(p) for p in peaks]
        self._peak_scale = max(1, int(scale))
        self.update()

    def peaks(self) -> list[int] | None:
        """The waveform overview, or None."""
        return None if self._peaks is None else list(self._peaks)

    def set_show_labels(self, show: bool) -> None:
        """Show or hide the time labels above the bar."""
        self._show_labels = bool(show)
        self.update()

    # ----- geometry --------------------------------------------------------------------

    def bar_rect(self) -> QRectF:
        """Rectangle of the bar itself, below the label strip."""
        strip = LABEL_STRIP_PX if self._show_labels else 4
        return QRectF(
            SIDE_MARGIN_PX,
            strip,
            max(1.0, self.width() - 2 * SIDE_MARGIN_PX),
            max(1.0, self.height() - strip - 4),
        )

    def x_at_time(self, seconds: float) -> float:
        """Horizontal pixel position of a time; the left edge of the bar when duration is 0."""
        bar = self.bar_rect()
        if self._duration_s <= 0.0:
            return bar.left()
        fraction = min(max(0.0, seconds / self._duration_s), 1.0)
        return bar.left() + fraction * bar.width()

    def time_at_x(self, x: float) -> float:
        """Time in seconds under a horizontal pixel position, clamped to the recording."""
        bar = self.bar_rect()
        if bar.width() <= 0.0 or self._duration_s <= 0.0:
            return 0.0
        fraction = min(max(0.0, (x - bar.left()) / bar.width()), 1.0)
        return fraction * self._duration_s

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt virtual)
        return QSize(640, MIN_HEIGHT_PX + 14)

    # ----- painting --------------------------------------------------------------------

    def _paint_peaks(self, painter: QPainter, bar: QRectF) -> None:
        if not self._peaks:
            return
        inset = 3.0
        half = (bar.height() - 2 * inset) / 2.0
        centre = bar.top() + inset + half
        columns = max(1, int(bar.width()))
        bins = len(self._peaks)
        pen = QPen(self._peak_colour)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        for column in range(columns):
            lo = int(column * bins / columns)
            hi = max(lo + 1, int((column + 1) * bins / columns))
            level = max(self._peaks[lo:hi]) / float(self._peak_scale)
            height = max(1.0, level * half)
            x = bar.left() + column + 0.5
            painter.drawLine(QPointF(x, centre - height), QPointF(x, centre + height))

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt virtual)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bar = self.bar_rect()
        text_colour = self.palette().color(self.foregroundRole())

        if self._show_labels:
            painter.setPen(text_colour)
            label_rect = QRectF(bar.left(), 0.0, bar.width(), LABEL_STRIP_PX)
            painter.drawText(label_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                             format_mss(0.0))
            painter.drawText(label_rect, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                             format_mss(self._duration_s))

        # Track and overview.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._track)
        painter.drawRoundedRect(bar, 3.0, 3.0)
        self._paint_peaks(painter, bar)

        # Marks. Translucent over an overview so the waveform shows through; the current one is
        # drawn last so its outline is not covered.
        inset = 3.0
        alpha = 150 if self._peaks else 255
        order = [i for i in range(len(self._marks)) if i != self._current]
        if self._current is not None:
            order.append(self._current)
        for index in order:
            start, end = self._marks[index]
            left = self.x_at_time(start)
            right = max(self.x_at_time(end), left + MIN_MARK_WIDTH_PX)
            rect = QRectF(left, bar.top() + inset, right - left, bar.height() - 2 * inset)
            fill = QColor(self._success if self._resolved[index] else self._accent)
            fill.setAlpha(alpha)
            painter.setBrush(fill)
            if index == self._current:
                pen = QPen(self._outline)
                pen.setWidthF(2.0)
                painter.setPen(pen)
                painter.drawRect(rect.adjusted(-1.0, -1.0, 1.0, 1.0))
            else:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(rect)

        # Playhead.
        x = self.x_at_time(self._playhead_s)
        pen = QPen(self._outline)
        pen.setWidthF(2.0)
        painter.setPen(pen)
        painter.drawLine(QPointF(x, bar.top() - 2.0), QPointF(x, bar.bottom() + 2.0))
        painter.end()

    # ----- mouse -----------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt virtual)
        if event.button() == Qt.MouseButton.LeftButton:
            self.seek_requested.emit(self.time_at_x(event.position().x()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt virtual)
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.seek_requested.emit(self.time_at_x(event.position().x()))
            event.accept()
            return
        super().mouseMoveEvent(event)
