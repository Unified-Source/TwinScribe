"""The player bar: timeline with the waveform overview and the review marks, transport
buttons, the time readout, follow toggle, playback speed and volume.

The bar owns no media object; it shows state and emits requests, so the window can drive one
QMediaPlayer for the transcript pane, the timeline and the bar together.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QSlider, QToolButton, QVBoxLayout, QWidget

from twinscribe.app.icons import make_icon
from twinscribe.app.theme import Theme, theme_for
from twinscribe.app.timeline import Timeline
from twinscribe.outputs.transcript_doc import clock

SKIP_S = 5.0
RATES: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0)


class PlayerBar(QFrame):
    """Transport controls under the transcript."""

    seek_requested = Signal(float)
    skip_requested = Signal(float)
    play_toggled = Signal()
    follow_toggled = Signal(bool)
    rate_changed = Signal(float)
    volume_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None, theme: Theme | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("playerbar")
        self._theme = theme if theme is not None else theme_for(False)
        self._duration_s = 0.0
        self._position_s = 0.0
        self._playing = False

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 8, 14, 8)
        root.setSpacing(4)

        self.timeline = Timeline(self)
        self.timeline.set_show_labels(False)
        self.timeline.setMinimumHeight(56)
        self.timeline.seek_requested.connect(self.seek_requested)
        root.addWidget(self.timeline)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.back_button = self._tool("back", "Back 5 seconds (Left)")
        self.play_button = self._tool("play", "Play or pause (Space)")
        self.forward_button = self._tool("forward", "Forward 5 seconds (Right)")
        self.play_button.setIconSize(self.play_button.iconSize() * 1.35)
        self.back_button.clicked.connect(lambda: self.skip_requested.emit(-SKIP_S))
        self.forward_button.clicked.connect(lambda: self.skip_requested.emit(SKIP_S))
        self.play_button.clicked.connect(self.play_toggled)
        row.addWidget(self.back_button)
        row.addWidget(self.play_button)
        row.addWidget(self.forward_button)

        self.time_label = QLabel("0:00.0  /  0:00", self)
        self.time_label.setObjectName("muted")
        row.addSpacing(8)
        row.addWidget(self.time_label)
        row.addStretch(1)

        self.follow_button = self._tool("follow", "Keep the current line in view (F)")
        self.follow_button.setCheckable(True)
        self.follow_button.setChecked(True)
        self.follow_button.toggled.connect(self.follow_toggled)
        row.addWidget(self.follow_button)

        self.rate_box = QComboBox(self)
        for rate in RATES:
            self.rate_box.addItem(f"{rate:g}x", rate)
        self.rate_box.setCurrentIndex(RATES.index(1.0))
        self.rate_box.setToolTip("Playback speed")
        self.rate_box.currentIndexChanged.connect(self._on_rate)
        row.addWidget(self.rate_box)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setFixedWidth(96)
        self.volume_slider.setToolTip("Volume")
        self.volume_slider.valueChanged.connect(lambda v: self.volume_changed.emit(v / 100.0))
        row.addSpacing(6)
        row.addWidget(self.volume_slider)
        root.addLayout(row)
        self.set_theme(self._theme)

    def _tool(self, kind: str, tip: str) -> QToolButton:
        button = QToolButton(self)
        button.setToolTip(tip)
        button.setProperty("icon_kind", kind)
        button.setAutoRaise(True)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return button

    # ----- appearance ----------------------------------------------------------------------

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        colour = self.palette().color(self.palette().ColorRole.Text)
        for button in (self.back_button, self.forward_button):
            button.setIcon(make_icon(str(button.property("icon_kind")), colour))
        self.follow_button.setIcon(make_icon("follow", theme.accent if self.follow_button.isChecked() else theme.muted))
        self._refresh_play_icon()
        self.timeline.set_colours(theme.accent, theme.success, theme.track, theme.outline, theme.muted)

    def _refresh_play_icon(self) -> None:
        colour = self.palette().color(self.palette().ColorRole.Text)
        self.play_button.setIcon(make_icon("pause" if self._playing else "play", colour))

    # ----- state -------------------------------------------------------------------------

    def set_duration(self, seconds: float) -> None:
        self._duration_s = max(0.0, float(seconds))
        self.timeline.set_duration(self._duration_s)
        self._refresh_time()

    def set_position(self, seconds: float) -> None:
        self._position_s = max(0.0, float(seconds))
        self.timeline.set_playhead(self._position_s)
        self._refresh_time()

    def set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        self._refresh_play_icon()

    def set_marks(self, marks: Sequence[tuple[float, float]]) -> None:
        self.timeline.set_marks(marks)

    def set_peaks(self, peaks: Sequence[int] | None, scale: int = 100) -> None:
        self.timeline.set_peaks(peaks, scale)

    def set_follow(self, follow: bool) -> None:
        self.follow_button.setChecked(bool(follow))
        self.follow_button.setIcon(make_icon("follow", self._theme.accent if follow else self._theme.muted))

    def set_rate(self, rate: float) -> None:
        index = self.rate_box.findData(rate)
        if index >= 0:
            self.rate_box.setCurrentIndex(index)

    def set_volume(self, volume: float) -> None:
        self.volume_slider.setValue(int(round(100.0 * min(1.0, max(0.0, volume)))))

    def set_available(self, available: bool) -> None:
        """Grey the transport when nothing can be played."""
        for widget in (self.back_button, self.play_button, self.forward_button, self.timeline):
            widget.setEnabled(available)

    def _refresh_time(self) -> None:
        self.time_label.setText(f"{clock(self._position_s)}  /  {clock(self._duration_s, tenths=False)}")

    def _on_rate(self, index: int) -> None:
        rate = self.rate_box.itemData(index)
        if rate is not None:
            self.rate_changed.emit(float(rate))
