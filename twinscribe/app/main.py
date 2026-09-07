"""The application window: a library of recordings, a player whose transcript follows the
audio, and the batch that produces the outputs for every recording.

One window. Recordings and folders are dropped in or opened; each shows its state in the
library. A recording plays whether or not it has been transcribed; once it has, its transcript
lines follow the playhead, the review marks sit on the timeline and between the lines, and the
verification screen opens on the review list. A quality level is chosen, Transcribe runs the
pipeline over everything not yet done, in a thread, with progress per file. Nothing in this
module or below it reaches the network.

Entry point: `python -m twinscribe.app [paths...] [--models DIR] [--dark] [--shot out.png]`.

Screenshots (`--shot`) must be taken on the platform's real backend; under the offscreen
platform every glyph renders as a box.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent, QKeyEvent, QKeySequence
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from twinscribe import __version__
from twinscribe.app.icons import make_icon
from twinscribe.app.job_status import JobStatusCard
from twinscribe.app.library import STATUS_QUEUED, STATUS_RUNNING, LibraryModel, LibraryView, MediaItem
from twinscribe.app.player import PlayerBar
from twinscribe.app.settings import (
    ACCELERATIONS,
    OUTPUT_BESIDE,
    OUTPUT_FOLDER,
    AppSettings,
    load_settings,
    save_settings,
)
from twinscribe.app.theme import Theme, apply_styles, apply_theme, theme_for
from twinscribe.app.transcript_view import TranscriptView
from twinscribe.app.worker import PipelineWorker
from twinscribe.hardware import (
    BACKEND_CT2,
    BACKEND_NONE,
    BACKEND_ONNX,
    Plan,
    current_plan,
    describe_machine,
    probe_libraries,
    probe_machine,
    probe_nvidia_gpus,
)
from twinscribe.models import ModelSet, find_models
from twinscribe.outputs import render_all
from twinscribe.outputs.transcript_doc import (
    approximate_word_times,
    clock,
    load_document,
    set_speaker_name,
    speaker_names,
    write_document,
)
from twinscribe.pipeline import MEDIA_EXTENSIONS, Engines, FileResult, Progress, output_paths
from twinscribe.profiles import ModelsMissing, Profile, available_profiles, profile_for, select as select_level
from twinscribe.runrecord import write_json_atomic

ACCELERATION_TITLES: dict[str, str] = {"auto": "Automatic", "cpu": "Processor only", "cuda": "CUDA device"}

APP_TITLE = "twinscribe"
SHOT_DELAY_MS = 1200
NUDGE_S = 5.0
SESSION_SCHEMA = "twinscribe.review-session.v1"


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _file_filter() -> str:
    patterns = " ".join(f"*{ext}" for ext in sorted(MEDIA_EXTENSIONS))
    return f"Recordings ({patterns});;All files (*)"


def read_session_entries(path: Path, expected: int) -> list[dict[str, Any]] | None:
    """Entries of a review session beside a review set, when it has one entry per mark."""
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
    if not isinstance(entries, list) or len(entries) != expected:
        return None
    return [dict(e) for e in entries if isinstance(e, dict)] if all(isinstance(e, dict) for e in entries) else None


class SpeakerChip(QLabel):
    """One speaker with its colour and word count; double-click asks to rename it."""

    rename_requested = Signal(str)

    def __init__(self, label: str, name: str, words: int, colour: str, muted: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = label
        noun = "word" if words == 1 else "words"
        self.setText(
            f'<span style="color:{colour}; font-size:13pt;">&#9679;</span>&nbsp;<b>{name}</b>'
            f'&nbsp;<span style="color:{muted};">{words:,} {noun}</span>'
        )
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setToolTip("Double-click to rename this speaker")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt virtual)
        self.rename_requested.emit(self.label)
        event.accept()


class SettingsDialog(QDialog):
    """Models folder, where outputs go, author, threads and the palette."""

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(560)
        self.settings = settings
        form = QFormLayout(self)
        form.setSpacing(10)

        models_row = QHBoxLayout()
        self.models_edit = QLineEdit(settings.models_dir, self)
        self.models_edit.setPlaceholderText(str(find_models(None).root))
        browse_models = QPushButton("Browse", self)
        browse_models.clicked.connect(self._browse_models)
        models_row.addWidget(self.models_edit, 1)
        models_row.addWidget(browse_models)
        form.addRow("Models folder", models_row)

        self.models_report = QPlainTextEdit(self)
        self.models_report.setReadOnly(True)
        self.models_report.setMaximumHeight(150)
        form.addRow("", self.models_report)
        self.models_edit.editingFinished.connect(self._refresh_report)

        output_box = QVBoxLayout()
        self.beside_radio = QRadioButton("Beside each recording", self)
        self.folder_radio = QRadioButton("In one folder:", self)
        folder_row = QHBoxLayout()
        self.output_edit = QLineEdit(settings.output_dir, self)
        browse_output = QPushButton("Browse", self)
        browse_output.clicked.connect(self._browse_output)
        folder_row.addWidget(self.output_edit, 1)
        folder_row.addWidget(browse_output)
        output_box.addWidget(self.beside_radio)
        output_box.addWidget(self.folder_radio)
        output_box.addLayout(folder_row)
        (self.folder_radio if settings.output_mode == OUTPUT_FOLDER else self.beside_radio).setChecked(True)
        form.addRow("Outputs", output_box)

        self.author_edit = QLineEdit(settings.author, self)
        self.author_edit.setPlaceholderText("Written into the Word document properties")
        form.addRow("Author", self.author_edit)

        self.threads_spin = QSpinBox(self)
        self.threads_spin.setRange(0, 64)
        self.threads_spin.setSpecialValueText("Automatic")
        self.threads_spin.setValue(settings.threads)
        form.addRow("Threads per engine", self.threads_spin)

        self.acceleration_box = QComboBox(self)
        for key in ACCELERATIONS:
            self.acceleration_box.addItem(ACCELERATION_TITLES[key], key)
        self.acceleration_box.setCurrentIndex(max(0, self.acceleration_box.findData(settings.acceleration)))
        self.acceleration_box.currentIndexChanged.connect(self._refresh_plan)
        form.addRow("Acceleration", self.acceleration_box)

        self.plan_report = QPlainTextEdit(self)
        self.plan_report.setReadOnly(True)
        self.plan_report.setMaximumHeight(150)
        form.addRow("", self.plan_report)

        self.theme_box = QComboBox(self)
        self.theme_box.addItems(["Light", "Dark"])
        self.theme_box.setCurrentIndex(1 if settings.dark else 0)
        form.addRow("Appearance", self.theme_box)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._refresh_report()
        self._refresh_plan()

    def _refresh_plan(self) -> None:
        preference = str(self.acceleration_box.currentData() or "auto")
        threads = int(self.threads_spin.value()) or None
        try:
            lines = describe_machine(probe_machine(), probe_nvidia_gpus(), probe_libraries())
            lines += current_plan(preference, threads).describe()
        except Exception as exc:  # noqa: BLE001 - the dialog must open whatever the probe does
            lines = [f"The machine could not be probed: {exc}"]
        self.plan_report.setPlainText("\n".join(lines))

    def _browse_models(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Models folder", self.models_edit.text() or str(Path.home()))
        if chosen:
            self.models_edit.setText(chosen)
            self._refresh_report()

    def _browse_output(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Output folder", self.output_edit.text() or str(Path.home()))
        if chosen:
            self.output_edit.setText(chosen)
            self.folder_radio.setChecked(True)

    def _refresh_report(self) -> None:
        root = self.models_edit.text().strip() or None
        models = find_models(root)
        levels = available_profiles(models)
        level_text = ", ".join(p.title for p in levels) if levels else "none; no level has all of its models"
        self.models_report.setPlainText(f"{models.root}\n{models.describe()}\nQuality levels: {level_text}")

    def apply_to(self, settings: AppSettings) -> AppSettings:
        """Copy the dialog fields into `settings` and return it."""
        settings.models_dir = self.models_edit.text().strip()
        settings.output_mode = OUTPUT_FOLDER if self.folder_radio.isChecked() and self.output_edit.text().strip() else OUTPUT_BESIDE
        settings.output_dir = self.output_edit.text().strip()
        settings.author = self.author_edit.text().strip()
        settings.threads = int(self.threads_spin.value())
        settings.acceleration = str(self.acceleration_box.currentData() or "auto")
        settings.dark = self.theme_box.currentIndex() == 1
        return settings


class MainWindow(QMainWindow):
    """The one window.

    `engines` lets a test stand synthetic engines in for the real ones; `record_dir` is where
    batch records go (the application home by default); `plan` and `backends` replace the
    machine probe (tests), otherwise the machine is probed.
    """

    def __init__(
        self,
        settings: AppSettings,
        theme: Theme | None = None,
        engines: Engines | None = None,
        record_dir: Path | None = None,
        parent: QWidget | None = None,
        plan: Plan | None = None,
        backends: dict[str, bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.theme = theme if theme is not None else theme_for(settings.dark)
        self._engines = engines
        self._record_dir = record_dir
        self._plan_override = plan
        if backends is not None:
            self._backends = dict(backends)
        elif plan is not None:
            self._backends = dict(plan.backends)
        else:
            libraries = probe_libraries()
            self._backends = {BACKEND_CT2: libraries.whisper_ct2, BACKEND_ONNX: libraries.sherpa_onnx}
        self.models: ModelSet = find_models(settings.models_dir or None)
        self.worker: PipelineWorker | None = None
        self._current_row: int | None = None
        self._current_doc: dict[str, Any] | None = None
        self._current_path: Path | None = None
        self._stop_at_s: float | None = None
        self._player_error: str | None = None
        self._duration_s = 0.0
        self._position_s = 0.0
        self._batch_total = 0
        self._batch_done = 0
        self._batch_rows: list[int] = []
        self._batch_plan: Plan | None = None
        self._partials: dict[int, list[tuple[float, float, str]]] = {}

        self.library_model = LibraryModel(self, settings.output_dir_or_none)
        self._build_ui()
        self._build_player()
        self._refresh_quality_box()
        self.setAcceptDrops(True)
        self.setWindowTitle(APP_TITLE)
        self._show_empty_detail()
        existing = [Path(p) for p in settings.library if Path(p).exists()]
        if existing:
            self.add_paths(existing)
        self.set_status(f"{APP_TITLE} {__version__}. Drop recordings or folders here, or use Open.")

    # ----- construction ---------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top bar.
        top = QFrame(central)
        top.setObjectName("topbar")
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(16, 10, 16, 10)
        top_layout.setSpacing(8)
        brand = QLabel(APP_TITLE, top)
        brand.setObjectName("brand")
        tagline = QLabel("offline transcription with speaker labels", top)
        tagline.setObjectName("muted")
        top_layout.addWidget(brand)
        top_layout.addSpacing(4)
        top_layout.addWidget(tagline)
        top_layout.addStretch(1)

        self.open_files_button = QPushButton("Open files", top)
        self.open_files_button.setObjectName("flat")
        self.open_files_button.clicked.connect(self.open_files)
        self.open_folder_button = QPushButton("Open folder", top)
        self.open_folder_button.setObjectName("flat")
        self.open_folder_button.clicked.connect(self.open_folder)
        top_layout.addWidget(self.open_files_button)
        top_layout.addWidget(self.open_folder_button)
        top_layout.addSpacing(12)

        quality_label = QLabel("Quality", top)
        quality_label.setObjectName("muted")
        self.quality_box = QComboBox(top)
        self.quality_box.setMinimumWidth(130)
        self.quality_box.currentIndexChanged.connect(self._on_quality_changed)
        top_layout.addWidget(quality_label)
        top_layout.addWidget(self.quality_box)

        self.transcribe_button = QPushButton("Transcribe", top)
        self.transcribe_button.setObjectName("primary")
        self.transcribe_button.clicked.connect(self.start_transcription)
        self.stop_button = QPushButton("Stop", top)
        self.stop_button.clicked.connect(self.stop_transcription)
        self.stop_button.hide()
        top_layout.addWidget(self.transcribe_button)
        top_layout.addWidget(self.stop_button)

        self.settings_button = QToolButton(top)
        self.settings_button.setToolTip("Settings")
        self.settings_button.setAutoRaise(True)
        self.settings_button.clicked.connect(self.open_settings)
        top_layout.addSpacing(6)
        top_layout.addWidget(self.settings_button)
        root.addWidget(top)

        # Body: library on the left, the recording on the right.
        self.splitter = QSplitter(Qt.Orientation.Horizontal, central)
        root.addWidget(self.splitter, 1)

        sidebar = QFrame(self.splitter)
        sidebar.setObjectName("sidebar")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(10, 12, 6, 8)
        side_layout.setSpacing(6)
        side_header = QHBoxLayout()
        library_title = QLabel("LIBRARY", sidebar)
        library_title.setObjectName("section")
        self.library_count = QLabel("", sidebar)
        self.library_count.setObjectName("muted")
        self.clear_button = QPushButton("Clear", sidebar)
        self.clear_button.setObjectName("flat")
        self.clear_button.clicked.connect(self.clear_library)
        side_header.addWidget(library_title)
        side_header.addSpacing(6)
        side_header.addWidget(self.library_count)
        side_header.addStretch(1)
        side_header.addWidget(self.clear_button)
        side_layout.addLayout(side_header)

        self.library_stack = QStackedLayout()
        self.library_view = LibraryView(sidebar, self.theme)
        self.library_view.setModel(self.library_model)
        self.library_view.selectionModel().currentRowChanged.connect(self._on_current_changed)
        self.library_empty = QLabel("Drop recordings or folders here,\nor use Open files and Open folder.", sidebar)
        self.library_empty.setObjectName("hint")
        self.library_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.library_empty.setWordWrap(True)
        self.library_stack.addWidget(self.library_view)
        self.library_stack.addWidget(self.library_empty)
        side_layout.addLayout(self.library_stack, 1)
        self.library_model.rowsInserted.connect(lambda *_: self._refresh_library_count())
        self.library_model.rowsRemoved.connect(lambda *_: self._refresh_library_count())
        self.library_model.modelReset.connect(self._refresh_library_count)
        self.splitter.addWidget(sidebar)

        detail = QFrame(self.splitter)
        detail.setObjectName("detail")
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(0)

        header = QWidget(detail)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 14, 20, 6)
        header_layout.setSpacing(10)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.title_label = QLabel("", header)
        self.title_label.setObjectName("title")
        self.meta_label = QLabel("", header)
        self.meta_label.setObjectName("muted")
        titles.addWidget(self.title_label)
        titles.addWidget(self.meta_label)
        header_layout.addLayout(titles, 1)
        self.review_button = QPushButton("Review", header)
        self.review_button.setToolTip("Work through the spans where speech may be missing, with the audio")
        self.review_button.clicked.connect(self.open_review)
        self.folder_button = QPushButton("Show outputs", header)
        self.folder_button.setObjectName("flat")
        self.folder_button.setToolTip("Open the folder that holds the outputs")
        self.folder_button.clicked.connect(self.show_outputs)
        header_layout.addWidget(self.folder_button)
        header_layout.addWidget(self.review_button)
        detail_layout.addWidget(header)

        self.chips_widget = QWidget(detail)
        self.chips_layout = QHBoxLayout(self.chips_widget)
        self.chips_layout.setContentsMargins(20, 0, 20, 8)
        self.chips_layout.setSpacing(16)
        self.chips_layout.addStretch(1)
        detail_layout.addWidget(self.chips_widget)

        self.video_widget = QVideoWidget(detail)
        self.video_widget.setMinimumHeight(200)
        self.video_widget.setMaximumHeight(340)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.video_widget.hide()
        detail_layout.addWidget(self.video_widget)

        self.transcript_view = TranscriptView(detail, self.theme)
        self.transcript_view.seek_requested.connect(self.seek)
        self.transcript_view.mark_requested.connect(self.play_mark)
        self.job_card = JobStatusCard(detail, self.theme)
        self.detail_stack = QStackedLayout()
        self.detail_stack.addWidget(self.transcript_view)
        self.detail_stack.addWidget(self.job_card)
        detail_layout.addLayout(self.detail_stack, 1)

        self.player_bar = PlayerBar(detail, self.theme)
        self.player_bar.seek_requested.connect(self.seek)
        self.player_bar.skip_requested.connect(self.nudge)
        self.player_bar.play_toggled.connect(self.toggle_play)
        self.player_bar.follow_toggled.connect(self._on_follow_toggled)
        self.player_bar.rate_changed.connect(self._on_rate_changed)
        self.player_bar.volume_changed.connect(self._on_volume_changed)
        detail_layout.addWidget(self.player_bar)
        self.splitter.addWidget(detail)

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        sizes = self.settings.splitter if len(self.settings.splitter) == 2 else [300, 980]
        self.splitter.setSizes(sizes)

        self.setCentralWidget(central)
        self.batch_label = QLabel("", self)
        self.batch_label.setObjectName("muted")
        self.statusBar().addPermanentWidget(self.batch_label)
        self.statusBar().setSizeGripEnabled(False)

        for widget in (
            self.library_view,
            self.transcript_view,
            self.player_bar.timeline,
            self.transcribe_button,
            self.stop_button,
            self.open_files_button,
            self.open_folder_button,
            self.review_button,
            self.folder_button,
            self.clear_button,
        ):
            widget.installEventFilter(self)
        self._apply_theme_widgets()
        self._refresh_library_count()

        if len(self.settings.window_size) == 2:
            self.resize(max(900, self.settings.window_size[0]), max(600, self.settings.window_size[1]))

    def _build_player(self) -> None:
        self.player: QMediaPlayer | None = None
        self.audio_output: QAudioOutput | None = None
        try:
            self.player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.player.setAudioOutput(self.audio_output)
            self.player.setVideoOutput(self.video_widget)
            self.audio_output.setVolume(float(self.settings.volume))
            self.player.setPlaybackRate(float(self.settings.rate))
            self.player.positionChanged.connect(self._on_position_changed)
            self.player.durationChanged.connect(self._on_duration_changed)
            self.player.playbackStateChanged.connect(self._on_playback_state_changed)
            self.player.hasVideoChanged.connect(self._on_has_video_changed)
            self.player.errorOccurred.connect(self._on_player_error)
        except Exception as exc:  # noqa: BLE001 - any multimedia failure disables playback
            self.player = None
            self.audio_output = None
            self._player_error = f"{type(exc).__name__}: {exc}"
        self.player_bar.set_volume(self.settings.volume)
        self.player_bar.set_rate(self.settings.rate)
        self.player_bar.set_follow(self.settings.follow)
        self.transcript_view.set_follow(self.settings.follow)

    def _apply_theme_widgets(self) -> None:
        text_colour = self.palette().color(self.palette().ColorRole.Text)
        self.settings_button.setIcon(make_icon("gear", self.theme.muted))
        self.open_files_button.setIcon(make_icon("file", text_colour, 16))
        self.open_folder_button.setIcon(make_icon("folder", text_colour, 16))
        self.review_button.setIcon(make_icon("flag", self.theme.warning, 16))
        self.stop_button.setIcon(make_icon("stop", text_colour, 14))
        self.library_view.set_theme(self.theme)
        self.transcript_view.set_theme(self.theme)
        self.job_card.set_theme(self.theme)
        self.player_bar.set_theme(self.theme)
        if self._current_doc is not None:
            self._refresh_chips(self._current_doc)

    # ----- library --------------------------------------------------------------------

    def add_paths(self, paths: Sequence[Path]) -> int:
        """Add recordings and folders to the library; returns how many were added."""
        added = self.library_model.add_paths([Path(p) for p in paths])
        self._refresh_library_count()
        if added and self._current_row is None:
            self.library_view.setCurrentIndex(self.library_model.index(added[0], 0))
        if added:
            noun = "recording" if len(added) == 1 else "recordings"
            self.set_status(f"Added {len(added)} {noun}.")
        elif paths:
            self.set_status("Nothing new to add: no recordings found, or all are in the library already.")
        return len(added)

    def clear_library(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.set_status("Stop the batch before clearing the library.")
            return
        self.library_model.clear()
        self._current_row = None
        self._show_empty_detail()
        self._refresh_library_count()

    def remove_selected(self) -> None:
        rows = self.library_view.selected_rows()
        if not rows:
            return
        if self.worker is not None and self.worker.isRunning():
            self.set_status("Stop the batch before removing recordings.")
            return
        self.library_model.remove_rows(rows)
        self._current_row = None
        if self.library_model.rowCount() == 0:
            self._show_empty_detail()
        else:
            self.library_view.setCurrentIndex(self.library_model.index(min(rows[0], self.library_model.rowCount() - 1), 0))

    def _refresh_library_count(self) -> None:
        count = self.library_model.rowCount()
        self.library_count.setText(str(count) if count else "")
        self.library_stack.setCurrentIndex(0 if count else 1)
        self.clear_button.setVisible(count > 0)
        self._refresh_transcribe_button()

    def open_files(self) -> None:
        start = str(Path(self.settings.library[-1]).parent) if self.settings.library else str(Path.home())
        files, _ = QFileDialog.getOpenFileNames(self, "Open recordings", start, _file_filter())
        if files:
            self.add_paths([Path(f) for f in files])

    def open_folder(self) -> None:
        start = str(Path(self.settings.library[-1]).parent) if self.settings.library else str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Open a folder of recordings", start)
        if folder:
            self.add_paths([Path(folder)])

    def _on_current_changed(self, current, _previous) -> None:
        row = current.row() if current is not None and current.isValid() else -1
        if row < 0:
            return
        self.select_row(row)

    def select_row(self, row: int) -> None:
        """Show the recording at `row`: its transcript when it has one, and load it to play."""
        if not (0 <= row < self.library_model.rowCount()):
            return
        self._current_row = row
        item = self.library_model.item(row)
        self._load_item(item)

    def current_row(self) -> int | None:
        return self._current_row

    def current_document(self) -> dict[str, Any] | None:
        return self._current_doc

    # ----- detail ---------------------------------------------------------------------

    def _show_job_card(self, item: MediaItem, row: int) -> None:
        """Put the job card in front of the transcript pane for a queued or running recording."""
        position = self._batch_rows.index(row) + 1 if row in self._batch_rows else 1
        total = max(1, len(self._batch_rows))
        if item.status == STATUS_RUNNING:
            plan_lines = self._batch_plan.describe()[:3] if self._batch_plan is not None else []
            self.job_card.show_running(item.name, position, total, plan_lines)
            self.job_card.set_partials(self._partials.get(row, []))
        else:
            self.job_card.show_queued(item.name, position, total)
        self.detail_stack.setCurrentWidget(self.job_card)

    def _show_transcript_pane(self) -> None:
        self.job_card.stop()
        self.detail_stack.setCurrentWidget(self.transcript_view)

    def job_card_visible(self) -> bool:
        return self.detail_stack.currentWidget() is self.job_card

    def _show_empty_detail(self) -> None:
        self._current_doc = None
        self._current_path = None
        self.title_label.setText("No recording selected")
        self.meta_label.setText("Open a recording or a folder to begin.")
        self._show_transcript_pane()
        self.transcript_view.set_document(None)
        self.player_bar.set_marks([])
        self.player_bar.set_peaks(None)
        self.player_bar.set_duration(0.0)
        self.player_bar.set_position(0.0)
        self.player_bar.set_available(False)
        self.review_button.setEnabled(False)
        self.folder_button.setEnabled(False)
        self._clear_chips()
        if self.player is not None:
            self.player.stop()
            self.player.setSource(QUrl())
        self.video_widget.hide()
        self.setWindowTitle(APP_TITLE)

    def _load_item(self, item: MediaItem) -> None:
        doc = self._load_document_for(item)
        self._current_doc = doc
        self.title_label.setText(item.name)
        self.setWindowTitle(f"{item.name}  |  {APP_TITLE}")
        self.folder_button.setEnabled(True)
        if doc is not None:
            session = None
            paths = output_paths(item.path, self.settings.output_dir_or_none)
            if item.transcript_path is not None:
                session_path = paths.review.with_name(f"{paths.review.stem}.session.json")
                session = read_session_entries(session_path, len(doc.get("marks", [])))
            self.transcript_view.set_document(doc, session)
            self.player_bar.set_marks([(float(m["start"]), float(m["end"])) for m in doc.get("marks", [])])
            overview = doc.get("overview") or {}
            self.player_bar.set_peaks(overview.get("peaks"), int(overview.get("scale", 100)))
            self._duration_s = float(doc.get("duration_s", 0.0))
            self.player_bar.set_duration(self._duration_s)
            self._refresh_chips(doc)
            marks = int(doc.get("review", {}).get("marks", 0))
            self.review_button.setEnabled(marks > 0 and paths.review.is_file())
            self.review_button.setText(f"Review ({marks})" if marks else "Review")
            self.meta_label.setText(self._meta_text(item, doc))
        else:
            self.transcript_view.set_document(None)
            self.transcript_view.setPlaceholderText(
                "Not transcribed yet. Choose a quality level and press Transcribe; the recording plays meanwhile."
            )
            self.player_bar.set_marks([])
            self.player_bar.set_peaks(None)
            self._duration_s = float(item.duration_s or 0.0)
            self.player_bar.set_duration(self._duration_s)
            self._clear_chips()
            self.review_button.setEnabled(False)
            self.review_button.setText("Review")
            self.meta_label.setText(self._meta_text(item, None))
        if doc is None and item.status in (STATUS_QUEUED, STATUS_RUNNING):
            row = self.library_model.row_for_path(item.path)
            self._show_job_card(item, row if row is not None else 0)
        else:
            self._show_transcript_pane()
        self._set_source(item.path)

    def _meta_text(self, item: MediaItem, doc: dict[str, Any] | None) -> str:
        parts: list[str] = []
        duration = float(doc.get("duration_s", 0.0)) if doc is not None else float(item.duration_s or 0.0)
        if duration > 0.0:
            parts.append(clock(duration, tenths=False))
        if doc is not None:
            labelled = [s for s in doc.get("speakers", []) if s.get("label") is not None]
            parts.append(f"{len(labelled)} speaker" + ("" if len(labelled) == 1 else "s"))
            try:
                parts.append(profile_for(str(doc.get("profile", ""))).title)
            except KeyError:
                pass
            marks = int(doc.get("review", {}).get("marks", 0))
            share = 100.0 * float(doc.get("review", {}).get("fraction", 0.0))
            parts.append(f"{marks} span{'s' if marks != 1 else ''} to review, {share:.0f}% of the audio" if marks else "nothing to review")
            if doc.get("speaker_failure"):
                parts.append("speaker labelling did not complete")
            if approximate_word_times((doc.get("engines") or {}).get("detector")):
                parts.append("detector word times approximate")
        elif item.status == STATUS_RUNNING:
            parts.append(item.message or "Working")
        elif item.error:
            parts.append(f"Failed: {item.error}")
        else:
            parts.append("not transcribed yet")
        parts.append("video" if item.video else "audio")
        return "   |   ".join(parts)

    def _load_document_for(self, item: MediaItem) -> dict[str, Any] | None:
        transcript = output_paths(item.path, self.settings.output_dir_or_none).transcript
        if not transcript.is_file():
            return None
        try:
            doc = load_document(transcript)
        except (OSError, ValueError):
            return None
        if str(doc.get("source", {}).get("name", "")) != item.path.name:
            return None
        return doc

    def _clear_chips(self) -> None:
        while self.chips_layout.count() > 1:
            child = self.chips_layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.deleteLater()

    def _refresh_chips(self, doc: dict[str, Any]) -> None:
        self._clear_chips()
        names = speaker_names(doc)
        muted = self.theme.muted.name()
        index = 0
        for entry in doc.get("speakers", []):
            label = entry.get("label")
            if label is None:
                continue
            chip = SpeakerChip(str(label), names.get(label, label), int(entry.get("words", 0)),
                               self.theme.speaker(index).name(), muted, self.chips_widget)
            chip.rename_requested.connect(self.rename_speaker)
            self.chips_layout.insertWidget(index, chip)
            index += 1
        unlabelled = [e for e in doc.get("speakers", []) if e.get("label") is None]
        if unlabelled:
            words = int(unlabelled[0].get("words", 0))
            hint = QLabel(f'<span style="color:{muted};">{words:,} words without a speaker</span>', self.chips_widget)
            hint.setTextFormat(Qt.TextFormat.RichText)
            self.chips_layout.insertWidget(index, hint)

    def rename_speaker(self, label: str) -> None:
        """Rename a speaker in the current document and render its outputs again."""
        if self._current_doc is None or self._current_row is None:
            return
        names = speaker_names(self._current_doc)
        typed, accepted = QInputDialog.getText(self, "Rename speaker", "Name for this speaker:",
                                               QLineEdit.EchoMode.Normal, names.get(label, label))
        if not accepted or not typed.strip():
            return
        self.apply_speaker_name(label, typed)

    def apply_speaker_name(self, label: str, name: str) -> None:
        """Rename without a dialog; writes the document and the text, Word and subtitle files."""
        if self._current_doc is None or self._current_row is None:
            return
        item = self.library_model.item(self._current_row)
        try:
            updated = set_speaker_name(self._current_doc, label, name)
        except (KeyError, ValueError) as exc:
            self.set_status(f"Could not rename: {exc}")
            return
        paths = output_paths(item.path, self.settings.output_dir_or_none)
        try:
            write_document(updated, paths.transcript)
            render_all(updated, paths.text, paths.docx, paths.subtitles, author=self.settings.author)
        except OSError as exc:
            self.set_status(f"Could not write the outputs: {exc}")
            return
        self._current_doc = updated
        position = self._position_s
        self.transcript_view.set_document(updated, None)
        self.transcript_view.set_position(position)
        self._refresh_chips(updated)
        self.set_status(f"Speaker renamed to {name.strip()}; text, Word and subtitles written again.")

    def show_outputs(self) -> None:
        if self._current_row is None:
            return
        item = self.library_model.item(self._current_row)
        folder = output_paths(item.path, self.settings.output_dir_or_none).transcript.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def open_review(self) -> None:
        """Open the verification screen on the current review set."""
        if self._current_row is None or self._current_doc is None:
            return
        from twinscribe.app.verify import VerifyWindow, load_review_set

        item = self.library_model.item(self._current_row)
        review_path = output_paths(item.path, self.settings.output_dir_or_none).review
        try:
            review = load_review_set(review_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.set_status(f"Cannot read the review set: {exc}")
            return
        if self.player is not None:
            self.player.pause()
        window = VerifyWindow(review, review_path, self.theme, parent=None)
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.resize(1120, 740)
        window.destroyed.connect(lambda *_: self._reload_session())
        window.show()
        self._review_window = window

    def _reload_session(self) -> None:
        if self._current_row is not None and 0 <= self._current_row < self.library_model.rowCount():
            position = self._position_s
            self._load_item(self.library_model.item(self._current_row))
            self.transcript_view.set_position(position)

    # ----- quality and transcription --------------------------------------------------

    def _refresh_quality_box(self) -> None:
        self.quality_box.blockSignals(True)
        self.quality_box.clear()
        levels = available_profiles(self.models, self._backends)
        if levels:
            for profile in levels:
                self.quality_box.addItem(profile.title, profile.name)
                self.quality_box.setItemData(self.quality_box.count() - 1, profile.description, Qt.ItemDataRole.ToolTipRole)
            wanted = self.quality_box.findData(self.settings.quality)
            self.quality_box.setCurrentIndex(wanted if wanted >= 0 else self.quality_box.findData("standard") if self.quality_box.findData("standard") >= 0 else 0)
            self.quality_box.setEnabled(True)
            self.quality_box.setToolTip(str(self.models.root))
        else:
            self.quality_box.addItem("No models", None)
            self.quality_box.setEnabled(False)
            if not any(self._backends.values()):
                self.quality_box.setToolTip("No detector library is installed; install the engines extra.")
            else:
                self.quality_box.setToolTip(
                    f"No complete model set under {self.models.root} for the installed libraries. "
                    "Open Settings to choose the folder."
                )
        self.quality_box.blockSignals(False)
        self._refresh_transcribe_button()

    def _on_quality_changed(self, index: int) -> None:
        name = self.quality_box.itemData(index)
        if name:
            self.settings.quality = str(name)

    def selected_profile(self) -> Profile | None:
        name = self.quality_box.currentData()
        if not name:
            return None
        try:
            return profile_for(str(name))
        except KeyError:
            return None

    def _refresh_transcribe_button(self) -> None:
        running = self.worker is not None and self.worker.isRunning()
        pending = self.library_model.pending_rows()
        self.transcribe_button.setEnabled(bool(pending) and self.selected_profile() is not None and not running)
        if not pending:
            self.transcribe_button.setToolTip("Every recording in the library has been transcribed")
        elif self.selected_profile() is None:
            self.transcribe_button.setToolTip("No complete model set was found; open Settings")
        else:
            noun = "recording" if len(pending) == 1 else "recordings"
            self.transcribe_button.setToolTip(f"Transcribe {len(pending)} {noun} not yet done")

    def start_transcription(self, rows: Sequence[int] | None = None) -> bool:
        """Queue every pending recording (or the given rows) and run the pipeline in a thread."""
        if self.worker is not None and self.worker.isRunning():
            self.set_status("A batch is already running.")
            return False
        profile = self.selected_profile()
        if profile is None:
            QMessageBox.information(
                self,
                "No models",
                f"No complete model set was found under\n{self.models.root}\n\nOpen Settings to choose the models folder.",
            )
            return False
        plan = self._plan_override
        if plan is None:
            plan = current_plan(self.settings.acceleration, self.settings.threads_or_none)
        if plan.detector_backend == BACKEND_NONE and self._engines is None:
            QMessageBox.information(
                self,
                "Engine libraries not installed",
                "No detector library is installed in this environment.\n\n"
                "Install the 'engines' extra of this package (faster-whisper with CTranslate2 where it has a wheel, "
                "sherpa-onnx everywhere), then try again. Playback still works.",
            )
            return False
        try:
            selection = select_level(profile.name, self.models, self._backends)
        except ModelsMissing as exc:
            QMessageBox.information(self, "Models missing", str(exc))
            return False
        placement = plan.placement_for(selection.backend)
        chosen = list(rows) if rows is not None else self.library_model.pending_rows()
        if not chosen:
            self.set_status("Nothing to transcribe.")
            return False
        self.library_model.set_queued(chosen)
        sources = [self.library_model.item(row).path for row in chosen]
        self.worker = PipelineWorker(
            chosen,
            sources,
            profile,
            self.models,
            out_dir=self.settings.output_dir_or_none,
            threads=self.settings.threads_or_none,
            author=self.settings.author,
            engines=self._engines,
            record_dir=self._record_dir,
            parent=self,
            plan=plan,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.partial.connect(self._on_partial)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.file_failed.connect(self._on_file_failed)
        self.worker.finished_all.connect(self._on_batch_finished)
        self._batch_total = len(chosen)
        self._batch_done = 0
        self._batch_rows = list(chosen)
        self._batch_plan = plan
        self._partials = {}
        if self._current_row in chosen:
            self._show_job_card(self.library_model.item(self._current_row), self._current_row)
        self.stop_button.show()
        self.transcribe_button.setEnabled(False)
        self.quality_box.setEnabled(False)
        self.batch_label.setText(f"0 of {self._batch_total}")
        where = placement.describe() if placement is not None else "processor"
        backend = "CTranslate2" if selection.backend == BACKEND_CT2 else "ONNX"
        self.set_status(
            f"Transcribing {self._batch_total} recording" + ("" if self._batch_total == 1 else "s")
            + f" at the {profile.title} level; detector {backend} on {where}; publisher on "
            f"{plan.publisher.describe()}."
        )
        self.worker.start()
        return True

    def stop_transcription(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.stop_button.setEnabled(False)
            self.set_status("Stopping after the current step.")

    def _on_progress(self, row: int, report: Progress) -> None:
        was_queued = 0 <= row < self.library_model.rowCount() and self.library_model.item(row).status != STATUS_RUNNING
        self.library_model.set_progress(row, report.fraction, report.message)
        item = self.library_model.item(row) if 0 <= row < self.library_model.rowCount() else None
        name = item.name if item is not None else ""
        percent = int(round(100 * report.fraction))
        self.batch_label.setText(f"{self._batch_done} of {self._batch_total}   |   {name}   |   {report.message} {percent}%")
        if row == self._current_row and item is not None:
            self.meta_label.setText(self._meta_text(item, None))
            if was_queued or not self.job_card_visible() or not self.job_card.running():
                self._show_job_card(item, row)
            self.job_card.update_progress(report)
            self.setWindowTitle(f"{percent}%  {item.name}  |  {APP_TITLE}")

    def _on_partial(self, row: int, segment) -> None:
        text = str(getattr(segment, "text", "")).strip()
        if not text:
            return
        entry = (float(segment.start), float(segment.end), text)
        self._partials.setdefault(row, []).append(entry)
        if row == self._current_row and self.job_card_visible():
            self.job_card.add_partial(*entry)

    def _on_file_done(self, row: int, result: FileResult) -> None:
        self._batch_done += 1
        self._partials.pop(row, None)
        self.library_model.set_done(row, result.document, result.outputs.transcript)
        if row == self._current_row:
            position = self._position_s
            self._load_item(self.library_model.item(row))
            self.transcript_view.set_position(position)
        self.set_status(
            f"{result.source.name}: {result.marks} span" + ("" if result.marks == 1 else "s")
            + f" to review, {result.elapsed_s:.0f} s."
        )

    def _on_file_failed(self, row: int, error_class: str, message: str) -> None:
        self._batch_done += 1
        self._partials.pop(row, None)
        self.library_model.set_failed(row, f"{error_class}: {message}")
        if row == self._current_row:
            self._load_item(self.library_model.item(row))
        self.set_status(f"{self.library_model.item(row).name} failed: {error_class}: {message}")

    def _on_batch_finished(self, result) -> None:
        self.library_model.reset_pending()
        self._partials = {}
        self._batch_rows = []
        if self._current_row is not None and 0 <= self._current_row < self.library_model.rowCount():
            item = self.library_model.item(self._current_row)
            self.setWindowTitle(f"{item.name}  |  {APP_TITLE}")
            if self.job_card_visible():
                self._load_item(item)
        app = QApplication.instance()
        if isinstance(app, QApplication):
            app.alert(self)
        failures = len(result.failures) if result is not None else 0
        completed = len(result.completed) if result is not None else 0
        cancelled = any(o.cancelled for o in result.outcomes) if result is not None else False
        self.stop_button.hide()
        self.stop_button.setEnabled(True)
        self.quality_box.setEnabled(True)
        self.batch_label.setText("")
        text = f"Batch finished: {completed} done"
        if failures:
            text += f", {failures} failed"
        if cancelled:
            text += ", stopped early"
        if result is not None and result.record_path is not None:
            text += f". Record: {result.record_path.name}"
        self.set_status(text + ".")
        self.worker = None
        self._refresh_transcribe_button()

    # ----- playback -------------------------------------------------------------------

    def _set_source(self, path: Path) -> None:
        self._stop_at_s = None
        self._position_s = 0.0
        self.player_bar.set_position(0.0)
        self.transcript_view.set_position(-1.0)
        if self.player is None:
            self.player_bar.set_available(False)
            return
        if self._current_path is not None and self._current_path == path:
            return
        self._current_path = path
        self._player_error = None
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.player_bar.set_available(True)
        self.player_bar.set_playing(False)

    def is_playing(self) -> bool:
        return self.player is not None and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def toggle_play(self) -> None:
        """Space: play or pause; free playback does not stop at a span end."""
        if self.player is None or self._current_path is None:
            return
        self._stop_at_s = None
        if self.is_playing():
            self.player.pause()
        else:
            self.player.play()

    def seek(self, seconds: float) -> None:
        """Move the playhead; a plain seek cancels any pending stop at a span end."""
        self._stop_at_s = None
        self._seek(seconds)

    def _seek(self, seconds: float) -> None:
        limit = self._duration_s if self._duration_s > 0 else max(0.0, seconds)
        seconds = min(max(0.0, seconds), limit)
        self._position_s = seconds
        self.player_bar.set_position(seconds)
        self.transcript_view.set_position(seconds)
        if self.player is not None and self._current_path is not None:
            self.player.setPosition(int(round(seconds * 1000.0)))

    def nudge(self, delta_s: float) -> None:
        self.seek(self._position_s + delta_s)

    def play_mark(self, index: int) -> None:
        """Play the padded span of one review mark, pausing at its end."""
        if self._current_doc is None:
            return
        marks = self._current_doc.get("marks", [])
        if not (0 <= index < len(marks)):
            return
        mark = marks[index]
        self.play_span(float(mark["start"]), float(mark["end"]))
        self.player_bar.timeline.set_current(index)

    def play_span(self, start: float, end: float) -> None:
        self._seek(start)
        if self.player is None or self._current_path is None:
            return
        self._stop_at_s = end
        self.player.play()
        self.set_status(f"Playing {clock(start)} to {clock(end)}.")

    def next_mark(self, backwards: bool = False) -> None:
        """J and K: play the next or the previous review span from the playhead."""
        if self._current_doc is None:
            return
        marks = self._current_doc.get("marks", [])
        if not marks:
            return
        if backwards:
            candidates = [i for i, m in enumerate(marks) if float(m["start"]) < self._position_s - 0.5]
            index = candidates[-1] if candidates else len(marks) - 1
        else:
            candidates = [i for i, m in enumerate(marks) if float(m["start"]) > self._position_s + 0.05]
            index = candidates[0] if candidates else 0
        self.play_mark(index)

    def _on_position_changed(self, position_ms: int) -> None:
        seconds = position_ms / 1000.0
        self._position_s = seconds
        self.player_bar.set_position(seconds)
        self.transcript_view.set_position(seconds)
        if self._stop_at_s is not None and seconds >= self._stop_at_s:
            self._stop_at_s = None
            if self.player is not None:
                self.player.pause()

    def _on_duration_changed(self, duration_ms: int) -> None:
        if duration_ms <= 0:
            return
        seconds = duration_ms / 1000.0
        if self._current_doc is None or self._duration_s <= 0.0:
            self._duration_s = seconds
            self.player_bar.set_duration(seconds)
        if self._current_row is not None:
            item = self.library_model.item(self._current_row)
            if not item.duration_s:
                self.library_model.set_duration(self._current_row, seconds)
                self.meta_label.setText(self._meta_text(item, self._current_doc))

    def _on_playback_state_changed(self, state) -> None:
        self.player_bar.set_playing(state == QMediaPlayer.PlaybackState.PlayingState)

    def _on_has_video_changed(self, has_video: bool) -> None:
        self.video_widget.setVisible(bool(has_video))

    def _on_player_error(self, error, message: str) -> None:
        if error == QMediaPlayer.Error.NoError:
            return
        self._player_error = message or str(error)
        self.set_status(f"Playback is not possible for this file: {self._player_error}")
        self.player_bar.set_available(False)

    def _on_follow_toggled(self, follow: bool) -> None:
        self.settings.follow = bool(follow)
        self.transcript_view.set_follow(bool(follow))
        self.player_bar.set_follow(bool(follow))

    def _on_rate_changed(self, rate: float) -> None:
        self.settings.rate = float(rate)
        if self.player is not None:
            self.player.setPlaybackRate(float(rate))

    def _on_volume_changed(self, volume: float) -> None:
        self.settings.volume = float(volume)
        if self.audio_output is not None:
            self.audio_output.setVolume(float(volume))

    # ----- settings -------------------------------------------------------------------

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        was_dark = self.settings.dark
        dialog.apply_to(self.settings)
        self.apply_settings(theme_changed=was_dark != self.settings.dark)

    def apply_settings(self, theme_changed: bool = False) -> None:
        """Re-read the models, the outputs location and the palette after the settings changed."""
        self.models = find_models(self.settings.models_dir or None)
        self.library_model.set_output_dir(self.settings.output_dir_or_none)
        for row in range(self.library_model.rowCount()):
            self.library_model.refresh_item(row)
        self._refresh_quality_box()
        if theme_changed:
            app = QApplication.instance()
            if isinstance(app, QApplication):
                self.theme = apply_theme(app, self.settings.dark)
                apply_styles(app, self.settings.dark)
            self._apply_theme_widgets()
        if self._current_row is not None and self._current_row < self.library_model.rowCount():
            position = self._position_s
            self._load_item(self.library_model.item(self._current_row))
            self.transcript_view.set_position(position)
        save_settings(self.settings)
        self.set_status("Settings saved.")

    def set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    # ----- keys, drops, lifecycle -----------------------------------------------------

    def handle_key(self, key: int, modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier) -> bool:
        """Apply one of the keys of the window; True when the key was one of them."""
        control = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if control and key == Qt.Key.Key_O:
            if shift:
                self.open_folder()
            else:
                self.open_files()
            return True
        if modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
            return False
        if key == Qt.Key.Key_Space:
            self.toggle_play()
        elif key == Qt.Key.Key_Left:
            self.nudge(-NUDGE_S)
        elif key == Qt.Key.Key_Right:
            self.nudge(NUDGE_S)
        elif key == Qt.Key.Key_J:
            self.next_mark()
        elif key == Qt.Key.Key_K:
            self.next_mark(backwards=True)
        elif key == Qt.Key.Key_F:
            self._on_follow_toggled(not self.transcript_view.follow())
        elif key == Qt.Key.Key_Delete:
            self.remove_selected()
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
            if event.matches(QKeySequence.StandardKey.Open):
                self.open_files()
                return True
            if self.handle_key(event.key(), event.modifiers()):
                return True
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 (Qt virtual)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 (Qt virtual)
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.add_paths(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt virtual)
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(15000)
        if self.player is not None:
            self._stop_at_s = None
            self.player.stop()
            self.player.setSource(QUrl())
        self.settings.library = [str(item.path) for item in self.library_model.items()]
        self.settings.splitter = list(self.splitter.sizes())
        self.settings.window_size = [self.width(), self.height()]
        try:
            save_settings(self.settings)
        except OSError:
            pass
        super().closeEvent(event)


# ----- entry point ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Command line: recordings or folders to add, models root, palette, screenshot."""
    parser = argparse.ArgumentParser(
        prog="python -m twinscribe.app",
        description="Open the twinscribe window.",
    )
    parser.add_argument("paths", nargs="*", type=Path, help="recordings or folders to add to the library")
    parser.add_argument("--models", type=Path, default=None, help="models root folder")
    parser.add_argument("--dark", action="store_true", help="use the dark palette")
    parser.add_argument("--shot", type=Path, default=None, metavar="OUT_PNG",
                        help="render the window, save a screenshot after a short delay, exit")
    parser.add_argument("--select", type=int, default=None, metavar="ROW",
                        help="select this library row after the paths are added")
    parser.add_argument("--play", action="store_true", help="start playback at once (with --shot, for a moving playhead)")
    parser.add_argument("--seek", type=float, default=None, metavar="SECONDS", help="seek to this time after loading")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the window; returns the process exit status."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    settings = load_settings()
    if args.dark:
        settings.dark = True
    if args.models is not None:
        settings.models_dir = str(args.models)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_TITLE)
    theme = apply_theme(app, dark=settings.dark)
    apply_styles(app, dark=settings.dark)
    window = MainWindow(settings, theme)
    if len(settings.window_size) != 2:
        window.resize(1280, 820)
    if args.paths:
        window.add_paths(list(args.paths))
    if args.select is not None:
        window.library_view.setCurrentIndex(window.library_model.index(args.select, 0))
    window.show()
    if args.seek is not None:
        QTimer.singleShot(400, lambda: window.seek(float(args.seek)))
    if args.play:
        QTimer.singleShot(500, window.toggle_play)

    if args.shot is not None:
        shot_path: Path = args.shot

        def take_shot() -> None:
            saved = window.grab().save(str(shot_path))
            if not saved:
                print(f"could not save screenshot to {shot_path}", file=sys.stderr)
            window.close()
            app.quit()

        QTimer.singleShot(SHOT_DELAY_MS + (2500 if args.play else 0), take_shot)

    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
