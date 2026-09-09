"""The models dialog: what the store lacks for the chosen quality levels, where it will go,
how large it is and under which licences, and a Download button that fetches it in a thread
with progress. The one place the window reaches the network, and only on that button.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from twinscribe.fetch import (
    STAGE_DONE,
    STAGE_EXTRACT,
    Cancelled,
    Progress,
    fetch_specs,
    level_names,
    missing_for_levels,
    proposed_root,
)
from twinscribe.models import ModelSet, ModelSpec, find_models
from twinscribe.profiles import profile_for

COLUMNS: tuple[str, ...] = ("Model", "Role", "Size", "Licence", "From")
FetchFn = Callable[..., object]


def _host(url: str) -> str:
    return url.split("//", 1)[-1].split("/", 1)[0]


def _size_text(size_mb: int) -> str:
    return f"{size_mb / 1000:.1f} GB" if size_mb >= 1000 else f"{size_mb} MB"


class FetchThread(QThread):
    """Runs the fetch off the window's thread; progress and the outcome come back as signals."""

    progress = Signal(object)
    finished_ok = Signal()
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, specs: Sequence[ModelSpec], root: Path, fetch: FetchFn = fetch_specs, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._specs = list(specs)
        self._root = Path(root)
        self._fetch = fetch
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: D401 (Qt virtual)
        try:
            self._fetch(self._specs, self._root, progress=self.progress.emit, cancel=lambda: self._stop)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - reported to the person, never lost
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished_ok.emit()


class ModelsDialog(QDialog):
    """Offer the models a quality level lacks and fetch them on request.

    `fetched` is emitted with the root once a fetch ended (complete, stopped or cancelled), so the
    window can read the store again. `fetch` is the function the thread runs; tests replace it.
    `explicit` says the store was named by the person (settings), so it is proposed as the folder.
    """

    fetched = Signal(object)

    def __init__(
        self,
        models: ModelSet | None = None,
        parent: QWidget | None = None,
        fetch: FetchFn = fetch_specs,
        explicit: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Models")
        self.setMinimumWidth(720)
        self._fetch = fetch
        self._thread: FetchThread | None = None
        self.models = models if models is not None else find_models(None)
        self.root = proposed_root(self.models, explicit)
        self.specs: list[ModelSpec] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        intro = QLabel(
            "TwinScribe transcribes with open models that are fetched once, from the sources the "
            "catalogue names, and pinned by digest. Nothing else in the program reaches the network.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        levels_row = QHBoxLayout()
        levels_row.addWidget(QLabel("Quality levels to complete:", self))
        self.level_checks: dict[str, QCheckBox] = {}
        for name in level_names():
            check = QCheckBox(profile_for(name).title, self)
            check.setChecked(name == "standard")
            check.setToolTip(profile_for(name).description)
            check.toggled.connect(lambda _checked=False: self.refresh())
            self.level_checks[name] = check
            levels_row.addWidget(check)
        levels_row.addStretch(1)
        layout.addLayout(levels_row)

        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Folder:", self))
        self.folder_edit = QLineEdit(str(self.root), self)
        browse = QPushButton("Browse", self)
        browse.clicked.connect(self._browse)
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(browse)
        layout.addLayout(folder_row)

        self.summary = QLabel("", self)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.progress = QProgressBar(self)
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.status = QLabel("", self)
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        self.download_button = QPushButton("Download", self)
        self.download_button.setObjectName("primary")
        self.download_button.clicked.connect(lambda _checked=False: self.start())
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.hide()
        self.later_button = QPushButton("Later", self)
        self.later_button.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(self.later_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.download_button)
        layout.addLayout(buttons)
        self.refresh()

    # ----- the plan -------------------------------------------------------------------

    def chosen_levels(self) -> list[str]:
        return [name for name, check in self.level_checks.items() if check.isChecked()]

    def refresh(self) -> None:
        """Recompute what the chosen levels lack under the folder and show it."""
        root = Path(self.folder_edit.text().strip() or str(self.root))
        store = find_models(root)
        self.specs = missing_for_levels(self.chosen_levels(), store)
        self.table.setRowCount(len(self.specs))
        for row, spec in enumerate(self.specs):
            values = (spec.title, spec.role, _size_text(spec.size_mb),
                      spec.licence, ", ".join(sorted({_host(source.url) for source in spec.sources})))
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setToolTip(spec.credit + (f". {spec.note}" if spec.note else ""))
                self.table.setItem(row, column, item)
        total_mb = sum(spec.size_mb for spec in self.specs)
        if self.specs:
            noun = "model" if len(self.specs) == 1 else "models"
            self.summary.setText(f"{len(self.specs)} {noun} to fetch, about {_size_text(total_mb)}, into {root}.")
        else:
            self.summary.setText(f"Nothing to fetch: the chosen levels are complete under {root}.")
        self.download_button.setEnabled(bool(self.specs) and self._thread is None)

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Models folder", self.folder_edit.text() or str(self.root))
        if chosen:
            self.folder_edit.setText(chosen)
            self.refresh()

    # ----- the fetch ------------------------------------------------------------------

    def start(self) -> FetchThread | None:
        """Begin the fetch in a thread; returns it, or None when there is nothing to do."""
        self.refresh()
        if not self.specs or self._thread is not None:
            return None
        self.root = Path(self.folder_edit.text().strip() or str(self.root))
        self._thread = FetchThread(self.specs, self.root, self._fetch, self)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished_ok.connect(self._on_finished)
        self._thread.failed.connect(self._on_failed)
        self._thread.cancelled.connect(self._on_cancelled)
        for check in self.level_checks.values():
            check.setEnabled(False)
        self.folder_edit.setEnabled(False)
        self.download_button.setEnabled(False)
        self.later_button.hide()
        self.cancel_button.show()
        self.progress.setValue(0)
        self.progress.show()
        self.status.setText("Starting")
        self._thread.start()
        return self._thread

    def cancel(self) -> None:
        if self._thread is not None:
            self._thread.cancel()
            self.status.setText("Stopping after the current chunk")

    def running(self) -> bool:
        """True from Download until the outcome has been shown."""
        return self._thread is not None

    def _on_progress(self, report: Progress) -> None:
        total = max(1, report.files_total)
        if report.stage == STAGE_DONE:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(round(1000 * report.files_done / total)))
            self.status.setText(f"{report.files_done} of {report.files_total} in place: {report.key}/{report.file}")
            return
        files = f"{min(report.files_done + 1, report.files_total)} of {report.files_total}"
        if report.stage == STAGE_EXTRACT:
            self.progress.setRange(0, 0)
            self.status.setText(f"{files}: extracting {report.file}")
        elif report.fraction is None:
            self.progress.setRange(0, 0)
            self.status.setText(f"{files}: {report.key}/{report.file}, {report.done_bytes / 1e6:.0f} MB")
        else:
            self.progress.setRange(0, 1000)
            overall = (report.files_done + report.fraction) / total
            self.progress.setValue(int(round(1000 * overall)))
            self.status.setText(f"{files}: {report.key}/{report.file}, {int(round(100 * report.fraction))}%")

    def _done(self, text: str) -> None:
        self._thread = None
        for check in self.level_checks.values():
            check.setEnabled(True)
        self.folder_edit.setEnabled(True)
        self.cancel_button.hide()
        self.later_button.show()
        self.progress.setRange(0, 1000)
        self.status.setText(text)
        self.refresh()

    def _on_finished(self) -> None:
        self.progress.setValue(1000)
        self._done("Done; the models are in place.")
        self.fetched.emit(self.root)
        self.later_button.setText("Close")

    def _on_failed(self, message: str) -> None:
        self._done(f"The fetch stopped: {message}. What was fetched is kept; Download continues from there.")
        self.fetched.emit(self.root)

    def _on_cancelled(self) -> None:
        self._done("Cancelled; what was fetched is kept, and Download continues from there.")
        self.fetched.emit(self.root)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt virtual)
        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel()
            self._thread.wait(15000)
        super().closeEvent(event)
