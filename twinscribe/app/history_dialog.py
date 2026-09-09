"""History dialog: every recording transcribed on this machine, newest first, with what was
produced and where; from here a recording goes back into the library, its outputs folder
opens, its outputs are exported again, or the entry is dropped.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from twinscribe.app.export_dialog import ExportDialog
from twinscribe.app.theme import Theme
from twinscribe.history import HistoryEntry, clear_history, read_history, remove_entries
from twinscribe.outputs.transcript_doc import clock, load_document
from twinscribe.pipeline import output_paths
from twinscribe.profiles import profile_for

COLUMNS: tuple[str, ...] = ("When", "Recording", "Duration", "Level", "Marks", "Speakers", "Outputs")


def _level_title(name: str) -> str:
    try:
        return profile_for(name).title
    except KeyError:
        return name or "-"


def _when(stamp: str) -> str:
    """2026-09-09 10:23 from an ISO stamp, or the stamp as given."""
    if len(stamp) >= 16 and stamp[10] == "T":
        return stamp[:10] + " " + stamp[11:16]
    return stamp


class HistoryDialog(QDialog):
    """The table of past transcriptions and the actions on a selection."""

    add_requested = Signal(list)

    def __init__(self, theme: Theme, parent: QWidget | None = None, author: str = "", history_path: Path | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("History")
        self.setMinimumSize(860, 420)
        self.theme = theme
        self.author = author
        self.history_path = history_path
        self.entries: list[HistoryEntry] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        self.summary = QLabel("", self)
        self.summary.setObjectName("muted")
        layout.addWidget(self.summary)

        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._refresh_buttons)
        self.table.doubleClicked.connect(lambda _index: self.add_to_library())
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add to library", self)
        self.add_button.setObjectName("primary")
        self.add_button.clicked.connect(self.add_to_library)
        self.folder_button = QPushButton("Show outputs", self)
        self.folder_button.clicked.connect(self.show_outputs)
        self.export_button = QPushButton("Export", self)
        self.export_button.clicked.connect(self.export_selected)
        self.remove_button = QPushButton("Remove from history", self)
        self.remove_button.setObjectName("flat")
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button = QPushButton("Clear history", self)
        self.clear_button.setObjectName("flat")
        self.clear_button.clicked.connect(self.clear_all)
        close = QPushButton("Close", self)
        close.clicked.connect(self.accept)
        for button in (self.add_button, self.folder_button, self.export_button, self.remove_button, self.clear_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.refresh()

    # ----- data -----------------------------------------------------------------------

    def refresh(self) -> None:
        """Read the history again and rebuild the table."""
        self.entries = read_history(self.history_path)
        self.table.setRowCount(len(self.entries))
        muted = QColor(self.theme.muted)
        for row, entry in enumerate(self.entries):
            present = entry.outputs_present
            values = (
                _when(entry.produced_utc),
                entry.name,
                clock(entry.duration_s, tenths=False),
                _level_title(entry.profile),
                str(entry.marks),
                str(entry.speakers),
                str(entry.folder) + ("" if present else "   (outputs missing)"),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if not present:
                    item.setForeground(muted)
                if column == 1:
                    item.setToolTip(entry.path)
                self.table.setItem(row, column, item)
        total = sum(e.duration_s for e in self.entries)
        noun = "recording" if len(self.entries) == 1 else "recordings"
        self.summary.setText(
            f"{len(self.entries)} {noun} transcribed, {clock(total, tenths=False)} of audio in all."
            if self.entries else "Nothing has been transcribed on this machine yet."
        )
        self._refresh_buttons()

    def selected_entries(self) -> list[HistoryEntry]:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        return [self.entries[row] for row in rows if 0 <= row < len(self.entries)]

    def _refresh_buttons(self) -> None:
        selected = self.selected_entries()
        self.add_button.setEnabled(any(Path(e.path).is_file() for e in selected))
        self.folder_button.setEnabled(len(selected) == 1 and selected[0].folder.is_dir())
        self.export_button.setEnabled(len(selected) == 1 and selected[0].outputs_present)
        self.remove_button.setEnabled(bool(selected))
        self.clear_button.setEnabled(bool(self.entries))

    # ----- actions --------------------------------------------------------------------

    def add_to_library(self) -> None:
        """Hand the selected recordings that still exist to the window."""
        paths = [Path(e.path) for e in self.selected_entries() if Path(e.path).is_file()]
        if paths:
            self.add_requested.emit(paths)
            self.summary.setText(f"{len(paths)} added to the library.")

    def show_outputs(self) -> None:
        selected = self.selected_entries()
        if len(selected) == 1 and selected[0].folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(selected[0].folder)))

    def export_selected(self, modal: bool = True) -> ExportDialog | None:
        """Open the export dialog on the selected entry's transcript document; `modal` False
        shows it without blocking, for tests."""
        selected = self.selected_entries()
        if len(selected) != 1 or not selected[0].outputs_present:
            return None
        entry = selected[0]
        try:
            doc = load_document(entry.transcript_path)
        except (OSError, ValueError) as exc:
            self.summary.setText(f"Cannot read the transcript document: {exc}")
            return None
        paths = output_paths(Path(entry.path), entry.folder)
        dialog = ExportDialog(doc, paths, author=self.author, parent=self)
        if modal:
            dialog.exec()
        else:
            dialog.show()
        return dialog

    def remove_selected(self) -> None:
        selected = self.selected_entries()
        if not selected:
            return
        remove_entries([e.path for e in selected], self.history_path)
        self.refresh()

    def clear_all(self) -> None:
        if not self.entries:
            return
        answer = QMessageBox.question(
            self, "Clear history", "Forget every entry? The recordings and their outputs are not touched.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            clear_history(self.history_path)
            self.refresh()
