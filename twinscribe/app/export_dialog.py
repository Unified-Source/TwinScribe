"""Export dialog: the formats to write, where, and under what name, for one recording's
transcript document; the rendered formats are produced again from the document, so a renamed
speaker or a listener's edit is carried, and the JSON files are copied as they are.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from twinscribe.outputs.export import DEFAULT_FORMATS, FORMAT_REVIEW, FORMAT_RUN, FORMATS, export_outputs, transcript_text
from twinscribe.pipeline import OutputPaths


class ExportDialog(QDialog):
    """Choose formats, a folder and a name; Export writes them and reports what was written."""

    def __init__(
        self,
        doc: Mapping[str, Any],
        paths: OutputPaths | None,
        author: str = "",
        parent: QWidget | None = None,
        destination: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setMinimumWidth(520)
        self.doc = dict(doc)
        self.paths = paths
        self.author = author
        self.written: list[Path] = []

        source = self.doc.get("source", {})
        default_stem = str(source.get("outputs") or str(source.get("name", "transcript")).rsplit(".", 1)[0])
        default_folder = destination if destination is not None else (paths.transcript.parent if paths is not None else Path.cwd())

        form = QFormLayout(self)
        form.setSpacing(10)
        form.addRow(QLabel(f"Export {source.get('name', 'the transcript')}", self))

        formats_box = QVBoxLayout()
        self.checks: dict[str, QCheckBox] = {}
        for key, title, suffix in FORMATS:
            check = QCheckBox(f"{title}  ({suffix})", self)
            check.setChecked(key in DEFAULT_FORMATS)
            if key in (FORMAT_REVIEW, FORMAT_RUN):
                present = paths is not None and getattr(paths, "review" if key == FORMAT_REVIEW else "run").is_file()
                check.setEnabled(present)
                if not present:
                    check.setToolTip("Not available for this recording")
            self.checks[key] = check
            formats_box.addWidget(check)
        form.addRow("Formats", formats_box)

        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit(str(default_folder), self)
        browse = QPushButton("Browse", self)
        browse.clicked.connect(self._browse)
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(browse)
        form.addRow("Folder", folder_row)

        self.stem_edit = QLineEdit(default_stem, self)
        self.stem_edit.setToolTip("The files are named <name> plus each format's suffix")
        form.addRow("File name", self.stem_edit)

        self.status = QLabel("", self)
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        form.addRow("", self.status)

        buttons = QDialogButtonBox(self)
        self.copy_button = buttons.addButton("Copy transcript text", QDialogButtonBox.ButtonRole.ActionRole)
        self.copy_button.clicked.connect(self.copy_text)
        self.export_button = buttons.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole)
        self.export_button.clicked.connect(self.export)
        close = buttons.addButton(QDialogButtonBox.StandardButton.Close)
        close.clicked.connect(self.reject)
        form.addRow(buttons)

    def selected_formats(self) -> tuple[str, ...]:
        return tuple(key for key, check in self.checks.items() if check.isChecked() and check.isEnabled())

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Export folder", self.folder_edit.text() or str(Path.home()))
        if chosen:
            self.folder_edit.setText(chosen)

    def copy_text(self) -> None:
        QApplication.clipboard().setText(transcript_text(self.doc))
        self.status.setText("The transcript text is on the clipboard.")

    def export(self) -> list[Path]:
        """Write the chosen formats; the dialog stays open so the result can be read."""
        formats = self.selected_formats()
        if not formats:
            self.status.setText("Choose at least one format.")
            return []
        sources: dict[str, Path] = {}
        if self.paths is not None:
            sources = {"transcript": self.paths.transcript, "review": self.paths.review, "run": self.paths.run}
        try:
            self.written = export_outputs(
                self.doc, self.folder_edit.text().strip() or ".", self.stem_edit.text().strip(), formats,
                author=self.author, sources=sources,
            )
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText(f"Export failed: {exc}")
            return []
        names = ", ".join(p.name for p in self.written)
        self.status.setText(f"Written to {self.folder_edit.text().strip()}: {names}")
        return self.written
