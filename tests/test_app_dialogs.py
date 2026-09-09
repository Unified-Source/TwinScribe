"""Tests for the export and history dialogs, run under the offscreen platform with a transcript
document beside a synthetic recording."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from tests._fixtures import make_document  # noqa: E402
from twinscribe import audio  # noqa: E402
from twinscribe.app.export_dialog import ExportDialog  # noqa: E402
from twinscribe.app.history_dialog import COLUMNS, HistoryDialog  # noqa: E402
from twinscribe.app.theme import theme_for  # noqa: E402
from twinscribe.history import HistoryEntry, append_entry, read_history  # noqa: E402
from twinscribe.outputs import render_all  # noqa: E402
from twinscribe.outputs.export import DEFAULT_FORMATS, FORMAT_REVIEW, FORMAT_RUN, FORMAT_VTT  # noqa: E402
from twinscribe.outputs.transcript_doc import write_document  # noqa: E402
from twinscribe.pipeline import output_paths  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    existing = QApplication.instance()
    return existing if existing is not None else QApplication([])


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    folder = tmp_path / "media"
    folder.mkdir()
    source = audio.synthetic_wav(folder / "call.wav", 30.0)
    doc = make_document("call.wav")
    paths = output_paths(source)
    write_document(doc, paths.transcript)
    render_all(doc, paths.text, paths.docx, paths.subtitles)
    return source


def test_export_dialog_formats_and_export(app: QApplication, recording: Path, tmp_path: Path) -> None:
    paths = output_paths(recording)
    doc = make_document("call.wav")
    dialog = ExportDialog(doc, paths, author="A Person", destination=tmp_path / "exports")
    assert set(dialog.selected_formats()) == set(DEFAULT_FORMATS)
    # The review list and the run record are absent beside this recording, so they cannot be chosen.
    assert not dialog.checks[FORMAT_REVIEW].isEnabled() and not dialog.checks[FORMAT_RUN].isEnabled()
    dialog.checks[FORMAT_VTT].setChecked(True)
    assert dialog.stem_edit.text() == "call" and dialog.folder_edit.text() == str(tmp_path / "exports")
    written = dialog.export()
    assert [p.name for p in written] == ["call.txt", "call.docx", "call.srt", "call.vtt"]
    assert "Written to" in dialog.status.text()
    dialog.stem_edit.setText("   ")
    assert dialog.export() == [] and "failed" in dialog.status.text().lower()
    for check in dialog.checks.values():
        check.setChecked(False)
    assert dialog.export() == [] and "at least one" in dialog.status.text()
    dialog.copy_text()
    assert QApplication.clipboard().text().startswith("[0:00.5] Speaker 1: good morning")
    dialog.close()


def test_history_dialog_lists_and_acts(app: QApplication, recording: Path, tmp_path: Path) -> None:
    history_file = tmp_path / "history.json"
    paths = output_paths(recording)
    entry = HistoryEntry(
        path=str(recording), name="call.wav", sha256="ab" * 32, duration_s=30.0, profile="standard", marks=2,
        speakers=2, produced_utc="2026-01-01T10:20:30+00:00",
        outputs={"transcript": str(paths.transcript), "text": str(paths.text), "docx": str(paths.docx),
                 "subtitles": str(paths.subtitles), "review": str(paths.review), "run": str(paths.run)},
    )
    gone = HistoryEntry(path=str(tmp_path / "gone.wav"), name="gone.wav", sha256="", duration_s=10.0, profile="quick",
                        marks=0, speakers=1, produced_utc="2025-12-31T00:00:00+00:00",
                        outputs={"transcript": str(tmp_path / "gone.transcript.json")})
    append_entry(gone, history_file)
    append_entry(entry, history_file)
    dialog = HistoryDialog(theme_for(False), author="A Person", history_path=history_file)
    assert dialog.table.columnCount() == len(COLUMNS) and dialog.table.rowCount() == 2
    assert dialog.table.item(0, 1).text() == "call.wav" and dialog.table.item(0, 0).text() == "2026-01-01 10:20"
    assert dialog.table.item(0, 3).text() == "Standard" and dialog.table.item(0, 2).text() == "0:30"
    assert "(outputs missing)" in dialog.table.item(1, 6).text()
    assert "2 recordings transcribed" in dialog.summary.text()
    assert not dialog.export_button.isEnabled()

    received: list[list[Path]] = []
    dialog.add_requested.connect(received.append)
    dialog.table.selectRow(0)
    assert dialog.export_button.isEnabled() and dialog.add_button.isEnabled() and dialog.folder_button.isEnabled()
    dialog.add_to_library()
    assert received == [[recording]]
    export = dialog.export_selected(modal=False)
    assert export is not None and export.stem_edit.text() == "call"
    export.close()

    dialog.table.selectRow(1)
    assert not dialog.export_button.isEnabled() and not dialog.add_button.isEnabled()
    dialog.remove_selected()
    assert dialog.table.rowCount() == 1 and [e.name for e in read_history(history_file)] == ["call.wav"]
    dialog.close()
