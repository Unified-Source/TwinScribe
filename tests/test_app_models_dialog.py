"""Tests for the models dialog under the offscreen platform, with a stand-in for the fetch
that writes the store, and for the window's offer: the Get models button while no level is
complete, and the levels read again once a fetch has ended."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tests._fixtures import make_models, make_plan_for  # noqa: E402
from twinscribe.app import main as app_main  # noqa: E402
from twinscribe.app.models_dialog import COLUMNS, ModelsDialog  # noqa: E402
from twinscribe.app.settings import AppSettings  # noqa: E402
from twinscribe.app.theme import theme_for  # noqa: E402
from twinscribe.fetch import Cancelled, Progress  # noqa: E402
from twinscribe.models import KEY_PARAKEET_V2, KEY_WHISPER_LARGE, KEY_WHISPER_TURBO, find_models  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    existing = QApplication.instance()
    return existing if existing is not None else QApplication([])


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("TWINSCRIBE_MODELS", raising=False)
    monkeypatch.setenv("TWINSCRIBE_HOME", str(tmp_path / "home"))
    return tmp_path


def wait_until(condition: Callable[[], bool], timeout_ms: int = 10000) -> None:
    """Spin an event loop until the condition holds, so queued signals from the thread arrive."""
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: loop.quit() if condition() else None)
    timer.start()
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    timer.stop()
    assert condition(), "timed out waiting for the dialog"


def writing_fetch(specs, root, progress=None, cancel=None, log=None):
    """Stands in for the network: a one-byte file for every file of every model, with reports."""
    base = Path(root)
    total = sum(len(spec.sources) for spec in specs)
    done = 0
    for spec in specs:
        folder = base / spec.key
        folder.mkdir(parents=True, exist_ok=True)
        for source in spec.sources:
            if cancel is not None and cancel():
                raise Cancelled(source.target)
            if progress is not None:
                progress(Progress(spec.key, source.target, 50, 100, done, total))
            (folder / source.target).write_bytes(b"x")
            done += 1
            if progress is not None:
                progress(Progress(spec.key, source.target, 100, 100, done, total))
        for name in spec.required:
            (folder / name).write_bytes(b"x")
    return {}


def waiting_fetch(specs, root, progress=None, cancel=None, log=None):
    """Stands in for a slow network: waits to be cancelled."""
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if cancel is not None and cancel():
            raise Cancelled("waiting")
        time.sleep(0.01)
    raise RuntimeError("never cancelled")


def failing_fetch(specs, root, progress=None, cancel=None, log=None):
    raise RuntimeError("no route to the source")


def test_plan_follows_the_levels_and_the_folder(app: QApplication, isolated: Path) -> None:
    store = find_models(isolated / "store")
    dialog = ModelsDialog(store, fetch=writing_fetch)
    assert [dialog.table.horizontalHeaderItem(i).text() for i in range(dialog.table.columnCount())] == list(COLUMNS)
    assert dialog.root == isolated / "store" and dialog.folder_edit.text() == str(isolated / "store")
    assert dialog.chosen_levels() == ["standard"]
    keys = [spec.key for spec in dialog.specs]
    assert dialog.table.rowCount() == 6 and KEY_PARAKEET_V2 in keys and KEY_WHISPER_TURBO in keys
    assert "6 models to fetch, about 2.4 GB" in dialog.summary.text()
    assert dialog.table.item(0, 2).text().endswith("MB") or dialog.table.item(0, 2).text().endswith("GB")
    assert dialog.table.item(0, 4).text() == "github.com"
    assert dialog.download_button.isEnabled()
    dialog.level_checks["careful"].setChecked(True)
    assert KEY_WHISPER_LARGE in [spec.key for spec in dialog.specs] and dialog.table.rowCount() == 7
    dialog.level_checks["standard"].setChecked(False)
    assert KEY_WHISPER_TURBO not in [spec.key for spec in dialog.specs]
    complete = isolated / "complete"
    make_models(complete)
    dialog.folder_edit.setText(str(complete))
    dialog.refresh()
    assert dialog.table.rowCount() == 0 and "Nothing to fetch" in dialog.summary.text()
    assert not dialog.download_button.isEnabled() and dialog.start() is None
    dialog.close()


def test_fetch_runs_in_a_thread_and_the_store_is_read_again(app: QApplication, isolated: Path) -> None:
    root = isolated / "store"
    dialog = ModelsDialog(find_models(root), fetch=writing_fetch)
    received: list[object] = []
    dialog.fetched.connect(received.append)
    thread = dialog.start()
    assert thread is not None and dialog.running() and not dialog.download_button.isEnabled()
    assert dialog.cancel_button.isVisible() is False or dialog.isVisible() is False  # hidden dialog: visibility follows the parent
    wait_until(lambda: not dialog.running())
    assert received == [root]
    assert dialog.status.text().startswith("Done") and dialog.later_button.text() == "Close"
    assert dialog.progress.value() == 1000
    assert find_models(root).has(KEY_PARAKEET_V2) and find_models(root).has(KEY_WHISPER_TURBO)
    assert dialog.table.rowCount() == 0 and dialog.start() is None
    dialog.close()


def test_cancel_and_failure_are_reported_and_leave_the_dialog_usable(app: QApplication, isolated: Path) -> None:
    root = isolated / "store"
    dialog = ModelsDialog(find_models(root), fetch=waiting_fetch)
    assert dialog.start() is not None
    wait_until(lambda: dialog.running())
    dialog.cancel()
    assert "Stopping" in dialog.status.text()
    wait_until(lambda: not dialog.running())
    assert dialog.status.text().startswith("Cancelled") and dialog.download_button.isEnabled()
    assert dialog.table.rowCount() == 6
    dialog.close()

    dialog = ModelsDialog(find_models(root), fetch=failing_fetch)
    received: list[object] = []
    dialog.fetched.connect(received.append)
    assert dialog.start() is not None
    wait_until(lambda: not dialog.running())
    assert "stopped" in dialog.status.text() and "RuntimeError" in dialog.status.text()
    assert received == [root] and dialog.download_button.isEnabled()
    dialog.close()


def test_window_offers_the_fetch_while_no_level_is_complete(app: QApplication, isolated: Path) -> None:
    empty = isolated / "empty"
    settings = AppSettings(models_dir=str(empty))
    window = app_main.MainWindow(settings, theme_for(False), engines=None, record_dir=isolated / "runs", plan=make_plan_for())
    window.show()
    app.processEvents()
    assert window.needs_models() and window.models_button.isVisible()
    assert window.quality_box.currentText() == "No models" and not window.transcribe_button.isEnabled()
    assert "Get models" in window.quality_box.toolTip()

    dialog = window.open_models(modal=False)
    assert dialog.root == empty and dialog.table.rowCount() == 6
    other = isolated / "elsewhere"
    make_models(other)
    dialog.fetched.emit(other)
    app.processEvents()
    assert window.settings.models_dir == str(other) and window.models.has(KEY_PARAKEET_V2)
    assert not window.needs_models() and not window.models_button.isVisible()
    assert window.selected_profile() is not None and window.quality_box.isEnabled()
    assert window.statusBar().currentMessage().startswith("Models in place")
    dialog.close()

    # A fetch into the store already in use changes no setting.
    window.settings.models_dir = str(other)
    dialog = window.open_models(modal=False)
    dialog.fetched.emit(other)
    app.processEvents()
    assert window.settings.models_dir == str(other)
    dialog.close()
    window.close()
    window.deleteLater()
    app.processEvents()


def test_window_without_a_detector_library_does_not_offer(app: QApplication, isolated: Path) -> None:
    settings = AppSettings(models_dir=str(isolated / "empty"))
    window = app_main.MainWindow(settings, theme_for(False), engines=None, record_dir=isolated / "runs", plan=make_plan_for(ct2=False, sherpa=False))
    window.show()
    app.processEvents()
    assert not window.needs_models() and not window.models_button.isVisible()
    assert "detector library" in window.quality_box.toolTip()
    window.close()
    window.deleteLater()
    app.processEvents()
