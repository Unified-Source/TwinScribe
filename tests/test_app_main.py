"""Tests for the application window, run under the offscreen platform.

A synthetic recording with a transcript document beside it, one without, a fake model store
and fake engines exercise the library, the transcript pane, the player controls that do not
need sound, the keys, speaker renaming, settings and a whole batch through the worker thread.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDropEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tests._fixtures import make_document, make_engines, make_models, make_plan_for  # noqa: E402
from twinscribe import audio  # noqa: E402
from twinscribe.app import main as app_main  # noqa: E402
from twinscribe.app.library import STATUS_DONE, STATUS_NEW, read_document_facts  # noqa: E402
from twinscribe.app.settings import AppSettings, load_settings, save_settings  # noqa: E402
from twinscribe.app.theme import apply_theme, stylesheet, theme_for  # noqa: E402
from twinscribe.app.timeline import Timeline  # noqa: E402
from twinscribe.outputs import render_all  # noqa: E402
from twinscribe.outputs.transcript_doc import write_document  # noqa: E402
from twinscribe.pipeline import output_paths  # noqa: E402
from twinscribe.review import review_set, write_review_set  # noqa: E402
from tests._fixtures import detector_transcript, published_transcript  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    existing = QApplication.instance()
    if existing is not None:
        return existing
    return QApplication([])


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "home"
    monkeypatch.setenv("TWINSCRIBE_HOME", str(folder))
    return folder


@pytest.fixture
def media(tmp_path: Path) -> dict[str, Path]:
    folder = tmp_path / "media"
    folder.mkdir()
    done = audio.synthetic_wav(folder / "call.wav", 30.0)
    doc = make_document("call.wav")
    paths = output_paths(done)
    write_document(doc, paths.transcript)
    render_all(doc, paths.text, paths.docx, paths.subtitles)
    from twinscribe.review import build_review

    published = published_transcript()
    detector = detector_transcript()
    marks = build_review(published.words, detector.words, 30.0)
    write_review_set(review_set(published, detector, "call.wav", marks), paths.review)
    fresh = audio.synthetic_wav(folder / "fresh.wav", 4.0)
    return {"folder": folder, "done": done, "fresh": fresh}


def make_window(app: QApplication, tmp_path: Path, engines=None, plan=None, **settings_overrides) -> app_main.MainWindow:
    models_root = tmp_path / "models"
    make_models(models_root)
    settings = AppSettings(models_dir=str(models_root), **settings_overrides)
    window = app_main.MainWindow(
        settings, theme_for(False), engines=engines, record_dir=tmp_path / "runs",
        plan=plan if plan is not None else make_plan_for(),
    )
    window.resize(1100, 700)
    window.show()
    app.processEvents()
    return window


def dispose(app: QApplication, window: app_main.MainWindow) -> None:
    window.close()
    window.deleteLater()
    app.processEvents()


def wait_until(app: QApplication, condition, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        app.processEvents()
        time.sleep(0.02)


# ----- settings, theme, helpers ---------------------------------------------------------------


def test_settings_round_trip_and_tolerance(tmp_path: Path) -> None:
    settings = AppSettings(models_dir="m", quality="quick", dark=True, threads=4, volume=0.5, library=["a", "b"], acceleration="cpu")
    path = save_settings(settings, tmp_path / "s.json")
    loaded = load_settings(path)
    assert loaded == settings
    path.write_text(json.dumps({"schema": "x", "threads": "many", "volume": 7, "output_mode": "odd", "library": "no", "acceleration": "npu"}), encoding="utf-8")
    tolerant = load_settings(path)
    assert tolerant.threads == 0 and tolerant.volume == 1.0 and tolerant.output_mode == "beside" and tolerant.library == []
    assert tolerant.acceleration == "auto"
    assert load_settings(tmp_path / "absent.json") == AppSettings()
    assert AppSettings(output_mode="folder", output_dir="x").output_dir_or_none == Path("x")
    assert AppSettings(output_mode="folder").output_dir_or_none is None
    assert AppSettings(threads=0).threads_or_none is None and AppSettings(threads=3).threads_or_none == 3


def test_theme_has_speaker_colours_and_stylesheet(app: QApplication) -> None:
    light, dark = theme_for(False), theme_for(True)
    assert len(light.speakers) == 8 and light.speaker(0) != light.speaker(1) and light.speaker(8) == light.speaker(0)
    assert light.muted != dark.muted
    sheet = stylesheet(True)
    assert "QPushButton#primary" in sheet and dark.accent.name() in sheet
    apply_theme(app, dark=False)
    assert app.style().objectName().lower() == "fusion"


def test_timeline_peaks(app: QApplication) -> None:
    bar = Timeline()
    bar.resize(400, 60)
    bar.set_duration(10.0)
    assert bar.peaks() is None
    bar.set_peaks([0, 50, 100], 100)
    assert bar.peaks() == [0, 50, 100]
    bar.set_show_labels(False)
    assert bar.bar_rect().top() < 10
    bar.set_peaks(None)
    assert bar.peaks() is None


def test_read_document_facts(media: dict[str, Path], tmp_path: Path) -> None:
    facts = read_document_facts(output_paths(media["done"]).transcript)
    assert facts == {"name": "call.wav", "duration_s": 30.0, "speakers": 2, "marks": 2}
    assert read_document_facts(tmp_path / "absent.json") is None
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"schema": "x"}), encoding="utf-8")
    assert read_document_facts(other) is None


# ----- the window --------------------------------------------------------------------------------


def test_window_starts_empty(app: QApplication, tmp_path: Path, home: Path) -> None:
    window = make_window(app, tmp_path)
    assert window.windowTitle() == "twinscribe"
    assert window.library_model.rowCount() == 0
    assert window.library_stack.currentIndex() == 1
    assert not window.transcribe_button.isEnabled()
    assert window.quality_box.count() == 3 and window.quality_box.currentData() == "standard"
    assert not window.transcript_view.document_loaded()
    assert not window.player_bar.play_button.isEnabled()
    dispose(app, window)


def test_adding_a_folder_finds_states(app: QApplication, tmp_path: Path, home: Path, media: dict[str, Path]) -> None:
    window = make_window(app, tmp_path)
    added = window.add_paths([media["folder"]])
    assert added == 2 and window.library_model.rowCount() == 2
    items = window.library_model.items()
    by_name = {item.name: item for item in items}
    assert by_name["call.wav"].status == STATUS_DONE and by_name["call.wav"].speakers == 2 and by_name["call.wav"].marks == 2
    assert by_name["call.wav"].duration_s == 30.0
    assert by_name["fresh.wav"].status == STATUS_NEW and by_name["fresh.wav"].summary() == "media"
    assert window.add_paths([media["folder"]]) == 0
    assert window.add_paths([tmp_path / "nothing.txt"]) == 0
    assert window.library_stack.currentIndex() == 0
    assert window.transcribe_button.isEnabled()
    dispose(app, window)


def test_selecting_a_transcribed_recording_shows_it(app: QApplication, tmp_path: Path, home: Path, media: dict[str, Path]) -> None:
    window = make_window(app, tmp_path)
    window.add_paths([media["done"]])
    app.processEvents()
    assert window.current_row() == 0
    assert window.windowTitle().startswith("call.wav")
    assert window.title_label.text() == "call.wav"
    assert "2 speakers" in window.meta_label.text() and "Standard" in window.meta_label.text() and "2 spans" in window.meta_label.text()
    view = window.transcript_view
    assert view.document_loaded() and view.line_count() == 4
    assert view.toPlainText().count("Possible missed speech") == 2
    assert "Speaker 2  you can start now" in view.toPlainText()
    assert window.review_button.isEnabled() and window.review_button.text() == "Review (2)"
    assert window.chips_layout.count() == 3          # two chips and the stretch
    assert window.player_bar.timeline.duration() == 30.0
    assert window.player_bar.timeline.peaks() is not None
    dispose(app, window)


def test_untranscribed_recording_still_loads_for_playback(app: QApplication, tmp_path: Path, home: Path, media: dict[str, Path]) -> None:
    window = make_window(app, tmp_path)
    window.add_paths([media["fresh"]])
    app.processEvents()
    assert not window.transcript_view.document_loaded()
    assert "Not transcribed yet" in window.transcript_view.placeholderText()
    assert "not transcribed yet" in window.meta_label.text()
    assert not window.review_button.isEnabled()
    assert window.player_bar.play_button.isEnabled()
    dispose(app, window)


def test_position_tracking_and_seeking(app: QApplication, tmp_path: Path, home: Path, media: dict[str, Path]) -> None:
    window = make_window(app, tmp_path)
    window.add_paths([media["done"]])
    app.processEvents()
    view = window.transcript_view
    assert view.line_at(0.1) is None
    assert view.line_at(0.5) == 0 and view.line_at(6.5) == 1 and view.line_at(12.5) == 2 and view.line_at(29.0) == 3
    view.set_position(12.5)
    assert view.current_line() == 2 and len(view.extraSelections()) == 1
    view.set_position(0.1)
    assert view.current_line() is None and view.extraSelections() == []
    window.seek(7.0)
    assert window.player_bar.timeline.playhead() == pytest.approx(7.0)
    assert view.current_line() == 1
    assert window.player_bar.time_label.text().startswith("0:07.0")
    window.seek(500.0)
    assert window.player_bar.timeline.playhead() == pytest.approx(30.0)
    dispose(app, window)


def test_keys_nudge_and_jump_between_marks(app: QApplication, tmp_path: Path, home: Path, media: dict[str, Path]) -> None:
    window = make_window(app, tmp_path)
    window.add_paths([media["done"]])
    app.processEvents()
    window.seek(0.0)
    QTest.keyClick(window.library_view, Qt.Key.Key_Right)
    assert window.player_bar.timeline.playhead() == pytest.approx(5.0)
    QTest.keyClick(window.transcript_view, Qt.Key.Key_Left)
    assert window.player_bar.timeline.playhead() == pytest.approx(0.0)
    QTest.keyClick(window, Qt.Key.Key_J)
    assert window.player_bar.timeline.playhead() == pytest.approx(3.1)      # first mark, padded start
    assert window.player_bar.timeline.current() == 0
    QTest.keyClick(window, Qt.Key.Key_J)
    assert window.player_bar.timeline.playhead() == pytest.approx(13.5)     # second mark
    QTest.keyClick(window, Qt.Key.Key_K)
    assert window.player_bar.timeline.playhead() == pytest.approx(3.1)
    assert window.transcript_view.follow()
    QTest.keyClick(window, Qt.Key.Key_F)
    assert not window.transcript_view.follow() and not window.player_bar.follow_button.isChecked()
    dispose(app, window)


def test_clicking_a_gap_marker_plays_its_span(app: QApplication, tmp_path: Path, home: Path, media: dict[str, Path]) -> None:
    window = make_window(app, tmp_path)
    window.add_paths([media["done"]])
    app.processEvents()
    received: list[int] = []
    window.transcript_view.mark_requested.connect(received.append)
    window.transcript_view.mark_requested.emit(1)
    assert received == [1]
    assert window.player_bar.timeline.playhead() == pytest.approx(13.5)
    window.play_mark(7)                                    # out of range: nothing happens
    assert window.player_bar.timeline.playhead() == pytest.approx(13.5)
    dispose(app, window)


def test_rename_speaker_rewrites_outputs(app: QApplication, tmp_path: Path, home: Path, media: dict[str, Path]) -> None:
    window = make_window(app, tmp_path)
    window.add_paths([media["done"]])
    app.processEvents()
    window.apply_speaker_name("speaker_01", "Caller")
    doc = window.current_document()
    assert doc is not None and [s["name"] for s in doc["speakers"]] == ["Speaker 1", "Caller"]
    paths = output_paths(media["done"])
    assert "[0:06.0] Caller: you can start now" in paths.text.read_text(encoding="utf-8")
    assert "Caller: you can start now" in paths.subtitles.read_text(encoding="utf-8")
    assert json.loads(paths.transcript.read_text(encoding="utf-8"))["speakers"][1]["name"] == "Caller"
    assert "Caller" in window.transcript_view.toPlainText()
    dispose(app, window)


def test_batch_runs_in_the_worker(app: QApplication, tmp_path: Path, home: Path, media: dict[str, Path]) -> None:
    window = make_window(app, tmp_path, engines=make_engines())
    window.add_paths([media["fresh"]])
    app.processEvents()
    assert window.library_model.item(0).status == STATUS_NEW
    assert window.start_transcription() is True
    assert window.worker is not None and window.stop_button.isVisible()
    wait_until(app, lambda: window.worker is None)
    item = window.library_model.item(0)
    assert item.status == STATUS_DONE and item.marks == 2 and item.speakers == 2
    paths = output_paths(media["fresh"])
    for path in paths.all:
        assert path.is_file(), path
    assert window.transcript_view.document_loaded() and window.transcript_view.line_count() == 4
    assert "Batch finished: 1 done" in window.statusBar().currentMessage()
    assert not window.transcribe_button.isEnabled()          # nothing pending any more
    assert list((tmp_path / "runs").glob("batch_*.json"))
    dispose(app, window)


def test_job_card_shows_progress_and_provisional_lines(app: QApplication, tmp_path: Path, home: Path, media: dict[str, Path]) -> None:
    from twinscribe.engines.base import Segment
    from twinscribe.pipeline import Progress

    window = make_window(app, tmp_path)
    window.add_paths([media["fresh"]])
    app.processEvents()
    assert not window.job_card_visible()
    window._batch_rows = [0]
    window._batch_total = 1
    window._batch_plan = make_plan_for()
    window.library_model.set_queued([0])
    window._load_item(window.library_model.item(0))
    assert window.job_card_visible() and not window.job_card.running()
    assert "Queued" in window.job_card.message_label.text()

    window._on_progress(0, Progress("publisher", 0.2, "Transcribing (published engine)", 5.0, 20.0))
    assert window.job_card.running() and window.job_card.stage_index() == 2
    assert window.windowTitle().startswith("20%")
    assert "20%" in window.batch_label.text()
    window._on_partial(0, Segment(start=1.0, end=2.0, text="hello there", words=()))
    window._on_partial(0, Segment(start=3.0, end=4.0, text="   ", words=()))
    assert window.job_card.partial_count() == 1

    # Selecting another recording and coming back keeps the provisional lines.
    window.add_paths([media["done"]])
    window.library_view.setCurrentIndex(window.library_model.index(1, 0))
    app.processEvents()
    assert not window.job_card_visible()
    window.library_view.setCurrentIndex(window.library_model.index(0, 0))
    app.processEvents()
    assert window.job_card_visible() and window.job_card.partial_count() == 1
    dispose(app, window)


def test_worker_batch_shows_the_card_then_the_transcript(app: QApplication, tmp_path: Path, home: Path, media: dict[str, Path]) -> None:
    window = make_window(app, tmp_path, engines=make_engines())
    window.add_paths([media["fresh"]])
    app.processEvents()
    assert window.start_transcription() is True
    assert window.job_card_visible()
    wait_until(app, lambda: window.worker is None)
    assert not window.job_card_visible() and window.transcript_view.document_loaded()
    assert window.windowTitle().startswith("fresh.wav")
    dispose(app, window)


def test_batch_failure_is_shown(app: QApplication, tmp_path: Path, home: Path, media: dict[str, Path]) -> None:
    window = make_window(app, tmp_path, engines=make_engines(fail_publisher=True))
    window.add_paths([media["fresh"]])
    app.processEvents()
    assert window.start_transcription() is True
    wait_until(app, lambda: window.worker is None)
    item = window.library_model.item(0)
    assert item.status == "failed" and "publisher exploded" in item.error
    assert "1 failed" in window.statusBar().currentMessage()
    assert window.transcribe_button.isEnabled()              # a failed file can be retried
    dispose(app, window)


def test_no_models_disables_transcription(app: QApplication, tmp_path: Path, home: Path, media: dict[str, Path]) -> None:
    settings = AppSettings(models_dir=str(tmp_path / "nowhere"))
    window = app_main.MainWindow(settings, theme_for(False), record_dir=tmp_path / "runs", plan=make_plan_for())
    window.show()
    app.processEvents()
    window.add_paths([media["fresh"]])
    assert window.quality_box.count() == 1 and not window.quality_box.isEnabled()
    assert window.selected_profile() is None
    assert not window.transcribe_button.isEnabled()
    dispose(app, window)


def test_levels_follow_the_usable_backends(app: QApplication, tmp_path: Path, home: Path) -> None:
    none = make_window(app, tmp_path, plan=make_plan_for(ct2=False, sherpa=False))
    assert none.quality_box.count() == 1 and "No detector library" in none.quality_box.toolTip()
    dispose(app, none)
    onnx = make_window(app, tmp_path, plan=make_plan_for(ct2=False))
    assert onnx.quality_box.count() == 3
    dispose(app, onnx)


def test_batch_reports_the_placement(app: QApplication, tmp_path: Path, home: Path, media: dict[str, Path]) -> None:
    window = make_window(app, tmp_path, engines=make_engines(), plan=make_plan_for(cuda=True))
    window.add_paths([media["fresh"]])
    app.processEvents()
    assert window.start_transcription() is True
    assert "detector CTranslate2 on cuda:0, float16" in window.statusBar().currentMessage()
    wait_until(app, lambda: window.worker is None)
    assert window.library_model.item(0).status == STATUS_DONE
    record = json.loads(output_paths(media["fresh"]).run.read_text(encoding="utf-8"))
    assert record["settings"]["plan"]["detector"]["device"] == "cuda" and record["settings"]["parallel_engines"] is True
    dispose(app, window)


def test_drop_adds_recordings(app: QApplication, tmp_path: Path, home: Path, media: dict[str, Path]) -> None:
    window = make_window(app, tmp_path)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(media["folder"]))])
    event = QDropEvent(QPointF(10.0, 10.0), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)
    window.dropEvent(event)
    assert window.library_model.rowCount() == 2
    dispose(app, window)


def test_close_saves_the_library(app: QApplication, tmp_path: Path, home: Path, media: dict[str, Path]) -> None:
    window = make_window(app, tmp_path)
    window.add_paths([media["done"]])
    dispose(app, window)
    saved = load_settings(home / "settings.json")
    assert saved.library == [str(media["done"])]
    assert len(saved.window_size) == 2 and len(saved.splitter) == 2

    again = app_main.MainWindow(saved, theme_for(False), record_dir=tmp_path / "runs")
    again.show()
    app.processEvents()
    assert again.library_model.rowCount() == 1 and again.current_row() == 0
    dispose(app, again)


def test_attach_log_only_without_console(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    assert app_main.attach_log(tmp_path) is None
    saved_hook = sys.excepthook
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    path = app_main.attach_log(tmp_path)
    handle = sys.stdout
    try:
        assert path == tmp_path / "twinscribe.log" and handle is not None and sys.stderr is handle
        print("hello from the window")
        sys.excepthook(ValueError, ValueError("boom"), None)
    finally:
        sys.excepthook = saved_hook
        monkeypatch.undo()
        handle.close()
    text = path.read_text(encoding="utf-8")
    assert "started without a console" in text and "hello from the window" in text and "ValueError: boom" in text


def test_parser_and_main_reject_nothing(tmp_path: Path) -> None:
    args = app_main.build_parser().parse_args([str(tmp_path), "--dark", "--shot", "x.png", "--seek", "12.5", "--select", "0"])
    assert args.dark and args.shot == Path("x.png") and args.seek == 12.5 and args.select == 0
