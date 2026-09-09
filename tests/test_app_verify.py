"""Tests for the verification screen, run under the offscreen platform.

A synthetic review set is written to `tmp_path` with no audio file beside it, so playback
is disabled and every other behaviour of the screen is exercised: construction, the mark
list, the current-mark panels, the keys, the session file and the test-only panel.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from twinscribe.app import verify  # noqa: E402
from twinscribe.app.theme import apply_theme, palette_for, theme_for  # noqa: E402
from twinscribe.app.timeline import Timeline, format_mss  # noqa: E402

TICK = "\u2713"


@pytest.fixture(scope="module")
def app() -> QApplication:
    existing = QApplication.instance()
    if existing is not None:
        return existing
    return QApplication([])


# ----- synthetic review set ----------------------------------------------------------------

# Published words. Gaps: 3.5 to 6.0, 7.6 to 12.0, and 20.0 to 24.0 (then words to 26.0).
WORDS = [
    (0.5, 0.9, "good"), (0.9, 1.3, "morning"), (1.3, 1.8, "this"), (1.8, 2.2, "is"),
    (2.2, 2.6, "the"), (2.6, 3.0, "first"), (3.0, 3.5, "call"),
    (6.0, 6.4, "you"), (6.4, 6.8, "can"), (6.8, 7.2, "start"), (7.2, 7.6, "now"),
    (12.0, 12.5, "thank"), (12.5, 13.0, "you"), (13.0, 13.4, "for"), (13.4, 13.9, "waiting"),
    (13.9, 14.3, "on"), (14.3, 14.8, "the"), (14.8, 15.2, "line"), (15.2, 15.6, "and"),
    (15.6, 16.0, "for"), (16.0, 16.4, "your"), (16.4, 16.9, "patience"), (16.9, 17.3, "today"),
    (17.3, 17.8, "while"), (17.8, 18.2, "you"), (18.2, 18.6, "look"), (18.6, 19.0, "into"),
    (19.0, 19.5, "it"), (19.5, 20.0, "please"),
    (24.0, 24.5, "hold"), (24.5, 25.0, "on"), (25.0, 26.0, "goodbye"),
]

MARKS = [
    {"start": 3.1, "end": 6.4, "span_start": 3.5, "span_end": 6.0,
     "detector_words": 4, "detector_text": "yes I am here"},
    {"start": 7.2, "end": 12.4, "span_start": 7.6, "span_end": 12.0,
     "detector_words": 5, "detector_text": "hold on while I check"},
    {"start": 19.6, "end": 24.4, "span_start": 20.0, "span_end": 24.0,
     "detector_words": 2, "detector_text": "all right"},
]

REFERENCE = [
    {"reference_words": 3, "reference_speakers": ["A"]},
    {"reference_words": 4, "reference_speakers": ["A", "B"]},
    {"reference_words": 0, "reference_speakers": []},
]

EVALUATION = {
    "marks": 3, "marks_on_speech": 2, "precision": 2 / 3,
    "dropped_words": 10, "dropped_covered": 8, "recall": 0.8,
    "audio_to_review_s": 13.3, "audio_to_review_fraction": 13.3 / 30.0,
}


def review_document(audio: str, with_reference: bool = False) -> dict:
    marks = []
    for index, mark in enumerate(MARKS):
        record = dict(mark)
        if with_reference:
            record.update(REFERENCE[index])
        else:
            record.update({"reference_words": None, "reference_speakers": None})
        marks.append(record)
    return {
        "schema": "twinscribe.review.v1",
        "audio": audio,
        "duration_s": 30.0,
        "publisher": {"engine": "transducer", "model": "tdt-0.6b", "preset": "default"},
        "detector": {"engine": "whisper", "model": "small", "preset": "default"},
        "transcript": [{"s": s, "e": e, "w": w} for s, e, w in WORDS],
        "marks": marks,
        "evaluation": EVALUATION if with_reference else None,
    }


def write_review(tmp_path: Path, with_reference: bool = False, marks: list | None = None) -> Path:
    doc = review_document(str(tmp_path / "call.wav"), with_reference)
    if marks is not None:
        doc["marks"] = marks
        doc["evaluation"] = None
    path = tmp_path / "call.review.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def make_window(app: QApplication, review_path: Path) -> verify.VerifyWindow:
    review = verify.load_review_set(review_path)
    window = verify.VerifyWindow(review, review_path, theme_for(False))
    window.show()
    app.processEvents()
    return window


def dispose(app: QApplication, window: verify.VerifyWindow) -> None:
    window.close()
    window.deleteLater()
    app.processEvents()


@pytest.fixture
def window(app: QApplication, tmp_path: Path):
    win = make_window(app, write_review(tmp_path))
    yield win
    dispose(app, win)


@pytest.fixture
def test_mode_window(app: QApplication, tmp_path: Path):
    win = make_window(app, write_review(tmp_path, with_reference=True))
    yield win
    dispose(app, win)


# ----- pure helpers --------------------------------------------------------------------------


def test_format_mss_t_hand_values() -> None:
    assert verify.format_mss_t(0.0) == "0:00.0"
    assert verify.format_mss_t(3.1) == "0:03.1"
    assert verify.format_mss_t(65.34) == "1:05.3"
    assert verify.format_mss_t(125.26) == "2:05.3"
    assert verify.format_mss_t(59.96) == "1:00.0"
    assert verify.format_mss_t(-2.0) == "0:00.0"


def test_format_mss_hand_values() -> None:
    assert format_mss(0.0) == "0:00"
    assert format_mss(65.9) == "1:05"
    assert format_mss(1231.0) == "20:31"


def test_context_around_gap_short_sides() -> None:
    words = [verify.TranscriptWord(s, e, w) for s, e, w in WORDS]
    text = verify.context_around_gap(words, 7.6, 12.0)
    # Eleven words before the gap (all of them), fourteen after it, then an ellipsis.
    assert text == ("good morning this is the first call you can start now "
                    "[ GAP ] thank you for waiting on the line and for your patience today while you ...")


def test_context_around_gap_truncates_to_fourteen_each_side() -> None:
    before = [verify.TranscriptWord(i * 1.0, i * 1.0 + 0.5, f"b{i}") for i in range(20)]
    after = [verify.TranscriptWord(30.0 + i, 30.5 + i, f"a{i}") for i in range(20)]
    text = verify.context_around_gap(before + after, 20.0, 30.0)
    parts = text.split(" ")
    gap = parts.index("[")
    assert parts[gap:gap + 3] == ["[", "GAP", "]"]
    assert parts[0] == "..." and parts[-1] == "..."
    assert parts[1:gap] == [f"b{i}" for i in range(6, 20)]
    assert parts[gap + 3:-1] == [f"a{i}" for i in range(14)]


def test_context_around_gap_leading_and_trailing() -> None:
    words = [verify.TranscriptWord(s, e, w) for s, e, w in WORDS[:3]]
    assert verify.context_around_gap(words, 0.0, 0.5) == "[ GAP ] good morning this"
    assert verify.context_around_gap(words, 1.8, 5.0) == "good morning this [ GAP ]"


def test_coverage_fraction_hand_values() -> None:
    marks = [verify._mark_from_dict(m) for m in MARKS]
    assert verify.coverage_fraction(marks, 30.0) == pytest.approx(13.3 / 30.0)
    overlapping = [verify._mark_from_dict({"start": 0.0, "end": 10.0}),
                   verify._mark_from_dict({"start": 5.0, "end": 15.0})]
    assert verify.coverage_fraction(overlapping, 30.0) == pytest.approx(0.5)
    assert verify.coverage_fraction([], 30.0) == 0.0
    assert verify.coverage_fraction(marks, 0.0) == 0.0


def test_coverage_fraction_bounded() -> None:
    wide = [verify._mark_from_dict({"start": -5.0, "end": 500.0})]
    assert 0.0 <= verify.coverage_fraction(wide, 30.0) <= 1.0


def test_summary_and_header_text(tmp_path: Path) -> None:
    plain = verify.review_set_from_dict(review_document("call.wav"))
    assert verify.summary_text(plain) == "3 marks covering 44.3% of the recording"
    tested = verify.review_set_from_dict(review_document("call.wav", with_reference=True))
    assert verify.summary_text(tested) == (
        "3 marks covering 44.3% of the recording; 2 of 3 on real speech; "
        "marks cover 80% of dropped words (8 of 10)")
    header = verify.header_text(plain, tmp_path / "call.wav")
    assert header.startswith("call.wav")
    assert "0.5 min" in header
    assert "published by transducer tdt-0.6b" in header
    assert "checked against whisper small, which is never published" in header


def test_mark_row_text() -> None:
    mark = verify._mark_from_dict(MARKS[0])
    row = verify.mark_row_text(0, mark, False)
    assert row.startswith("    1   ")
    assert "0:03.1" in row and "2.5 s" in row and "4 words" in row
    assert verify.mark_row_text(0, mark, True).startswith(TICK)


def test_describe_reference() -> None:
    spoken = verify._mark_from_dict({**MARKS[1], **REFERENCE[1]})
    text, flag = verify.describe_reference(spoken)
    assert flag and text == "4 reference words fall in this span; speakers: A, B."
    false_alarm = verify._mark_from_dict({**MARKS[2], **REFERENCE[2]})
    text, flag = verify.describe_reference(false_alarm)
    assert not flag and text.startswith("False alarm")
    none = verify._mark_from_dict(MARKS[0])
    assert verify.describe_reference(none)[1] is False


def test_review_set_schema_check() -> None:
    doc = review_document("call.wav")
    doc["schema"] = "something.else"
    with pytest.raises(ValueError):
        verify.review_set_from_dict(doc)


def test_session_path_beside_review_set(tmp_path: Path) -> None:
    review = tmp_path / "sub" / "recording.review.json"
    assert verify.session_path_for(review) == tmp_path / "sub" / "recording.review.session.json"


# ----- timeline ------------------------------------------------------------------------------


def test_timeline_mapping_round_trip(app: QApplication) -> None:
    bar = Timeline()
    bar.resize(500, 80)
    bar.set_duration(120.0)
    for seconds in (0.0, 1.5, 37.25, 60.0, 119.9, 120.0):
        assert bar.time_at_x(bar.x_at_time(seconds)) == pytest.approx(seconds, abs=1e-6)
    assert bar.x_at_time(0.0) < bar.x_at_time(60.0) < bar.x_at_time(120.0)


def test_timeline_bounds(app: QApplication) -> None:
    bar = Timeline()
    bar.resize(500, 80)
    bar.set_duration(120.0)
    assert bar.time_at_x(-1000.0) == 0.0
    assert bar.time_at_x(1e6) == 120.0
    assert bar.x_at_time(-5.0) == bar.x_at_time(0.0)
    assert bar.x_at_time(999.0) == bar.x_at_time(120.0)
    bar.set_duration(0.0)
    assert bar.time_at_x(250.0) == 0.0
    assert bar.minimumHeight() >= 70


def test_timeline_click_seeks(app: QApplication) -> None:
    bar = Timeline()
    bar.resize(500, 80)
    bar.set_duration(100.0)
    bar.show()
    app.processEvents()
    received: list[float] = []
    bar.seek_requested.connect(received.append)
    rect = bar.bar_rect()
    x = int(rect.left() + rect.width() / 2)
    y = int(rect.top() + rect.height() / 2)
    QTest.mouseClick(bar, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x, y))
    assert len(received) == 1
    assert received[0] == pytest.approx(50.0, abs=0.5)
    bar.set_marks([(1.0, 2.0)])
    bar.set_resolved(0, True)
    bar.set_current(0)
    assert bar.current() == 0
    bar.set_current(7)
    assert bar.current() is None
    bar.set_playhead(500.0)
    assert bar.playhead() == 100.0
    bar.close()


# ----- theme ---------------------------------------------------------------------------------


def test_theme_variants(app: QApplication) -> None:
    light = theme_for(False)
    dark = theme_for(True)
    assert not light.dark and dark.dark
    assert light.accent != light.success != light.warning
    from PySide6.QtGui import QPalette
    role = QPalette.ColorRole.Window
    assert palette_for(False).color(role).lightness() > palette_for(True).color(role).lightness()
    theme = apply_theme(app, dark=False)
    assert app.style().objectName().lower() == "fusion"
    assert theme == light


# ----- the window ----------------------------------------------------------------------------


def test_window_constructs(window: verify.VerifyWindow) -> None:
    assert window.isVisible()
    assert window.windowTitle().startswith("0 of 3 done")
    assert window.header_label.text().startswith("call.wav")
    assert "published by transducer tdt-0.6b" in window.header_label.text()
    assert "never published" in window.header_label.text()
    assert window.summary_label.text().startswith("3 marks covering 44.3%")
    assert "Space" in window.footer_label.text() and "Enter" in window.footer_label.text()
    assert window.timeline.duration() == 30.0


def test_missing_audio_disables_playback(window: verify.VerifyWindow) -> None:
    assert not window.audio_available
    assert "not found" in window.status_label.text()
    QTest.keyClick(window.mark_list, Qt.Key.Key_Space)
    assert not window.is_playing()
    QTest.keyClick(window.mark_list, Qt.Key.Key_Return)
    assert "Nothing to play" in window.status_label.text()
    assert window.position_s == pytest.approx(3.1)


def test_list_has_one_row_per_mark(window: verify.VerifyWindow) -> None:
    assert window.mark_list.count() == 3
    for index in range(3):
        text = window.mark_list.item(index).text()
        assert not text.startswith(TICK)
        assert f"{index + 1:>3}" in text
        assert verify.format_mss_t(MARKS[index]["start"]) in text
        assert f"{MARKS[index]['detector_words']} words" in text


def test_selecting_a_row_updates_panels(window: verify.VerifyWindow) -> None:
    window.mark_list.setCurrentRow(1)
    assert window.current_index == 1
    assert window.span_label.text() == "0:07.2 to 0:12.4 (5.2 seconds)"
    context = window.context_panel.toPlainText()
    assert context.startswith("good morning this is the first call you can start now [ GAP ] thank you")
    assert "[ GAP ]" in context
    assert window.hint_panel.toPlainText() == "hold on while I check"
    assert window.position_s == pytest.approx(7.2)
    assert window.timeline.playhead() == pytest.approx(7.2)
    assert window.timeline.current() == 1


def test_n_resolves_and_advances(window: verify.VerifyWindow) -> None:
    assert window.current_index == 0
    QTest.keyClick(window.mark_list, Qt.Key.Key_N)
    assert window.resolutions[0].status == "nothing"
    assert window.resolutions[0].note == ""
    assert window.current_index == 1
    assert window.mark_list.item(0).text().startswith(TICK)
    assert not window.mark_list.item(1).text().startswith(TICK)
    assert window.windowTitle().startswith("1 of 3 done")
    assert "nothing was said" in window.status_label.text()


def test_t_with_value_resolves_with_text(window: verify.VerifyWindow) -> None:
    seen: list[str] = []

    def prompt(earlier: str) -> str | None:
        seen.append(earlier)
        return "yes I am here"

    window.text_prompt = prompt
    QTest.keyClick(window.mark_list, Qt.Key.Key_T)
    # The prompt starts from what the second engine heard when there is no earlier note.
    assert seen == [window.review.marks[0].detector_text] == ["yes I am here"]
    assert window.resolutions[0].status == "text"
    assert window.resolutions[0].note == "yes I am here"
    assert window.current_index == 1
    assert window.mark_list.item(0).text().startswith(TICK)

    # T on the same mark again offers the earlier note; cancelling changes nothing.
    window.select_mark(0)
    window.text_prompt = lambda earlier: (seen.append(earlier), None)[1]
    QTest.keyClick(window.mark_list, Qt.Key.Key_T)
    assert seen[-1] == "yes I am here"
    assert window.resolutions[0].note == "yes I am here"
    assert window.current_index == 0


def test_j_and_k_move(window: verify.VerifyWindow) -> None:
    QTest.keyClick(window.mark_list, Qt.Key.Key_J)
    assert window.current_index == 1
    QTest.keyClick(window.mark_list, Qt.Key.Key_J)
    assert window.current_index == 2
    QTest.keyClick(window.mark_list, Qt.Key.Key_J)
    assert window.current_index == 2
    QTest.keyClick(window.mark_list, Qt.Key.Key_K)
    assert window.current_index == 1
    assert window.position_s == pytest.approx(7.2)
    # Keys sent to the window itself, with focus elsewhere, work the same way.
    QTest.keyClick(window, Qt.Key.Key_K)
    assert window.current_index == 0


def test_arrows_nudge_five_seconds(window: verify.VerifyWindow) -> None:
    assert window.position_s == pytest.approx(3.1)
    QTest.keyClick(window.mark_list, Qt.Key.Key_Left)
    assert window.position_s == 0.0
    QTest.keyClick(window.mark_list, Qt.Key.Key_Right)
    assert window.position_s == pytest.approx(5.0)
    QTest.keyClick(window.mark_list, Qt.Key.Key_Right)
    assert window.position_s == pytest.approx(10.0)
    assert window.timeline.playhead() == pytest.approx(10.0)
    for _ in range(10):
        QTest.keyClick(window.mark_list, Qt.Key.Key_Right)
    assert window.position_s == 30.0


def test_last_mark_stays_current_after_resolution(window: verify.VerifyWindow) -> None:
    window.select_mark(2)
    QTest.keyClick(window.mark_list, Qt.Key.Key_N)
    assert window.current_index == 2
    assert window.done_count() == 1


def test_session_file_written_on_resolution_and_close(app: QApplication, tmp_path: Path) -> None:
    review_path = write_review(tmp_path)
    win = make_window(app, review_path)
    session = win.session_path
    assert session == tmp_path / "call.review.session.json"
    assert not session.exists()

    QTest.keyClick(win.mark_list, Qt.Key.Key_N)
    doc = json.loads(session.read_text(encoding="utf-8"))
    assert doc["schema"] == "twinscribe.review-session.v1"
    assert [m["status"] for m in doc["marks"]] == ["nothing", "open", "open"]
    assert doc["marks"][0] == {"start": 3.1, "end": 6.4, "status": "nothing", "note": ""}

    win.text_prompt = lambda earlier: "one more thing"
    QTest.keyClick(win.mark_list, Qt.Key.Key_T)
    win.close()
    app.processEvents()
    doc = json.loads(session.read_text(encoding="utf-8"))
    assert [m["status"] for m in doc["marks"]] == ["nothing", "text", "open"]
    assert doc["marks"][1]["note"] == "one more thing"
    assert doc["marks"][2] == {"start": 19.6, "end": 24.4, "status": "open", "note": ""}
    assert not list(tmp_path.glob("*.tmp"))
    win.deleteLater()
    app.processEvents()


def test_close_without_resolutions_leaves_no_session(app: QApplication, tmp_path: Path) -> None:
    review_path = write_review(tmp_path)
    win = make_window(app, review_path)
    QTest.keyClick(win.mark_list, Qt.Key.Key_J)
    QTest.keyClick(win.mark_list, Qt.Key.Key_Right)
    dispose(app, win)
    assert not win.session_path.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["call.review.json"]


def test_earlier_session_is_resumed(app: QApplication, tmp_path: Path) -> None:
    review_path = write_review(tmp_path)
    marks = [verify._mark_from_dict(m) for m in MARKS]
    earlier = verify.fresh_resolutions(marks)
    earlier[1].status = "text"
    earlier[1].note = "carried over"
    verify.write_session(verify.session_path_for(review_path), earlier)

    win = make_window(app, review_path)
    assert win.done_count() == 1
    assert win.resolutions[1].note == "carried over"
    assert win.mark_list.item(1).text().startswith(TICK)
    assert "Resumed 1 of 3" in win.status_label.text()
    assert win.windowTitle().startswith("1 of 3 done")
    dispose(app, win)


def test_mismatched_session_is_ignored(app: QApplication, tmp_path: Path) -> None:
    review_path = write_review(tmp_path)
    session = verify.session_path_for(review_path)
    session.write_text(json.dumps({
        "schema": "twinscribe.review-session.v1",
        "marks": [{"start": 99.0, "end": 100.0, "status": "nothing", "note": ""}],
    }), encoding="utf-8")
    win = make_window(app, review_path)
    assert win.done_count() == 0
    assert all(r.status == "open" for r in win.resolutions)
    dispose(app, win)


def test_test_only_panel_absent_without_reference(window: verify.VerifyWindow) -> None:
    assert window.reference_panel is None
    assert window.reference_label is None
    assert not window.review.has_reference


def test_test_only_panel_present_with_reference(test_mode_window: verify.VerifyWindow) -> None:
    win = test_mode_window
    assert win.review.has_reference
    assert win.reference_panel is not None
    assert win.reference_panel.isVisible()
    assert "Test only" in win.reference_panel.title()
    assert win.reference_label.text() == "3 reference words fall in this span; speakers: A."
    assert win.reference_label.font().bold()
    assert "2 of 3 on real speech" in win.summary_label.text()

    win.mark_list.setCurrentRow(2)
    assert win.reference_label.text().startswith("False alarm")
    assert not win.reference_label.font().bold()


def test_no_marks_constructs(app: QApplication, tmp_path: Path) -> None:
    win = make_window(app, write_review(tmp_path, marks=[]))
    assert win.mark_list.count() == 0
    assert win.current_index is None
    assert not win.nothing_button.isEnabled()
    assert win.windowTitle().startswith("0 of 0 done")
    for key in (Qt.Key.Key_N, Qt.Key.Key_T, Qt.Key.Key_J, Qt.Key.Key_K, Qt.Key.Key_Return):
        QTest.keyClick(win, key)
    assert win.current_index is None
    dispose(app, win)


def test_main_rejects_missing_review_set(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    status = verify.main([str(tmp_path / "absent.json")])
    assert status == 2
    assert "cannot read review set" in capsys.readouterr().err


def test_parser_options(tmp_path: Path) -> None:
    args = verify.build_parser().parse_args([str(tmp_path / "r.json"), "--dark", "--shot", "out.png"])
    assert args.dark is True
    assert args.shot == Path("out.png")
    assert args.review_set == tmp_path / "r.json"


def test_apply_puts_the_listener_words_into_the_transcript(app: QApplication, tmp_path: Path) -> None:
    from tests._fixtures import detector_transcript, make_document, published_transcript
    from twinscribe.outputs import render_all
    from twinscribe.outputs.transcript_doc import load_document, write_document
    from twinscribe.pipeline import output_paths
    from twinscribe.review import build_review, review_set, write_review_set

    doc = make_document("call.wav")
    paths = output_paths(tmp_path / "call.wav")
    write_document(doc, paths.transcript)
    render_all(doc, paths.text, paths.docx, paths.subtitles)
    published, detector = published_transcript(), detector_transcript()
    write_review_set(review_set(published, detector, "call.wav", build_review(published.words, detector.words, 30.0)), paths.review)
    review = verify.load_review_set(paths.review)
    assert len(review.marks) == len(doc["marks"]) == 2
    window = verify.VerifyWindow(review, paths.review, theme_for(False), author="A Person")
    assert window.transcript_path() == paths.transcript
    offered: list[str] = []

    def prompt(earlier: str) -> str:
        offered.append(earlier)
        return "yes I am here"

    window.text_prompt = prompt
    window.select_mark(0)
    window.resolve_text()
    # The prompt started from what the second engine heard in the span.
    assert offered == [review.marks[0].detector_text] and review.marks[0].detector_text
    assert window.handle_key(Qt.Key.Key_A) is True
    revised = load_document(paths.transcript)
    assert revised["review_applied"]["text"] == 1 and revised["review_applied"]["open"] == 1
    listener = [line for line in revised["lines"] if line.get("src") == "listener"]
    assert len(listener) == 1 and listener[0]["text"] == "yes I am here"
    assert "(heard on review): yes I am here" in paths.text.read_text(encoding="utf-8")
    assert "Transcript updated: 1 span with the listener's words, 0 silent, 1 still open" in window.status_label.text()
    # Applying again after a change replaces rather than accumulates.
    window.text_prompt = lambda earlier: "yes I am still here"
    window.select_mark(0)
    window.resolve_text()
    assert window.apply_to_transcript() is not None
    again = [line for line in load_document(paths.transcript)["lines"] if line.get("src") == "listener"]
    assert len(again) == 1 and again[0]["text"] == "yes I am still here"
    window.close()
    # Without a transcript document beside the review set nothing is applied.
    alone = tmp_path / "alone"
    alone.mkdir()
    lonely = alone / "call.review.json"
    write_review_set(review_set(published, detector, "call.wav", build_review(published.words, detector.words, 30.0)), lonely)
    window = verify.VerifyWindow(verify.load_review_set(lonely), lonely, theme_for(False))
    assert window.apply_to_transcript() is None and "No transcript document" in window.status_label.text()
    window.close()
