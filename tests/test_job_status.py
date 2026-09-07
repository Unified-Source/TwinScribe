"""Tests for the job card: the phrases for time left, the queued and running states, progress
reports moving the stage strip and the bar, and the provisional lines: following the newest
line only while the reader is at the bottom, the way back to the latest line, clicks that seek
and play, and the highlight under the playhead."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from twinscribe.app.job_status import (  # noqa: E402
    STAGE_LABELS,
    JobStatusCard,
    describe_remaining,
    format_span,
    latest_button_text,
)
from twinscribe.pipeline import STAGES, Progress  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    existing = QApplication.instance()
    if existing is not None:
        return existing
    return QApplication([])


def shown_card(app: QApplication) -> JobStatusCard:
    card = JobStatusCard()
    card.resize(420, 320)
    card.show()
    app.processEvents()
    card.show_running("a.wav", 1, 1)
    return card


def fill(card: JobStatusCard, count: int) -> None:
    for index in range(count):
        card.add_partial(2.0 * index, 2.0 * index + 1.5, f"line number {index} with enough words to wrap in a narrow card")


def test_phrases_for_time_left_and_spans() -> None:
    assert describe_remaining(None) == "estimating the time left"
    assert describe_remaining(5.0) == "almost done"
    assert describe_remaining(44.0) == "about 40 s left"
    assert describe_remaining(200.0) == "roughly 3 min left"
    assert format_span(65.0) == "1:05" and format_span(3725.0) == "1:02:05" and format_span(-2.0) == "0:00"
    assert latest_button_text(0) == "Jump to latest"
    assert latest_button_text(1) == "Jump to latest (1 new line)"
    assert latest_button_text(3) == "Jump to latest (3 new lines)"


def test_stage_labels_cover_every_stage() -> None:
    assert set(STAGE_LABELS) == {name for name, _ in STAGES}


def test_card_queued_then_running(app: QApplication) -> None:
    card = JobStatusCard()
    card.show_queued("a.wav", 2, 3)
    assert not card.running() and card.stage_index() is None
    assert "Queued" in card.message_label.text() and card.position_label.text() == "2 of 3"
    assert card.provisional.isHidden()

    card.show_running("a.wav", 1, 3, ["Detector: x", "Publisher: y"])
    assert card.running() and card.stage_index() == 0
    assert "Detector: x" in card.plan_label.text() and not card.provisional.isHidden()
    card.update_progress(Progress(stage="detector", fraction=0.43, message="Checking (second engine)", elapsed_s=62.0, eta_s=80.0))
    assert card.stage_index() == 3 and card.percent_label.text() == "43%" and card.message() == "Checking (second engine)"
    assert card.progress_bar.value() == 430
    assert "Elapsed 1:02" in card.timing_label.text() and "about 80 s left" in card.timing_label.text()
    assert "Transcribing" in card.stage_label.text() and "Checking" in card.stage_label.text()

    card.add_partial(1.0, 3.0, "  Good   morning ")
    card.add_partial(4.0, 5.0, "   ")
    assert card.partial_count() == 1 and card.provisional.item(0).text() == "[0:01.0]  Good morning"
    assert card.line_start(0) == 1.0 and card.line_text(0) == "Good morning"
    card.set_partials([(0.5, 1.0, "a"), (2.0, 3.0, "b")])
    assert card.partial_count() == 2 and card.provisional.count() == 2 and card.following_latest()
    card.stop()
    assert not card.running()


def test_following_stops_when_the_reader_scrolls_up_and_resumes_on_request(app: QApplication) -> None:
    card = shown_card(app)
    fill(card, 40)
    app.processEvents()
    bar = card.provisional.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()
    assert card.following_latest() and card.latest_button.isHidden()

    bar.setValue(0)                                   # the reader scrolls up to read
    assert not card.following_latest() and not card.latest_button.isHidden()
    card.add_partial(100.0, 101.0, "arrived meanwhile")
    card.add_partial(102.0, 103.0, "and another")
    assert bar.value() == 0                           # the view stays where the reader left it
    assert card.unseen_count() == 2 and "2 new lines" in card.latest_button.text()

    card.latest_button.click()
    assert card.following_latest() and card.unseen_count() == 0 and card.latest_button.isHidden()
    assert bar.value() == bar.maximum()
    card.add_partial(104.0, 105.0, "followed")
    assert bar.value() == bar.maximum() and card.unseen_count() == 0

    bar.setValue(bar.maximum() - 5)                  # a small scroll up holds the view
    assert not card.following_latest()
    bar.setValue(bar.maximum())                      # scrolling back to the bottom follows again
    assert card.following_latest() and card.latest_button.isHidden()

    card.set_partials([(0.0, 1.0, "again")])         # selecting the recording again starts afresh
    assert card.following_latest() and card.unseen_count() == 0 and card.provisional.count() == 1
    card.close()


def test_clicking_a_line_seeks_and_double_clicking_plays(app: QApplication) -> None:
    card = shown_card(app)
    card.add_partial(1.0, 3.0, "first")
    card.add_partial(4.0, 6.0, "second")
    app.processEvents()
    seeks: list[float] = []
    plays: list[float] = []
    card.seek_requested.connect(seeks.append)
    card.play_requested.connect(plays.append)
    centre = card.provisional.visualItemRect(card.provisional.item(1)).center()
    QTest.mouseClick(card.provisional.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, centre)
    assert seeks == [4.0] and plays == []
    assert not card.following_latest()                # the reader is looking at a line
    QTest.mouseDClick(card.provisional.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, centre)
    assert plays == [4.0]
    card.close()


def test_playhead_highlights_the_line_under_it(app: QApplication) -> None:
    card = shown_card(app)
    card.add_partial(1.0, 3.0, "first")
    card.add_partial(4.0, 6.0, "second")
    assert card.line_at(0.5) is None and card.line_at(1.0) == 0 and card.line_at(4.5) == 1 and card.line_at(99.0) == 1
    card.set_position(4.5)
    assert card.current_line() == 1
    assert card.provisional.item(1).background().style() != Qt.BrushStyle.NoBrush
    assert card.provisional.item(0).background().style() == Qt.BrushStyle.NoBrush
    card.set_position(1.5)
    assert card.current_line() == 0
    assert card.provisional.item(1).background().style() == Qt.BrushStyle.NoBrush
    card.set_position(0.2)
    assert card.current_line() is None
    assert card.provisional.item(0).background().style() == Qt.BrushStyle.NoBrush
    card.set_follow(False)
    card.hold_position()
    card.set_position(4.5)                            # no scrolling when following is off; no error either
    assert card.current_line() == 1
    card.close()
