"""Tests for the job card: the phrases for time left, the queued and running states, progress
reports moving the stage strip and the bar, and the provisional lines."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from twinscribe.app.job_status import STAGE_LABELS, JobStatusCard, describe_remaining, format_span  # noqa: E402
from twinscribe.pipeline import STAGES, Progress  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    existing = QApplication.instance()
    if existing is not None:
        return existing
    return QApplication([])


def test_phrases_for_time_left_and_spans() -> None:
    assert describe_remaining(None) == "estimating the time left"
    assert describe_remaining(5.0) == "almost done"
    assert describe_remaining(44.0) == "about 40 s left"
    assert describe_remaining(200.0) == "roughly 3 min left"
    assert format_span(65.0) == "1:05" and format_span(3725.0) == "1:02:05" and format_span(-2.0) == "0:00"


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
    assert card.partial_count() == 1 and "[0:01.0]  Good morning" in card.provisional.toPlainText()
    card.set_partials([(0.5, 1.0, "a"), (2.0, 3.0, "b")])
    assert card.partial_count() == 2 and card.provisional.toPlainText().count("\n") == 1
    card.stop()
    assert not card.running()
