"""Tests for applying a listener's resolutions to a transcript document: listener lines with
their source, speaker inference, idempotence, the speaker recount, the session reader and the
header note."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._fixtures import make_document
from twinscribe.amend import (
    SOURCE_LISTENER,
    apply_resolutions,
    apply_session,
    infer_speaker,
    listener_words,
    read_resolutions,
    reviewed_note,
    strip_listener,
)
from twinscribe.outputs.transcript_doc import load_document, write_document


def _resolutions(doc: dict, **by_index: tuple[str, str]) -> list[dict]:
    out = [{"status": "open", "note": ""} for _ in doc["marks"]]
    for key, (status, note) in by_index.items():
        out[int(key[1:])] = {"status": status, "note": note}
    return out


def test_listener_words_are_spread_and_marked() -> None:
    words = listener_words("yes I am here", 3.5, 6.0)
    assert [w["w"] for w in words] == ["yes", "I", "am", "here"]
    assert words[0]["s"] == pytest.approx(3.5) and words[-1]["e"] == pytest.approx(6.0)
    assert words[1]["s"] == pytest.approx(3.5 + 2.5 / 4) and all(w["src"] == SOURCE_LISTENER for w in words)
    assert listener_words("   ", 0.0, 1.0) == []


def test_infer_speaker_needs_the_same_speaker_either_side() -> None:
    lines = [
        {"start": 0.0, "end": 2.0, "speaker": "a"},
        {"start": 4.0, "end": 6.0, "speaker": "a"},
        {"start": 7.0, "end": 9.0, "speaker": "b"},
    ]
    assert infer_speaker(lines, 2.0, 4.0) == "a"
    assert infer_speaker(lines, 6.0, 7.0) is None
    assert infer_speaker(lines, 9.0, 11.0) is None
    assert infer_speaker([], 0.0, 1.0) is None


def test_apply_adds_listener_lines_and_records_resolutions() -> None:
    doc = make_document("call.wav")
    marks = doc["marks"]
    assert len(marks) == 2
    revised = apply_resolutions(doc, _resolutions(doc, m0=("text", "yes I am here"), m1=("nothing", "")))
    listener = [line for line in revised["lines"] if line.get("src") == SOURCE_LISTENER]
    assert len(listener) == 1
    line = listener[0]
    assert line["start"] == pytest.approx(marks[0]["span_start"]) and line["end"] == pytest.approx(marks[0]["span_end"])
    assert line["text"] == "yes I am here" and len(line["words"]) == 4
    # The gap sits between two different speakers, so the line carries no label.
    assert line["speaker"] is None
    assert revised["marks"][0]["resolution"] == {"status": "text", "note": "yes I am here"}
    assert revised["marks"][1]["resolution"] == {"status": "nothing", "note": ""}
    assert revised["review_applied"]["text"] == 1 and revised["review_applied"]["nothing"] == 1
    assert revised["review_applied"]["listener_words"] == 4 and revised["review_applied"]["open"] == 0
    starts = [float(l["start"]) for l in revised["lines"]]
    assert starts == sorted(starts)
    # The engine's lines and words are untouched.
    engine_words = [w for l in revised["lines"] if l.get("src") != SOURCE_LISTENER for w in l["words"]]
    assert len(engine_words) == sum(len(l["words"]) for l in doc["lines"])
    # The unlabelled words appear in the speaker summary.
    unlabelled = [s for s in revised["speakers"] if s["label"] is None]
    assert unlabelled and unlabelled[0]["words"] == 4
    labelled_before = {s["label"]: s["words"] for s in doc["speakers"] if s["label"] is not None}
    labelled_after = {s["label"]: s["words"] for s in revised["speakers"] if s["label"] is not None}
    assert labelled_after == labelled_before


def test_apply_is_idempotent_and_editable() -> None:
    doc = make_document("call.wav")
    once = apply_resolutions(doc, _resolutions(doc, m0=("text", "yes I am here")))
    twice = apply_resolutions(once, _resolutions(doc, m0=("text", "yes I am still here")))
    listener = [l for l in twice["lines"] if l.get("src") == SOURCE_LISTENER]
    assert len(listener) == 1 and listener[0]["text"] == "yes I am still here"
    cleared = strip_listener(twice)
    assert all(l.get("src") != SOURCE_LISTENER for l in cleared["lines"])
    assert "review_applied" not in cleared and all("resolution" not in m for m in cleared["marks"])
    assert len(cleared["lines"]) == len(doc["lines"])
    with pytest.raises(ValueError):
        apply_resolutions(doc, [{"status": "open", "note": ""}])


def test_same_speaker_gap_takes_the_speaker() -> None:
    doc = make_document("call.wav")
    doc = json.loads(json.dumps(doc))
    for line in doc["lines"]:
        line["speaker"] = "speaker_00"
    revised = apply_resolutions(doc, _resolutions(doc, m0=("text", "indeed")))
    listener = [l for l in revised["lines"] if l.get("src") == SOURCE_LISTENER][0]
    assert listener["speaker"] == "speaker_00"
    assert [s for s in revised["speakers"] if s["label"] == "speaker_00"][0]["words"] == sum(len(l["words"]) for l in revised["lines"] if l["speaker"] == "speaker_00")


def test_reviewed_note() -> None:
    doc = make_document("call.wav")
    assert reviewed_note(doc) is None
    revised = apply_resolutions(doc, _resolutions(doc, m0=("text", "yes"), m1=("nothing", "")))
    assert reviewed_note(revised) == "Reviewed by a listener: 2 spans checked; 1 carries words typed after listening, shown as heard on review."
    partial = apply_resolutions(doc, _resolutions(doc, m1=("nothing", "")))
    assert reviewed_note(partial) == "Reviewed by a listener: 1 span checked; 1 still open."
    assert reviewed_note(apply_resolutions(doc, _resolutions(doc))) is None


def test_read_resolutions_and_apply_session(tmp_path: Path) -> None:
    doc = make_document("call.wav")
    transcript = tmp_path / "call.transcript.json"
    write_document(doc, transcript)
    session = tmp_path / "call.review.session.json"
    session.write_text(json.dumps({
        "schema": "twinscribe.review-session.v1",
        "marks": [
            {"start": 3.1, "end": 6.4, "status": "text", "note": "yes I am here"},
            {"start": 13.5, "end": 24.4, "status": "open", "note": ""},
        ],
    }), encoding="utf-8")
    assert read_resolutions(session, 2) == [{"status": "text", "note": "yes I am here"}, {"status": "open", "note": ""}]
    assert read_resolutions(session, 3) is None
    assert read_resolutions(tmp_path / "missing.json", 2) is None
    revised = apply_session(transcript, session, author="A Person")
    assert revised["review_applied"]["text"] == 1
    on_disk = load_document(transcript)
    assert any(l.get("src") == SOURCE_LISTENER for l in on_disk["lines"])
    assert (tmp_path / "call.txt").is_file() and (tmp_path / "call.docx").is_file() and (tmp_path / "call.srt").is_file()
    bad = tmp_path / "bad.session.json"
    bad.write_text(json.dumps({"schema": "twinscribe.review-session.v1", "marks": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        apply_session(transcript, bad, render=False)
