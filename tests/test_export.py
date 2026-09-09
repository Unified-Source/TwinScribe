"""Tests for exporting a recording's outputs: formats rendered again from the document, JSON
files copied or written, names, and the clipboard text."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tests._fixtures import make_document
from twinscribe.outputs.export import (
    DEFAULT_FORMATS,
    FORMAT_DOCX,
    FORMAT_REVIEW,
    FORMAT_RUN,
    FORMAT_SRT,
    FORMAT_TEXT,
    FORMAT_TRANSCRIPT,
    FORMAT_VTT,
    FORMATS,
    export_outputs,
    format_suffix,
    format_title,
    transcript_text,
)
from twinscribe.outputs.transcript_doc import write_document


def test_formats_table_is_consistent() -> None:
    keys = [key for key, _, _ in FORMATS]
    assert len(keys) == len(set(keys)) and set(DEFAULT_FORMATS) <= set(keys)
    for key, title, suffix in FORMATS:
        assert format_title(key) == title and format_suffix(key) == suffix and suffix.startswith(".")


def test_rendered_formats_are_written(tmp_path: Path) -> None:
    doc = make_document("call.wav")
    written = export_outputs(doc, tmp_path / "out", "call", (FORMAT_TEXT, FORMAT_DOCX, FORMAT_SRT, FORMAT_VTT), author="A Person")
    assert [p.name for p in written] == ["call.txt", "call.docx", "call.srt", "call.vtt"]
    text = (tmp_path / "out" / "call.txt").read_text(encoding="utf-8")
    assert "good morning this is the first call" in text
    assert (tmp_path / "out" / "call.vtt").read_text(encoding="utf-8").startswith("WEBVTT")
    assert "1\n" in (tmp_path / "out" / "call.srt").read_text(encoding="utf-8")
    with zipfile.ZipFile(tmp_path / "out" / "call.docx") as archive:
        assert "word/document.xml" in archive.namelist()
        assert "A Person" in archive.read("docProps/core.xml").decode("utf-8")


def test_json_formats_copied_or_written(tmp_path: Path) -> None:
    doc = make_document("call.wav")
    source = tmp_path / "src"
    source.mkdir()
    write_document(doc, source / "call.transcript.json")
    (source / "call.review.json").write_text('{"schema": "twinscribe.review.v1"}', encoding="utf-8")
    sources = {"transcript": source / "call.transcript.json", "review": source / "call.review.json", "run": source / "call.run.json"}
    written = export_outputs(doc, tmp_path / "out", "call", (FORMAT_TRANSCRIPT, FORMAT_REVIEW, FORMAT_RUN), sources=sources)
    # The run record is absent at its source and is skipped; the other two are copies.
    assert [p.name for p in written] == ["call.transcript.json", "call.review.json"]
    assert json.loads((tmp_path / "out" / "call.transcript.json").read_text(encoding="utf-8"))["schema"] == doc["schema"]
    # Without a source the transcript document is written from the document itself.
    written = export_outputs(doc, tmp_path / "out2", "copy", (FORMAT_TRANSCRIPT,))
    assert json.loads(written[0].read_text(encoding="utf-8"))["source"]["name"] == "call.wav"
    # Exporting into the folder the sources live in does not copy a file onto itself.
    written = export_outputs(doc, source, "call", (FORMAT_TRANSCRIPT, FORMAT_REVIEW), sources=sources)
    assert len(written) == 2 and (source / "call.review.json").read_text(encoding="utf-8").startswith('{"schema"')


def test_bad_arguments(tmp_path: Path) -> None:
    doc = make_document("call.wav")
    with pytest.raises(KeyError):
        export_outputs(doc, tmp_path, "call", ("pdf",))
    with pytest.raises(ValueError):
        export_outputs(doc, tmp_path, "   ", (FORMAT_TEXT,))


def test_transcript_text_is_the_lines_only() -> None:
    text = transcript_text(make_document("call.wav"))
    assert text.startswith("[0:00.5] Speaker 1: good morning") and "Review list" not in text
