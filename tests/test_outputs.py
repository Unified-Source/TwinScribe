"""Tests for the transcript document and the renderers: plain text, subtitles and Word."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tests._fixtures import make_document
from twinscribe.outputs import render_all
from twinscribe.outputs.plain_text import DRAFT_NOTICE, header_lines, render_text, transcript_lines
from twinscribe.outputs.subtitles import (
    Cue,
    build_cues,
    render_srt,
    render_vtt,
    split_rows,
    srt_timestamp,
    vtt_timestamp,
)
from twinscribe.outputs.transcript_doc import (
    TRANSCRIPT_SCHEMA,
    clock,
    load_document,
    overview_peaks,
    set_speaker_name,
    speaker_count,
    speaker_names,
    write_document,
)
from twinscribe.outputs.word_docx import APPLICATION_NAME, document_xml, write_docx

# ----- document ----------------------------------------------------------------------------


def test_clock_hand_values() -> None:
    assert clock(0.0) == "0:00.0"
    assert clock(65.34) == "1:05.3"
    assert clock(59.96) == "1:00.0"
    assert clock(3725.0) == "1:02:05.0"
    assert clock(3725.0, tenths=False) == "1:02:05"
    assert clock(-3.0) == "0:00.0"


def test_overview_peaks_scaling_and_shape() -> None:
    assert overview_peaks(np.zeros(1000, dtype=np.float32), bins=10) == [0] * 10
    assert overview_peaks(np.zeros(0, dtype=np.float32), bins=4) == [0] * 4
    tone = np.concatenate([np.zeros(500, dtype=np.float32), 0.5 * np.ones(500, dtype=np.float32)])
    peaks = overview_peaks(tone, bins=4)
    assert peaks == [0, 0, 100, 100]
    with pytest.raises(ValueError):
        overview_peaks(tone, bins=0)


def test_document_shape() -> None:
    doc = make_document()
    assert doc["schema"] == TRANSCRIPT_SCHEMA
    assert doc["source"]["name"] == "call.wav" and doc["duration_s"] == 30.0
    assert doc["source"]["outputs"] == "call"
    assert [s["name"] for s in doc["speakers"]] == ["Speaker 1", "Speaker 2"]
    assert [s["words"] for s in doc["speakers"]] == [11, 7]
    assert speaker_count(doc) == 2
    assert len(doc["lines"]) == 4
    assert doc["lines"][1] == {
        "start": 6.0, "end": 7.6, "speaker": "speaker_01", "text": "you can start now",
        "words": [{"s": 6.0, "e": 6.4, "w": "you"}, {"s": 6.4, "e": 6.8, "w": "can"},
                  {"s": 6.8, "e": 7.2, "w": "start"}, {"s": 7.2, "e": 7.6, "w": "now"}],
    }
    assert doc["review"]["marks"] == 2 and len(doc["marks"]) == 2
    assert doc["marks"][0]["span_start"] == 3.5 and doc["marks"][0]["span_end"] == 6.0
    assert "detector_text" not in doc["marks"][0]
    # Padded windows 3.1 to 6.4 and 13.5 to 24.4: 3.3 + 10.9 seconds.
    assert doc["review"]["seconds"] == pytest.approx(14.2)
    assert doc["review"]["fraction"] == pytest.approx(14.2 / 30.0)
    assert doc["engines"]["publisher"]["engine"] == "parakeet_tdt"
    assert doc["engines"]["detector"]["preset"] == "production"
    assert doc["engines"]["diarization"]["labels_found"] == 2
    assert len(doc["overview"]["peaks"]) == 1200 and doc["overview"]["scale"] == 100


def test_document_without_speakers() -> None:
    doc = make_document(with_speakers=False)
    assert doc["engines"]["diarization"] is None
    assert [s["label"] for s in doc["speakers"]] == [None]
    assert speaker_count(doc) == 0
    assert doc["speakers"][0]["words"] == 18


def test_rename_and_round_trip(tmp_path: Path) -> None:
    doc = make_document()
    renamed = set_speaker_name(doc, "speaker_01", "  Caller  ")
    assert speaker_names(renamed) == {"speaker_00": "Speaker 1", "speaker_01": "Caller"}
    assert speaker_names(doc)["speaker_01"] == "Speaker 2"
    with pytest.raises(KeyError):
        set_speaker_name(doc, "speaker_09", "x")
    with pytest.raises(ValueError):
        set_speaker_name(doc, "speaker_00", "   ")
    path = tmp_path / "call.transcript.json"
    write_document(renamed, path)
    assert load_document(path) == renamed
    (tmp_path / "other.json").write_text(json.dumps({"schema": "x"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_document(tmp_path / "other.json")


# ----- plain text --------------------------------------------------------------------------


def test_plain_text_header_and_lines() -> None:
    doc = make_document()
    header = header_lines(doc)
    assert header[0] == "call.wav"
    assert header[1].startswith("Duration 0:30   |   2 speakers")
    assert "Published engine: parakeet_tdt parakeet-tdt-0.6b-v2-int8 (vad)" in header
    assert any(line.startswith("Checked against: whisper_ct2") and "never published" in line for line in header)
    assert any(line.strip().startswith("Speaker 1") and "11 words" in line for line in header)
    assert any("Review list: 2 spans" in line and "call.review.json" in line for line in header)
    assert header[-1] == DRAFT_NOTICE

    lines = transcript_lines(doc)
    assert lines[0] == "[0:00.5] Speaker 1: good morning this is the first call"
    assert lines[1] == ""                         # blank line at a speaker change
    assert lines[2] == "[0:06.0] Speaker 2: you can start now"
    text = render_text(doc)
    assert text.endswith("goodbye\n") and "\nTranscript\n" in text


def test_plain_text_names_a_fixed_speaker_count() -> None:
    doc = make_document()
    assert any("clustering threshold 0.5" in line for line in header_lines(doc))
    doc["engines"]["diarization"]["settings"]["num_speakers"] = 2
    header = header_lines(doc)
    assert any("speaker count fixed at 2" in line and "hidden inside another label" in line for line in header)
    assert not any("clustering threshold" in line for line in header)


def test_plain_text_without_marks_or_speakers() -> None:
    doc = make_document(with_speakers=False)
    doc["marks"] = []
    doc["review"] = {"marks": 0, "seconds": 0.0, "fraction": 0.0}
    text = render_text(doc)
    assert "no span where speech may be missing" in text
    assert "Unknown speaker" in text


# ----- subtitles ---------------------------------------------------------------------------


def test_timestamps() -> None:
    assert srt_timestamp(3661.5) == "01:01:01,500"
    assert vtt_timestamp(3661.5) == "01:01:01.500"
    assert srt_timestamp(0.0) == "00:00:00,000"
    assert srt_timestamp(-1.0) == "00:00:00,000"


def test_split_rows() -> None:
    assert split_rows("short") == "short"
    long = "Speaker 1: good morning this is the first call of the day"
    rows = split_rows(long, max_row_chars=30).split("\n")
    assert len(rows) == 2 and all(len(r) <= 34 for r in rows)
    assert split_rows("x" * 60, max_row_chars=30) == "x" * 60


def test_cues_prefix_every_cue_with_the_speaker() -> None:
    doc = make_document()
    cues = build_cues(doc)
    assert len(cues) == 4
    assert cues[0].start == 0.5 and cues[0].end == 3.5
    assert cues[0].text.startswith("Speaker 1: good morning")
    assert "\n" in cues[0].text                     # 46 characters wrap into two rows
    assert cues[1].text == "Speaker 2: you can start now"
    for cue in cues:
        assert cue.text.split(":")[0] in ("Speaker 1", "Speaker 2")
    starts = [c.start for c in cues]
    assert starts == sorted(starts)


def test_cues_split_long_lines_and_lengthen_short_ones() -> None:
    doc = make_document()
    many = [{"s": i * 0.3, "e": i * 0.3 + 0.25, "w": f"word{i}"} for i in range(40)]
    doc["lines"] = [
        {"start": 0.0, "end": 12.0, "speaker": "speaker_00", "text": "", "words": many},
        {"start": 20.0, "end": 20.2, "speaker": "speaker_01", "text": "hi", "words": [{"s": 20.0, "e": 20.2, "w": "hi"}]},
        {"start": 25.0, "end": 26.0, "speaker": "speaker_01", "text": "bye", "words": [{"s": 25.0, "e": 26.0, "w": "bye"}]},
    ]
    cues = build_cues(doc)
    long_cues = [c for c in cues if c.start < 12.0]
    assert len(long_cues) >= 3
    for cue in long_cues:
        assert len(cue.text.replace("\n", " ")) <= 84
        assert cue.end - cue.start <= 7.0 + 1e-9
    short = [c for c in cues if c.start == 20.0][0]
    assert short.end == pytest.approx(21.0)         # lengthened to the minimum
    last = cues[-1]
    assert last.start == 25.0 and last.end == 26.0
    with pytest.raises(ValueError):
        build_cues(doc, max_chars=4)


def test_render_srt_and_vtt() -> None:
    cues = [Cue(0.5, 3.5, "Speaker 1: hello"), Cue(6.0, 7.6, "Speaker 2: there")]
    srt = render_srt(cues)
    assert srt.startswith("1\n00:00:00,500 --> 00:00:03,500\nSpeaker 1: hello\n\n2\n")
    vtt = render_vtt(cues)
    assert vtt.startswith("WEBVTT\n\n00:00:00.500 --> 00:00:03.500\nSpeaker 1: hello\n")


# ----- Word --------------------------------------------------------------------------------


def test_docx_parts_are_well_formed_and_carry_the_text(tmp_path: Path) -> None:
    doc = make_document()
    doc["lines"][0]["text"] = "good <morning> & welcome"
    target = tmp_path / "call.docx"
    write_docx(doc, target, author="A. Person")
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        assert names[0] == "[Content_Types].xml"
        for name in names:
            ET.fromstring(archive.read(name))
        body = archive.read("word/document.xml").decode("utf-8")
        assert "good &lt;morning&gt; &amp; welcome" in body
        assert "Speaker 1" in body and "Speaker 2" in body and "call.wav" in body
        assert "Review list: 2 spans" in body and DRAFT_NOTICE in body
        core = archive.read("docProps/core.xml").decode("utf-8")
        assert "<dc:creator>A. Person</dc:creator>" in core
        app = archive.read("docProps/app.xml").decode("utf-8")
        assert f"<Application>{APPLICATION_NAME}" in app
    assert not list(tmp_path.glob("*.tmp"))


def test_docx_default_author_is_the_application(tmp_path: Path) -> None:
    write_docx(make_document(), tmp_path / "x.docx")
    with zipfile.ZipFile(tmp_path / "x.docx") as archive:
        assert f"<dc:creator>{APPLICATION_NAME}</dc:creator>" in archive.read("docProps/core.xml").decode("utf-8")


def test_docx_reads_back_with_a_word_library(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    target = tmp_path / "call.docx"
    write_docx(make_document(), target, author="Reader")
    opened = docx.Document(str(target))
    texts = [p.text for p in opened.paragraphs]
    assert texts[0] == "call.wav"
    assert any("good morning this is the first call" in t for t in texts)
    assert len(opened.tables) == 1 and len(opened.tables[0].rows) == 3
    assert opened.core_properties.author == "Reader"


def test_document_xml_without_lines() -> None:
    doc = make_document()
    doc["lines"] = []
    assert "No words were published" in document_xml(doc)


def test_scene_phrases() -> None:
    from twinscribe.outputs.transcript_doc import format_seconds, non_speech_summary, scene_phrase, scene_tag

    assert format_seconds(6.4) == "6 s" and format_seconds(60.0) == "1 min" and format_seconds(150.0) == "2 min 30 s"
    assert scene_phrase({"start": 10.0, "end": 16.0, "kind": "silence"}) == "silence, 6 s"
    assert scene_phrase({"start": 10.0, "end": 22.0, "kind": "music", "label": ""}) == "music, no speech, 12 s"
    assert scene_phrase({"start": 0.0, "end": 8.0, "kind": "noise", "label": "Waterfall"}) == "background noise, no speech, 8 s"
    assert scene_phrase({"start": 0.0, "end": 5.0, "kind": "sound", "label": "Siren"}) == "sound (Siren), no speech, 5 s"
    assert scene_tag({"kind": "music"}) == "[music]" and scene_tag({"kind": "noise"}) == "[background noise]"
    assert scene_tag({"kind": "sound", "label": "Siren"}) == "[sound: Siren]" and scene_tag({"kind": "sound"}) == "[sound]"
    assert non_speech_summary({"non_speech": {"seconds_by_kind": {"silence": 14.0, "music": 21.0}}}) == "silence 14 s, music 21 s"
    assert non_speech_summary({}) == ""


def test_scenes_render_in_text_word_and_subtitles(tmp_path: Path) -> None:
    from twinscribe.scenes import Scene

    doc = make_document(scenes=(Scene(7.6, 12.0, "silence"), Scene(13.9, 24.0, "music", "", 0.8)))
    assert [s["kind"] for s in doc["scenes"]] == ["silence", "music"]
    assert doc["non_speech"]["suppressed_detector_words"] == 2 and doc["review"]["marks"] == 1
    text = render_text(doc)
    assert "Without speech: silence 4 s, music 10 s; marked in the transcript." in text
    assert "2 words the second engine placed inside silence, music or noise were left out of the review list" in text
    lines = transcript_lines(doc)
    assert "[0:07.6] (silence, 4 s)" in lines and "[0:13.9] (music, no speech, 10 s)" in lines
    assert lines.index("[0:07.6] (silence, 4 s)") < lines.index("[0:12.0] Speaker 1: thank you for waiting")
    assert lines[lines.index("[0:07.6] (silence, 4 s)") - 1] == ""                # set off by a blank line
    cues = build_cues(doc)
    tags = [c for c in cues if c.text.startswith("[")]
    assert [(c.start, c.end, c.text) for c in tags] == [(13.9, 24.0, "[music]")]  # silence gets no cue
    assert [c.start for c in cues] == sorted(c.start for c in cues)
    body = document_xml(doc)
    assert "(music, no speech, 10 s)" in body and "<w:i/>" in body and "Without speech: silence 4 s" in body
    write_docx(doc, tmp_path / "s.docx")
    with zipfile.ZipFile(tmp_path / "s.docx") as archive:
        for name in archive.namelist():
            ET.fromstring(archive.read(name))


def test_render_all_writes_three_files(tmp_path: Path) -> None:
    doc = make_document()
    render_all(doc, tmp_path / "c.txt", tmp_path / "c.docx", tmp_path / "c.srt", author="x")
    assert (tmp_path / "c.txt").read_text(encoding="utf-8").startswith("call.wav\n")
    assert zipfile.is_zipfile(tmp_path / "c.docx")
    assert (tmp_path / "c.srt").read_text(encoding="utf-8").startswith("1\n00:00:00,500")
