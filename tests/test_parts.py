"""Tests for a recording in parts: the AVI header read for the start and the duration, the
chaining of timed files into recordings, discovery with and without joining, the WAV join
with gaps filled and overlaps cut, and the pipeline over two parts with a gap."""

from __future__ import annotations

import json
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from tests._fixtures import make_engines, make_models, make_plan_for
from twinscribe import audio, pipeline
from twinscribe.paths import playable_copy_path
from twinscribe.pipeline import Job, Parts, Recording, discover_recordings, group_parts
from twinscribe.profiles import profile_for

UTC = timezone.utc
START = datetime(2026, 8, 26, 15, 3, 1, 90000, tzinfo=UTC)


def chunk(tag: bytes, body: bytes) -> bytes:
    return tag + struct.pack("<I", len(body)) + body + (b"\0" if len(body) % 2 else b"")


def a_list(kind: bytes, body: bytes) -> bytes:
    return chunk(b"LIST", kind + body)


def avi_bytes(start: datetime | None = START, audio_length: int | None = 14065, scale: int = 1024, rate: int = 48000,
              frames: int = 8987, micros: int = 33333) -> bytes:
    """A minimal AVI header as a court recorder writes it: the main header, an audio stream
    header, the start as a FILETIME in a TUTC chunk, then the movie list."""
    avih = struct.pack("<14I", micros, 0, 0, 0, frames, *([0] * 9))
    streams = b""
    if audio_length is not None:
        strh = struct.pack("<4s4sIHHIIIIIIII8s", b"auds", b"\0" * 4, 0, 0, 0, 0, scale, rate, 0, audio_length, 0, 0, 0, b"\0" * 8)
        streams = a_list(b"strl", chunk(b"strh", strh) + chunk(b"strf", b"\0" * 16))
    hdrl = a_list(b"hdrl", chunk(b"avih", avih) + streams)
    times = b""
    if start is not None:
        filetime = int((start - datetime(1601, 1, 1, tzinfo=UTC)).total_seconds() * 10_000_000)
        times = chunk(b"TUTC", struct.pack("<Q", filetime)) + chunk(b"TLOC", struct.pack("<Q", filetime))
    movi = a_list(b"movi", chunk(b"00dc", b"\0" * 4) + chunk(b"01wb", b"\0" * 6))
    body = b"AVI " + hdrl + times + movi
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_avi_facts_read_the_start_and_the_duration(tmp_path: Path) -> None:
    path = tmp_path / "room_0903.trm"
    path.write_bytes(avi_bytes())
    facts = audio.avi_facts(path)
    assert facts.start_utc == START
    assert facts.duration_s == pytest.approx(14065 * 1024 / 48000)
    # Without an audio stream the frame count and the frame time give the duration.
    (tmp_path / "video_only.avi").write_bytes(avi_bytes(audio_length=None))
    assert audio.avi_facts(tmp_path / "video_only.avi").duration_s == pytest.approx(8987 * 33333 / 1e6)
    # Without a start chunk there is no start; a WAV or a text file gives nothing at all.
    (tmp_path / "untimed.avi").write_bytes(avi_bytes(start=None))
    untimed = audio.avi_facts(tmp_path / "untimed.avi")
    assert untimed.start_utc is None and untimed.duration_s == pytest.approx(14065 * 1024 / 48000)
    wav = audio.synthetic_wav(tmp_path / "tone.wav", 0.5)
    assert audio.avi_facts(wav) == audio.MediaFacts(None, None)
    (tmp_path / "notes.trm").write_bytes(b"not a container")
    assert audio.avi_facts(tmp_path / "notes.trm") == audio.MediaFacts(None, None)


def facts_by_name(table: dict[str, tuple[datetime | None, float | None]]):
    def fake(path: Path) -> audio.MediaFacts:
        start, duration = table.get(path.name, (None, None))
        return audio.MediaFacts(start, duration)
    return fake


def test_group_parts_chains_files_that_follow_one_another(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    names = ["a.trm", "b.trm", "c.trm", "d.trm", "e.trm", "f.avi"]
    for name in names:
        (tmp_path / name).write_bytes(b"")
    b_start = START + timedelta(seconds=110 + 145.2)          # a pause of 145 s after the first file
    c_start = b_start + timedelta(seconds=300.2)               # 0.2 s after b ends
    d_start = c_start + timedelta(seconds=300 + 20 * 60)       # twenty minutes after c ends: another recording
    monkeypatch.setattr(pipeline, "recording_facts", facts_by_name({
        "a.trm": (START, 110.0),
        "b.trm": (b_start, 300.0),
        "c.trm": (c_start, 300.0),
        "d.trm": (d_start, 300.0),
        "e.trm": (None, None),                                  # no header time: on its own
        "f.avi": (c_start + timedelta(seconds=300), 60.0),      # follows c in time but not in kind
    }))
    found = group_parts([tmp_path / name for name in names])
    assert [r.source.name for r in found] == ["a.trm", "d.trm", "e.trm", "f.avi"]
    joined = found[0]
    assert joined.parts is not None and [p.name for p in joined.parts.paths] == ["a.trm", "b.trm", "c.trm"]
    assert joined.parts.offsets_s == pytest.approx((0.0, 255.2, 555.4))
    assert all(r.parts is None for r in found[1:])


def test_group_parts_allows_a_small_overlap_and_splits_a_large_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("a.trm", "b.trm", "c.trm"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(pipeline, "recording_facts", facts_by_name({
        "a.trm": (START, 300.0),
        "b.trm": (START + timedelta(seconds=298.5), 300.0),    # begins 1.5 s before a ends: the rollover
        "c.trm": (START + timedelta(seconds=590.0), 300.0),    # begins 8.5 s before b ends: something else
    }))
    found = group_parts([tmp_path / n for n in ("a.trm", "b.trm", "c.trm")])
    assert [r.source.name for r in found] == ["a.trm", "c.trm"]
    assert found[0].parts is not None and found[0].parts.offsets_s == pytest.approx((0.0, 298.5))
    assert found[1].parts is None


def test_discover_recordings_joins_only_where_two_of_a_kind_sit_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("a.trm", "b.trm", "lone.avi", "call.wav"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.trm").write_bytes(b"")
    seen: list[str] = []

    def fake(path: Path) -> audio.MediaFacts:
        seen.append(path.name)
        table = {"a.trm": (START, 300.0), "b.trm": (START + timedelta(seconds=300), 300.0), "c.trm": (START, 300.0),
                 "lone.avi": (START, 300.0)}
        return audio.MediaFacts(*table.get(path.name, (None, None)))

    monkeypatch.setattr(pipeline, "recording_facts", fake)
    found = discover_recordings([tmp_path])
    assert [(r.source.name, len(r.parts) if r.parts else None) for r in found] == [
        ("a.trm", 2), ("call.wav", None), ("lone.avi", None), ("c.trm", None),
    ]
    assert sorted(seen) == ["a.trm", "b.trm"]  # only a folder with two files of one kind has its headers read
    apart = discover_recordings([tmp_path], join=False)
    assert [r.source.name for r in apart] == ["a.trm", "b.trm", "call.wav", "lone.avi", "c.trm"]
    assert all(r.parts is None for r in apart)


def test_parts_record_checks_its_shape() -> None:
    with pytest.raises(ValueError):
        Parts((Path("a"), Path("b")), (0.0,))
    with pytest.raises(ValueError):
        Parts((Path("a"), Path("b")), (5.0, 0.0))
    assert len(Parts((Path("a"),), (0.0,))) == 1
    assert Recording(Path("a")).parts is None


def test_join_wavs_fills_gaps_and_cuts_overlaps(tmp_path: Path) -> None:
    one = audio.synthetic_wav(tmp_path / "one.wav", 1.0, tone_hz=440.0)
    two = audio.synthetic_wav(tmp_path / "two.wav", 1.0, tone_hz=660.0)
    three = audio.synthetic_wav(tmp_path / "three.wav", 1.0, tone_hz=880.0)
    target = tmp_path / "joined.wav"
    total = audio.join_wavs([(one, 0.0), (two, 1.5), (three, 2.2)], target)
    assert total == pytest.approx(3.2)
    joined = audio.read_wav_mono16k(target)
    rate = audio.SAMPLE_RATE
    assert len(joined) == int(3.2 * rate)
    np.testing.assert_array_equal(joined[: rate], audio.read_wav_mono16k(one))
    assert not joined[rate : int(1.5 * rate)].any()                                    # the gap is silence
    np.testing.assert_array_equal(joined[int(1.5 * rate) : int(2.2 * rate)], audio.read_wav_mono16k(two)[: int(0.7 * rate)])
    np.testing.assert_array_equal(joined[int(2.2 * rate) :], audio.read_wav_mono16k(three))
    with pytest.raises(ValueError):
        audio.join_wavs([(one, 1.0), (two, 0.0)], tmp_path / "x.wav")
    with pytest.raises(ValueError):
        audio.join_wavs([], tmp_path / "x.wav")


def test_the_pipeline_transcribes_two_parts_as_one_recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("TWINSCRIBE_HOME", str(home))
    folder = tmp_path / "session"
    folder.mkdir()
    first = audio.synthetic_wav(folder / "room_0903.wav", 3.0)
    second = audio.synthetic_wav(folder / "room_0907.wav", 3.0, tone_hz=660.0)
    models = make_models(tmp_path / "models")
    job = Job(source=first, profile=profile_for("standard"), models=models, work_folder=tmp_path / "work",
              plan=make_plan_for(), parts=Parts((first, second), (0.0, 5.0)))
    result = pipeline.process_file(job, engines=make_engines())
    source = result.document["source"]
    assert source["name"] == "room_0903.wav" and len(source["parts"]) == 2
    assert [p["name"] for p in source["parts"]] == ["room_0903.wav", "room_0907.wav"]
    assert [p["offset_s"] for p in source["parts"]] == [0.0, 5.0]
    assert [p["duration_s"] for p in source["parts"]] == pytest.approx([3.0, 3.0])
    assert source["bytes"] == first.stat().st_size + second.stat().st_size
    assert all(len(p["sha256"]) == 64 for p in source["parts"]) and source["sha256"] != source["parts"][0]["sha256"]
    review = json.loads(result.outputs.review.read_text(encoding="utf-8"))
    assert review["audio"] == "room_0903.wav"
    assert review["parts"] == [{"audio": "room_0903.wav", "offset_s": 0.0}, {"audio": "room_0907.wav", "offset_s": 5.0}]
    record = json.loads(result.outputs.run.read_text(encoding="utf-8"))
    assert record["input_path"].endswith("room_0903.wav") and [p["offset_s"] for p in record["parts"]] == [0.0, 5.0]
    # The joined audio is kept as the recording's playable copy, and spans the gap.
    copy = playable_copy_path(first)
    assert copy.is_file() and copy.parent == home / "play"
    assert audio.duration_s(copy) == pytest.approx(8.0)
    assert not list((tmp_path / "work").glob("*.part*"))
    assert result.outputs.transcript.name == "room_0903.transcript.json"
