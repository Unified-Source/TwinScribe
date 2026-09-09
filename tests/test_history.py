"""Tests for the history of transcribed recordings: entries from a result, the file round
trip, replacement of an earlier entry for the same recording, removal and clearing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tests._fixtures import make_document
from twinscribe import history
from twinscribe.history import HistoryEntry, append_entry, clear_history, entry_from_result, read_history, remove_entries, write_history
from twinscribe.pipeline import output_paths


def _result(tmp_path: Path, name: str = "call.wav") -> SimpleNamespace:
    source = tmp_path / name
    source.write_bytes(b"x")
    return SimpleNamespace(source=source, outputs=output_paths(source), document=make_document(name), run_record={})


def test_entry_from_result_carries_the_facts(tmp_path: Path) -> None:
    entry = entry_from_result(_result(tmp_path))
    assert entry.name == "call.wav" and entry.path == str(tmp_path / "call.wav")
    assert entry.duration_s == 30.0 and entry.profile == "standard" and entry.speakers == 2
    assert entry.marks == 2 and entry.publisher == "parakeet-tdt-0.6b-v2-int8" and entry.detector == "whisper-large-v3-turbo-ct2"
    assert entry.outputs["transcript"].endswith("call.transcript.json") and entry.transcript_path is not None
    assert entry.outputs_present is False and entry.folder == tmp_path
    assert entry.sha256 == "ab" * 32


def test_round_trip_and_replacement(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    assert read_history(path) == []
    first = entry_from_result(_result(tmp_path, "a.wav"))
    second = entry_from_result(_result(tmp_path, "b.wav"))
    append_entry(first, path)
    entries = append_entry(second, path)
    assert [e.name for e in entries] == ["b.wav", "a.wav"]
    assert [e.name for e in read_history(path)] == ["b.wav", "a.wav"]
    again = HistoryEntry(**{**first.to_dict(), "produced_utc": "2026-02-02T00:00:00+00:00"})
    entries = append_entry(again, path)
    assert [e.name for e in entries] == ["a.wav", "b.wav"]
    assert entries[0].produced_utc == "2026-02-02T00:00:00+00:00"
    # The same recording under another spelling of its path is the same entry.
    variant = HistoryEntry(**{**first.to_dict(), "path": str(tmp_path / "A.WAV").replace("/", "\\")})
    assert len(append_entry(variant, path)) == 2


def test_remove_clear_and_tolerance(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    a = entry_from_result(_result(tmp_path, "a.wav"))
    b = entry_from_result(_result(tmp_path, "b.wav"))
    write_history([a, b], path)
    assert [e.name for e in remove_entries([a.path], path)] == ["b.wav"]
    clear_history(path)
    assert read_history(path) == []
    path.write_text('{"schema": "other", "entries": []}', encoding="utf-8")
    assert read_history(path) == []
    path.write_text("not json", encoding="utf-8")
    assert read_history(path) == []
    write_history([HistoryEntry.from_dict({"path": str(tmp_path / "c.wav")})], path)
    entry = read_history(path)[0]
    assert entry.name == "c.wav" and entry.marks == 0 and entry.outputs == {}


def test_cap_on_length(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(history, "MAX_ENTRIES", 3)
    path = tmp_path / "history.json"
    for i in range(5):
        append_entry(HistoryEntry.from_dict({"path": str(tmp_path / f"{i}.wav")}), path)
    assert [e.name for e in read_history(path)] == ["4.wav", "3.wav", "2.wav"]


def test_history_path_follows_the_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TWINSCRIBE_HOME", str(tmp_path / "home"))
    assert history.history_path() == tmp_path / "home" / "history.json"
    assert history.history_path(tmp_path) == tmp_path / "history.json"
