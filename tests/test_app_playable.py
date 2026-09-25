"""Tests for the playable copy: where a copy goes, when one counts as current, and the copy
made through the decoder for a recording the platform's player cannot read."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from twinscribe import audio  # noqa: E402
from twinscribe.app import playable  # noqa: E402


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "home"
    monkeypatch.setenv("TWINSCRIBE_HOME", str(folder))
    return folder


def test_copy_paths_keep_two_recordings_of_one_name_apart(tmp_path: Path, home: Path) -> None:
    one = tmp_path / "a" / "call.trm"
    two = tmp_path / "b" / "call.trm"
    for path in (one, two):
        path.parent.mkdir()
        path.write_bytes(b"x")
    first, second = playable.playable_copy_path(one), playable.playable_copy_path(two)
    assert first.parent == home / playable.PLAY_FOLDER and first.parent.is_dir()
    assert first.name.startswith("call.") and first.suffix == ".wav"
    assert first != second
    assert playable.playable_copy_path(one) == first


def test_a_copy_is_current_only_when_present_and_not_older_than_its_recording(tmp_path: Path, home: Path) -> None:
    source = tmp_path / "call.trm"
    source.write_bytes(b"x")
    copy = playable.playable_copy_path(source)
    assert not playable.copy_is_current(source, copy)
    copy.write_bytes(b"y")
    assert playable.copy_is_current(source, copy)
    later = time.time() + 60
    os.utime(source, (later, later))
    assert not playable.copy_is_current(source, copy)
    assert not playable.copy_is_current(tmp_path / "missing.trm", copy)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not on the search path")
def test_the_copy_is_the_engines_own_format_and_is_made_once(tmp_path: Path, home: Path) -> None:
    source = audio.synthetic_wav(tmp_path / "tone.wav", 2.0)
    source = source.rename(tmp_path / "tone.trm")  # a plain WAV under another name: the decoder reads it
    outcomes: list[object] = []
    maker = playable.PlayableCopy(source)
    maker.ready.connect(outcomes.append)
    maker.failed.connect(outcomes.append)
    maker.run()
    assert outcomes and isinstance(outcomes[0], Path), outcomes
    copy = outcomes[0]
    assert copy == maker.target and copy.is_file() and not playable.partial_path(copy).exists()
    samples = audio.read_wav_mono16k(copy)
    assert abs(len(samples) / audio.SAMPLE_RATE - 2.0) < 0.05
    stamp = copy.stat().st_mtime_ns
    again = playable.PlayableCopy(source)
    again.ready.connect(outcomes.append)
    again.run()
    assert outcomes[-1] == copy and copy.stat().st_mtime_ns == stamp


def test_a_recording_the_decoder_cannot_read_reports_the_reason(tmp_path: Path, home: Path) -> None:
    source = tmp_path / "noise.trm"
    source.write_bytes(b"this is not a recording")
    outcomes: list[object] = []
    maker = playable.PlayableCopy(source)
    maker.ready.connect(outcomes.append)
    maker.failed.connect(outcomes.append)
    maker.run()
    assert len(outcomes) == 1 and isinstance(outcomes[0], str)
    assert not maker.target.exists() and not playable.partial_path(maker.target).exists()
