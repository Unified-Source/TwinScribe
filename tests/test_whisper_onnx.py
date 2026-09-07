"""Tests for the ONNX Whisper detector: model file resolution, the quiet-cut windowing, word
placement by segment, the token and segment decode paths on a stand-in recogniser, the
preset; the live call skips unless the model is named in the environment."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from twinscribe import audio
from twinscribe.engines import whisper_onnx
from twinscribe.engines.whisper_onnx import (
    DEFAULT_PRESET,
    TIMING_SEGMENT,
    TIMING_TOKEN,
    WHISPER_ONNX_PRESETS,
    decode_window,
    plan_windows,
    resolve_model_files,
    words_from_segments,
)

RATE = 16000


def test_preset_keeps_windows_under_the_stream_limit() -> None:
    preset = WHISPER_ONNX_PRESETS[DEFAULT_PRESET]
    assert preset["window_s"] <= whisper_onnx.MAX_STREAM_S
    assert preset["language"] == "en" and preset["task"] == "transcribe"
    assert preset["decoding_method"] == "greedy_search"


def test_resolve_model_files_by_ending(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_model_files(tmp_path / "absent")
    for name in ("turbo-encoder.onnx", "turbo-decoder.onnx", "turbo-tokens.txt"):
        (tmp_path / name).write_bytes(b"x")
    files = resolve_model_files(tmp_path)
    assert files["encoder"].name == "turbo-encoder.onnx" and files["tokens"].name == "turbo-tokens.txt"
    (tmp_path / "turbo-encoder.int8.onnx").write_bytes(b"x")
    assert resolve_model_files(tmp_path)["encoder"].name == "turbo-encoder.int8.onnx"
    (tmp_path / "turbo-tokens.txt").unlink()
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_model_files(tmp_path)
    assert "tokens" in str(excinfo.value)


def _noisy(seconds: float, silences: list[tuple[float, float]]) -> np.ndarray:
    rng = np.random.default_rng(3)
    signal = rng.normal(0.0, 0.3, int(seconds * RATE)).astype(np.float32)
    for start, end in silences:
        signal[int(start * RATE) : int(end * RATE)] = 0.0
    return signal


def test_windows_cut_at_quiet_moments_and_cover_everything() -> None:
    signal = _noisy(100.0, [(27.0, 27.6), (56.2, 56.8), (84.9, 85.5)])
    windows = plan_windows(signal, RATE)
    assert windows[0][0] == 0 and windows[-1][1] == len(signal)
    for (a, b), (c, _) in zip(windows, windows[1:]):
        assert b == c
    for a, b in windows:
        assert 0 < b - a <= 30 * RATE
    cuts = [b / RATE for a, b in windows[:-1]]
    assert len(cuts) == 3
    for cut, (start, end) in zip(cuts, [(27.0, 27.6), (56.2, 56.8), (84.9, 85.5)]):
        assert start <= cut <= end


def test_windows_without_silence_and_short_audio() -> None:
    signal = _noisy(70.0, [])
    windows = plan_windows(signal, RATE)
    assert windows[-1][1] == len(signal) and all(24 * RATE <= b - a <= 30 * RATE for a, b in windows[:-1])
    assert plan_windows(_noisy(12.0, []), RATE) == [(0, 12 * RATE)]
    assert plan_windows(np.zeros(0, dtype=np.float32), RATE) == []
    # A very long silence: the cut lands inside it, never before half the window.
    long_silence = _noisy(40.0, [(10.0, 30.0)])
    first_cut = plan_windows(long_silence, RATE)[0][1] / RATE
    assert 15.0 <= first_cut <= 30.0
    with pytest.raises(ValueError):
        plan_windows(signal, RATE, window_s=45.0)
    with pytest.raises(ValueError):
        plan_windows(signal, RATE, quiet_s=0.0)


def test_words_from_segments_spread_by_characters_and_clamp() -> None:
    words = words_from_segments(["ab cdef", "g"], [0.0, 4.0], [3.0, 10.0], window_len_s=6.0, offset=10.0)
    assert [w.text for w in words] == ["ab", "cdef", "g"]
    assert words[0].start == pytest.approx(10.0) and words[0].end == pytest.approx(10.0 + 3.0 * 3 / 8)
    assert words[1].end == pytest.approx(13.0)
    assert words[2].start == pytest.approx(14.0) and words[2].end == pytest.approx(16.0)   # clamped to the window
    starts = [w.start for w in words]
    assert starts == sorted(starts) and all(w.end >= w.start for w in words)


def test_words_are_confined_to_voiced_parts() -> None:
    voiced = [(11.0, 12.0), (13.0, 14.0), (20.0, 21.0)]
    words = words_from_segments(["hello there"], [0.0], [5.0], window_len_s=5.0, offset=10.0, voiced=voiced)
    assert [w.text for w in words] == ["hello", "there"]
    assert (words[0].start, words[0].end) == (pytest.approx(11.0), pytest.approx(12.0))
    assert (words[1].start, words[1].end) == (pytest.approx(13.0), pytest.approx(14.0))
    # No voiced part inside the span: the whole span is used.
    fallback = words_from_segments(["a b"], [0.0], [2.0], window_len_s=2.0, offset=30.0, voiced=voiced)
    assert fallback[0].start == pytest.approx(30.0) and fallback[1].end == pytest.approx(32.0)
    # Three words over two unequal voiced parts stay in order and inside the parts.
    three = words_from_segments(["aa bb cc"], [0.0], [10.0], window_len_s=10.0, offset=0.0, voiced=[(1.0, 2.0), (5.0, 7.0)])
    assert all(1.0 <= w.start <= 7.0 and w.end <= 7.0 for w in three)
    assert [w.start for w in three] == sorted(w.start for w in three)
    assert not any(2.0 < w.start < 5.0 for w in three)


def test_words_from_segments_without_timing_or_text() -> None:
    assert words_from_segments([], [], [], 5.0) == []
    whole = words_from_segments(["one two"], [], [], window_len_s=4.0)
    assert [(w.start, w.end) for w in whole] == [(0.0, 2.0), (2.0, 4.0)]
    assert words_from_segments(["   "], [0.0], [1.0], 2.0) == []
    zero = words_from_segments(["a", "b"], [1.0, 1.0], [0.0, 0.0], window_len_s=3.0)
    assert zero[0].end >= zero[0].start and zero[1].end == pytest.approx(3.0)


class _Stream:
    def __init__(self, result) -> None:
        self.result = result

    def accept_waveform(self, rate: int, samples) -> None:
        self.rate = rate
        self.count = len(samples)


class _Recognizer:
    def __init__(self, result) -> None:
        self._result = result
        self.decoded = 0

    def create_stream(self) -> _Stream:
        return _Stream(self._result)

    def decode_stream(self, stream: _Stream) -> None:
        self.decoded += 1


def test_decode_window_uses_token_times_when_present() -> None:
    result = SimpleNamespace(tokens=[" Good", " morning", ","], timestamps=[0.1, 0.5, 0.9], text=" Good morning,",
                             segment_texts=[], segment_timestamps=[], segment_durations=[])
    words, by_token = decode_window(_Recognizer(result), np.zeros(RATE, dtype=np.float32), offset_s=2.0)
    assert by_token is True
    assert [w.text for w in words] == ["Good", "morning,"]
    assert words[0].start == pytest.approx(2.1) and words[0].end == pytest.approx(2.5)
    assert words[1].end == pytest.approx(3.0)      # the window end, one second in


def test_decode_window_falls_back_to_segments() -> None:
    result = SimpleNamespace(tokens=[" Good", " morning"], timestamps=[], text=" Good morning",
                             segment_texts=[" Good morning"], segment_timestamps=[0.0], segment_durations=[7.0])
    words, by_token = decode_window(_Recognizer(result), np.zeros(2 * RATE, dtype=np.float32), offset_s=0.0)
    assert by_token is False
    assert [w.text for w in words] == ["Good", "morning"]
    assert words[-1].end == pytest.approx(2.0)      # clamped to the two second window
    bare = SimpleNamespace(tokens=[], timestamps=[], text=" hello there", segment_texts=[], segment_timestamps=[], segment_durations=[])
    words, by_token = decode_window(_Recognizer(bare), np.zeros(RATE, dtype=np.float32), offset_s=5.0)
    assert by_token is False and [w.text for w in words] == ["hello", "there"] and words[0].start == pytest.approx(5.0)


def test_timing_names() -> None:
    assert TIMING_TOKEN == "token" and TIMING_SEGMENT == "segment"


def _from_env(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"{variable} not set")
    path = Path(value)
    if not path.exists():
        pytest.skip(f"{variable} points at nothing: {path}")
    return path


def test_whisper_onnx_live(tmp_path: Path) -> None:
    pytest.importorskip("sherpa_onnx", reason="sherpa-onnx is not installed")
    model_dir = _from_env("TWINSCRIBE_WHISPER_ONNX_MODEL_DIR")
    vad_model = _from_env("TWINSCRIBE_SILERO_VAD")
    wav = audio.synthetic_wav(tmp_path / "tone.wav", 3.0)
    seen: list[float] = []
    transcript = whisper_onnx.transcribe(wav, model_dir, vad_model, threads=2, progress=seen.append)
    assert transcript.engine == "whisper_onnx" and transcript.preset == DEFAULT_PRESET
    assert transcript.audio_s == pytest.approx(3.0)
    assert transcript.load_s > 0.0 and transcript.transcribe_s >= 0.0
    assert transcript.settings["word_timing"] in (TIMING_TOKEN, TIMING_SEGMENT)
    assert transcript.settings["provider"] == "cpu" and transcript.extras["windows"] == 1
    assert seen and seen[-1] == 1.0 and seen == sorted(seen)
    assert "sherpa_onnx" in transcript.versions
