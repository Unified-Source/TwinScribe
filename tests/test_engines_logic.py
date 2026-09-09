"""Unit tests for the pure parts of twinscribe.engines: presets, records, token-to-word
grouping, timing bookkeeping and model-directory validation. Live engine calls skip with the
reason when a library or model is absent."""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from twinscribe import audio
from twinscribe.engines import base, diarize, parakeet, presets, whisper_ct2
from twinscribe.engines.base import Segment, SpeakerTurn, Transcript, Word

# ---------------------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------------------


def test_whisper_production_preset_values():
    p = presets.WHISPER_PRESETS["production"]
    assert p["beam_size"] == 5
    assert p["vad_filter"] is False
    assert p["no_speech_threshold"] == 0.7
    assert p["condition_on_previous_text"] is False
    assert p["word_timestamps"] is True
    assert p["compression_ratio_threshold"] == 2.4
    assert p["log_prob_threshold"] == -1.0
    assert tuple(p["temperature"]) == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert "vad_parameters" not in p


def test_whisper_benchmark_preset_values():
    p = presets.WHISPER_PRESETS["benchmark"]
    assert p["beam_size"] == 1
    assert p["vad_filter"] is True
    assert p["vad_parameters"] == {"min_silence_duration_ms": 500}
    assert p["condition_on_previous_text"] is True
    assert p["word_timestamps"] is False
    assert p["compression_ratio_threshold"] == 2.4
    assert p["log_prob_threshold"] == -1.0
    assert tuple(p["temperature"]) == presets.DEFAULT_TEMPERATURE_LADDER


def test_parakeet_vad_preset_values():
    p = presets.PARAKEET_PRESETS["vad"]
    assert p["vad"] == "silero"
    assert p["threshold"] == 0.5
    assert p["min_silence_duration"] == 0.5
    assert p["min_speech_duration"] == 0.25
    assert p["max_speech_duration"] == 20.0
    assert p["window_size"] == 512
    assert p["decoding_method"] == "greedy_search"


def test_whisper_preset_keys_are_library_parameters():
    for name, preset in presets.WHISPER_PRESETS.items():
        unknown = set(preset) - whisper_ct2.TRANSCRIBE_PARAMETERS
        assert not unknown, f"{name}: {unknown}"


def test_get_preset_returns_independent_copy():
    copy = presets.get_whisper_preset("benchmark")
    copy["vad_parameters"]["min_silence_duration_ms"] = 1
    copy["beam_size"] = 99
    assert presets.WHISPER_PRESETS["benchmark"]["vad_parameters"] == {"min_silence_duration_ms": 500}
    assert presets.WHISPER_PRESETS["benchmark"]["beam_size"] == 1
    other = presets.get_parakeet_preset("vad")
    other["threshold"] = 0.9
    assert presets.PARAKEET_PRESETS["vad"]["threshold"] == 0.5


def test_unknown_preset_names_the_known_ones():
    with pytest.raises(KeyError) as excinfo:
        presets.get_whisper_preset("fast")
    assert "production" in str(excinfo.value) and "benchmark" in str(excinfo.value)
    with pytest.raises(KeyError) as excinfo:
        presets.get_parakeet_preset("none")
    assert "vad" in str(excinfo.value)


def test_resolve_preset_accepts_name_or_mapping():
    name, settings = presets.resolve_preset("production", presets.WHISPER_PRESETS, "whisper")
    assert name == "production" and settings["beam_size"] == 5
    name, settings = presets.resolve_preset({"beam_size": 2}, presets.WHISPER_PRESETS, "whisper")
    assert name == presets.CUSTOM_PRESET_NAME and settings == {"beam_size": 2}
    with pytest.raises(TypeError):
        presets.resolve_preset(3, presets.WHISPER_PRESETS, "whisper")  # type: ignore[arg-type]


def test_transcribe_kwargs_converts_ladder_and_rejects_unknown_keys():
    kwargs = whisper_ct2.transcribe_kwargs(presets.WHISPER_PRESETS["production"])
    assert kwargs["temperature"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert isinstance(kwargs["temperature"], list)
    scalar = whisper_ct2.transcribe_kwargs({"temperature": 0.0})
    assert scalar["temperature"] == 0.0
    with pytest.raises(ValueError) as excinfo:
        whisper_ct2.transcribe_kwargs({"beam_size": 1, "bean_size": 2})
    assert "bean_size" in str(excinfo.value)


# ---------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------


def _transcript(segments) -> Transcript:
    return Transcript(
        engine="test",
        model="m",
        preset="p",
        segments=tuple(segments),
        audio_s=10.0,
        load_s=0.0,
        transcribe_s=0.0,
    )


def test_transcript_words_flatten_and_order_by_start():
    late = Segment(5.0, 7.0, "c d", (Word("c", 5.0, 6.0), Word("d", 6.0, 7.0)))
    early = Segment(0.0, 2.0, "a b", (Word("a", 0.0, 1.0), Word("b", 1.0, 2.0)))
    transcript = _transcript([late, early])
    assert [w.text for w in transcript.words] == ["a", "b", "c", "d"]


def test_transcript_words_stable_for_equal_starts():
    seg = Segment(0.0, 1.0, "x y", (Word("x", 0.0, 0.5), Word("y", 0.0, 0.5)))
    assert [w.text for w in _transcript([seg]).words] == ["x", "y"]


def test_transcript_text_joins_segments_and_skips_empty():
    segments = [
        Segment(0.0, 1.0, " hello "),
        Segment(1.0, 2.0, ""),
        Segment(2.0, 3.0, "   "),
        Segment(3.0, 4.0, "world"),
    ]
    assert _transcript(segments).text == "hello world"
    assert _transcript([]).text == ""
    assert _transcript([]).words == []


def test_records_are_immutable():
    word = Word("a", 0.0, 1.0)
    with pytest.raises(Exception):
        word.text = "b"  # type: ignore[misc]
    with pytest.raises(Exception):
        _transcript([]).engine = "x"  # type: ignore[misc]


def test_speaker_turn_duration_never_negative():
    assert SpeakerTurn(1.0, 3.5, "speaker_00").duration == pytest.approx(2.5)
    assert SpeakerTurn(3.0, 2.0, "speaker_00").duration == 0.0


def test_diarization_labels_in_first_appearance_order():
    turns = (
        SpeakerTurn(0.0, 1.0, "speaker_01"),
        SpeakerTurn(1.0, 2.0, "speaker_00"),
        SpeakerTurn(2.0, 3.0, "speaker_01"),
    )
    result = base.Diarization(
        segments=turns,
        seconds_per_label=diarize.seconds_per_label(turns),
        audio_s=3.0,
        load_s=0.0,
        diarize_s=0.0,
    )
    assert result.labels == ["speaker_01", "speaker_00"]


# ---------------------------------------------------------------------------------------
# Timing bookkeeping and thread defaults
# ---------------------------------------------------------------------------------------


def test_stopwatch_measures_elapsed_block():
    with base.Stopwatch() as watch:
        assert watch.seconds == 0.0
        time.sleep(0.02)
    assert watch.seconds >= 0.015
    assert watch.seconds < 5.0


def test_elapsed_since_is_non_negative_and_increases():
    start = time.perf_counter()
    first = base.elapsed_since(start)
    time.sleep(0.005)
    second = base.elapsed_since(start)
    assert 0.0 <= first <= second


def test_default_threads():
    assert base.default_threads(None) == max(1, min(8, os.cpu_count() or 1))
    assert base.default_threads() <= 8
    assert base.default_threads(3) == 3
    with pytest.raises(ValueError):
        base.default_threads(0)
    with pytest.raises(ValueError):
        base.default_threads(-2)


def test_offline_env_is_forced(monkeypatch):
    for key in whisper_ct2.OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    in_effect = whisper_ct2.set_offline_env()
    assert in_effect["HF_HUB_OFFLINE"] == "1"
    for key, value in whisper_ct2.OFFLINE_ENV.items():
        assert os.environ[key] == value


# ---------------------------------------------------------------------------------------
# Token-to-word grouping
# ---------------------------------------------------------------------------------------


def _spans(words):
    return [(w.text, w.start, w.end) for w in words]


def test_group_words_basic():
    words = parakeet.group_words([" hel", "lo", " world"], [0.1, 0.2, 0.5], segment_end=1.0)
    assert _spans(words) == [("hello", 0.1, 0.5), ("world", 0.5, 1.0)]
    assert all(w.prob is None for w in words)


def test_group_words_bare_marker_at_end_is_dropped():
    words = parakeet.group_words([" hi", " "], [0.0, 0.3], segment_end=0.6)
    assert _spans(words) == [("hi", 0.0, 0.6)]


def test_group_words_bare_marker_followed_by_continuation_starts_the_word():
    words = parakeet.group_words([" ", "hi"], [0.1, 0.2], segment_end=0.5)
    assert _spans(words) == [("hi", 0.1, 0.5)]


def test_group_words_bare_marker_between_words_is_dropped():
    words = parakeet.group_words([" a", " ", " b"], [0.0, 0.2, 0.4], segment_end=1.0)
    assert _spans(words) == [("a", 0.0, 0.4), ("b", 0.4, 1.0)]


def test_group_words_first_token_without_marker_opens_a_word():
    words = parakeet.group_words(["hel", "lo", " x"], [0.0, 0.1, 0.3], segment_end=0.4)
    assert _spans(words) == [("hello", 0.0, 0.3), ("x", 0.3, 0.4)]


def test_group_words_applies_offset():
    words = parakeet.group_words([" a", " b"], [0.0, 0.5], segment_end=1.0, offset=10.0)
    assert _spans(words) == [("a", 10.0, 10.5), ("b", 10.5, 11.0)]


def test_group_words_empty_and_only_markers():
    assert parakeet.group_words([], [], segment_end=1.0) == []
    assert parakeet.group_words([" ", " "], [0.0, 0.1], segment_end=1.0) == []


def test_group_words_end_never_before_start():
    words = parakeet.group_words([" a"], [0.9], segment_end=0.5)
    assert _spans(words) == [("a", 0.9, 0.9)]


def test_group_words_pads_short_timestamp_list():
    words = parakeet.group_words([" a", " b", " c"], [0.0, 0.2], segment_end=1.0)
    assert _spans(words) == [("a", 0.0, 0.2), ("b", 0.2, 0.2), ("c", 0.2, 1.0)]
    words = parakeet.group_words([" a", "b"], [], segment_end=1.0)
    assert _spans(words) == [("ab", 0.0, 1.0)]


@pytest.mark.parametrize(
    "tokens",
    [
        [" the", " quick", " brown", " fox"],
        [" un", "believ", "able", " ", "ly", " so"],
        ["x", " ", " y", " ", " "],
        [" a", " b", " ", "c", " d"],
    ],
)
def test_group_words_invariants(tokens):
    stamps = [0.05 * i for i in range(len(tokens))]
    end = 0.05 * len(tokens) + 0.1
    words = parakeet.group_words(tokens, stamps, segment_end=end, offset=2.0)
    for word in words:
        assert word.text and not word.text.startswith(parakeet.WORD_MARKER)
        assert word.end >= word.start
    for earlier, later in zip(words, words[1:]):
        assert earlier.end == later.start
        assert earlier.start <= later.start
    if words:
        assert words[-1].end == max(end + 2.0, words[-1].start)
    joined = " ".join(w.text for w in words)
    assert joined == " ".join("".join(tokens).split())


def test_segment_text_comes_from_grouped_words():
    # The bare marker opens a word that "b" continues, so "b" is its own word starting at
    # the marker's time; the text is built from those groups, never from an engine field.
    words = parakeet.group_words([" a", " ", "b", " c"], [0.0, 0.1, 0.2, 0.3], segment_end=0.5)
    segment = parakeet.segment_from_words(words, 0.0, 0.5)
    assert _spans(words) == [("a", 0.0, 0.1), ("b", 0.1, 0.3), ("c", 0.3, 0.5)]
    assert segment.text == "a b c"
    assert segment.words == tuple(words)
    assert segment.quality == {}
    assert parakeet.segment_from_words([], 1.0, 2.0).text == ""


# ---------------------------------------------------------------------------------------
# Whisper segment conversion on fake library objects
# ---------------------------------------------------------------------------------------


def _fake_library_segment(with_words: bool):
    words = None
    if with_words:
        words = [
            SimpleNamespace(word=" Hello", start=0.0, end=0.4, probability=0.9),
            SimpleNamespace(word=" there", start=0.4, end=0.8, probability=None),
        ]
    return SimpleNamespace(
        start=0.0,
        end=0.8,
        text=" Hello there ",
        avg_logprob=-0.25,
        no_speech_prob=0.05,
        compression_ratio=1.3,
        temperature=0.0,
        words=words,
    )


def test_segment_from_library_with_words():
    segment = whisper_ct2.segment_from_library(_fake_library_segment(True), want_words=True)
    assert segment.text == "Hello there"
    assert [w.text for w in segment.words] == ["Hello", "there"]
    assert segment.words[0].prob == 0.9 and segment.words[1].prob is None
    assert segment.quality == {
        "avg_logprob": -0.25,
        "no_speech_prob": 0.05,
        "compression_ratio": 1.3,
        "temperature": 0.0,
    }


def test_segment_from_library_without_words():
    segment = whisper_ct2.segment_from_library(_fake_library_segment(False), want_words=False)
    assert segment.words == ()
    segment = whisper_ct2.segment_from_library(_fake_library_segment(True), want_words=False)
    assert segment.words == ()
    bare = SimpleNamespace(start=1.0, end=2.0, text="x", words=None)
    segment = whisper_ct2.segment_from_library(bare, want_words=True)
    assert segment.quality == {
        "avg_logprob": None,
        "no_speech_prob": None,
        "compression_ratio": None,
        "temperature": None,
    }


# ---------------------------------------------------------------------------------------
# Diarization helpers on fake library objects
# ---------------------------------------------------------------------------------------


def test_label_for():
    assert diarize.label_for(0) == "speaker_00"
    assert diarize.label_for(12) == "speaker_12"
    with pytest.raises(ValueError):
        diarize.label_for(-1)


def test_turns_from_result_orders_and_labels():
    records = [
        SimpleNamespace(start=5.0, end=6.0, speaker=1),
        SimpleNamespace(start=0.0, end=2.5, speaker=0),
        SimpleNamespace(start=2.5, end=5.0, speaker=1),
    ]
    turns = diarize.turns_from_result(records)
    assert [(t.start, t.end, t.label) for t in turns] == [
        (0.0, 2.5, "speaker_00"),
        (2.5, 5.0, "speaker_01"),
        (5.0, 6.0, "speaker_01"),
    ]
    totals = diarize.seconds_per_label(turns)
    assert totals == {"speaker_00": pytest.approx(2.5), "speaker_01": pytest.approx(3.5)}
    assert sum(totals.values()) == pytest.approx(sum(t.duration for t in turns))
    assert diarize.seconds_per_label([]) == {}


def test_turns_from_result_renumbers_clusters_by_first_appearance():
    # The library can skip cluster numbers; labels are contiguous in order of first appearance.
    records = [
        SimpleNamespace(start=10.0, end=12.0, speaker=0),
        SimpleNamespace(start=0.0, end=4.0, speaker=3),
        SimpleNamespace(start=4.0, end=10.0, speaker=0),
        SimpleNamespace(start=12.0, end=13.0, speaker=7),
    ]
    turns = diarize.turns_from_result(records)
    assert [t.label for t in turns] == ["speaker_00", "speaker_01", "speaker_01", "speaker_02"]
    assert turns[0].start == 0.0 and turns[-1].end == 13.0


# ---------------------------------------------------------------------------------------
# Model-directory validation (no library needed)
# ---------------------------------------------------------------------------------------


def test_resolve_model_files_prefers_int8(tmp_path):
    for name in ("encoder.int8.onnx", "encoder.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (tmp_path / name).write_bytes(b"x")
    files = parakeet.resolve_model_files(tmp_path)
    assert files["encoder"].name == "encoder.int8.onnx"
    assert set(files) == {"encoder", "decoder", "joiner", "tokens"}


def test_resolve_model_files_accepts_fp32_names(tmp_path):
    for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        (tmp_path / name).write_bytes(b"x")
    files = parakeet.resolve_model_files(tmp_path)
    assert files["joiner"].name == "joiner.onnx"


def test_resolve_model_files_names_missing_parts(tmp_path):
    (tmp_path / "encoder.int8.onnx").write_bytes(b"x")
    with pytest.raises(FileNotFoundError) as excinfo:
        parakeet.resolve_model_files(tmp_path)
    message = str(excinfo.value)
    assert "decoder" in message and "joiner" in message and "tokens" in message
    with pytest.raises(FileNotFoundError):
        parakeet.resolve_model_files(tmp_path / "absent")


def test_whisper_check_model_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        whisper_ct2.check_model_dir(tmp_path / "absent")
    (tmp_path / "model.bin").write_bytes(b"x")
    with pytest.raises(FileNotFoundError) as excinfo:
        whisper_ct2.check_model_dir(tmp_path)
    assert "config.json" in str(excinfo.value)
    (tmp_path / "config.json").write_text("{}")
    assert whisper_ct2.check_model_dir(tmp_path) == tmp_path


def test_transcribe_validates_inputs_before_importing_libraries(tmp_path):
    wav = audio.synthetic_wav(tmp_path / "tone.wav", seconds=0.2)
    with pytest.raises(FileNotFoundError):
        whisper_ct2.transcribe(tmp_path / "missing.wav", tmp_path, "production")
    with pytest.raises(FileNotFoundError):
        whisper_ct2.transcribe(wav, tmp_path / "no-model", "production")
    with pytest.raises(FileNotFoundError):
        parakeet.transcribe(wav, tmp_path / "no-model", tmp_path / "no-vad.onnx", "vad")
    with pytest.raises(FileNotFoundError):
        diarize.diarize(wav, tmp_path / "seg.onnx", tmp_path / "emb.onnx")
    (tmp_path / "seg.onnx").write_bytes(b"x")
    (tmp_path / "emb.onnx").write_bytes(b"x")
    with pytest.raises(ValueError):
        diarize.diarize(wav, tmp_path / "seg.onnx", tmp_path / "emb.onnx", num_speakers=0)
    with pytest.raises(ValueError):
        diarize.diarize(wav, tmp_path / "seg.onnx", tmp_path / "emb.onnx", threshold=0.0)


def test_register_windows_dlls_returns_directories_or_nothing():
    registered = parakeet.register_windows_dlls()
    assert isinstance(registered, tuple)
    if os.name != "nt" or importlib.util.find_spec("sherpa_onnx") is None:
        assert registered == ()
    for directory in registered:
        assert Path(directory).is_dir()


def test_library_versions_without_libraries():
    if importlib.util.find_spec("faster_whisper") is None:
        assert "faster_whisper" not in whisper_ct2.library_versions()
    if importlib.util.find_spec("sherpa_onnx") is None:
        assert parakeet.library_versions() == {}
    fake = SimpleNamespace(version=lambda: "1.13.0", onnxruntime_version="1.17.1", git_sha1=None)
    assert parakeet.library_versions(fake) == {"sherpa_onnx": "1.13.0", "onnxruntime": "1.17.1"}


# ---------------------------------------------------------------------------------------
# Live calls: skip unless the library and the model are present
# ---------------------------------------------------------------------------------------


def _model_from_env(variable: str, kind: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"{variable} is not set; no {kind} to run against")
    path = Path(value)
    if not path.exists():
        pytest.skip(f"{variable} points at {path}, which does not exist")
    return path


def _require_library(module_name: str, package: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        pytest.skip(f"{package} is not installed")


def test_whisper_live(tmp_path):
    _require_library("faster_whisper", "faster-whisper")
    model_dir = _model_from_env("TWINSCRIBE_WHISPER_MODEL_DIR", "Whisper model directory")
    wav = audio.synthetic_wav(tmp_path / "tone.wav", seconds=3.0)
    transcript = whisper_ct2.transcribe(wav, model_dir, "production", threads=2)
    assert transcript.engine == whisper_ct2.ENGINE_NAME
    assert transcript.audio_s == pytest.approx(3.0, abs=0.05)
    assert transcript.load_s >= 0.0 and transcript.transcribe_s >= 0.0
    assert transcript.versions
    assert transcript.extras["segments"] == len(transcript.segments)


def test_parakeet_live(tmp_path):
    _require_library("sherpa_onnx", "sherpa-onnx")
    model_dir = _model_from_env("TWINSCRIBE_PARAKEET_MODEL_DIR", "transducer model directory")
    vad_model = _model_from_env("TWINSCRIBE_SILERO_VAD", "Silero VAD model file")
    wav = audio.synthetic_wav(tmp_path / "tone.wav", seconds=3.0)
    transcript = parakeet.transcribe(wav, model_dir, vad_model, "vad", threads=2)
    assert transcript.engine == parakeet.ENGINE_NAME
    assert transcript.audio_s == pytest.approx(3.0)
    assert transcript.load_s >= 0.0 and transcript.transcribe_s >= transcript.extras["decode_s"]
    assert transcript.extras["vad_segments"] == len(transcript.segments)
    assert transcript.versions


def test_diarize_live(tmp_path):
    _require_library("sherpa_onnx", "sherpa-onnx")
    segmentation = _model_from_env("TWINSCRIBE_SEGMENTATION_MODEL", "segmentation model file")
    embedding = _model_from_env("TWINSCRIBE_EMBEDDING_MODEL", "embedding model file")
    wav = audio.synthetic_wav(tmp_path / "tone.wav", seconds=3.0)
    result = diarize.diarize(wav, segmentation, embedding, threads=2)
    assert result.audio_s == pytest.approx(3.0)
    assert result.load_s >= 0.0 and result.diarize_s >= 0.0
    assert set(result.seconds_per_label) == set(result.labels)
    assert result.settings["num_speakers"] is None


def test_stream_kwargs_drop_the_voice_filter_and_any_clips():
    kwargs = whisper_ct2.transcribe_kwargs(presets.WHISPER_PRESETS["production"])
    streamed = whisper_ct2.stream_kwargs(dict(kwargs, clip_timestamps=[1.0, 2.0]))
    assert "vad_filter" not in streamed and "vad_parameters" not in streamed and "clip_timestamps" not in streamed
    assert streamed["beam_size"] == kwargs["beam_size"] and "vad_filter" in kwargs


def test_restore_time_maps_the_stream_back_to_the_recording():
    # Two pieces: recording 10-12 s sits at stream 0-2 s, recording 20-23 s at stream 2-5 s.
    pieces = [(0.0, 10.0, 12.0), (2.0, 20.0, 23.0)]
    assert whisper_ct2.restore_time(pieces, 0.0) == 10.0
    assert whisper_ct2.restore_time(pieces, 1.5) == 11.5
    assert whisper_ct2.restore_time(pieces, 2.0) == 20.0
    assert whisper_ct2.restore_time(pieces, 4.5) == 22.5
    # Past the last piece the time clamps to its end; without pieces it is unchanged.
    assert whisper_ct2.restore_time(pieces, 6.0) == 23.0
    assert whisper_ct2.restore_time([], 3.0) == 3.0


def test_concatenate_clips_joins_the_samples_and_records_the_pieces():
    import numpy as np

    samples = np.arange(16000 * 4, dtype=np.float32)
    stream, pieces = whisper_ct2.concatenate_clips(samples, [(0.5, 1.0), (3.0, 3.25), (3.9, 3.9)])
    assert len(stream) == 16000 // 2 + 16000 // 4 and stream.dtype == np.float32
    assert pieces == [(0.0, 0.5, 1.0), (0.5, 3.0, 3.25)]
    assert stream[0] == 8000.0 and stream[8000] == 48000.0


def test_speech_within_keeps_the_speech_inside_the_windows_and_merges():
    windows = [(1.0, 6.0), (8.0, 10.0)]
    speech = [(0.0, 2.0), (3.0, 4.0), (5.5, 9.0), (9.5, 12.0)]
    # The first window keeps 1-2, 3-4 and 5.5-6; the second keeps 8-9 and 9.5-10.
    assert whisper_ct2.speech_within(windows, speech) == [(1.0, 2.0), (3.0, 4.0), (5.5, 6.0), (8.0, 9.0), (9.5, 10.0)]
    # Speech spanning two touching windows merges into one clip; silence-only windows vanish.
    assert whisper_ct2.speech_within([(0.0, 5.0), (5.0, 8.0)], [(4.0, 6.0)]) == [(4.0, 6.0)]
    assert whisper_ct2.speech_within([(0.0, 5.0)], [(6.0, 7.0)]) == []
    assert whisper_ct2.speech_within([], [(0.0, 1.0)]) == [] and whisper_ct2.speech_within([(0.0, 1.0)], []) == []
