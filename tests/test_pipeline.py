"""Tests for the per-file pipeline and the batch driver, with synthetic engines."""

from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from tests._fixtures import make_engines, make_models, make_plan_for
from twinscribe import audio
from twinscribe.models import KEY_WHISPER_TURBO, KEY_WHISPER_TURBO_ONNX
from twinscribe.pipeline import (
    BATCH_SCHEMA,
    STAGES,
    Cancelled,
    Job,
    Progress,
    discover_media,
    is_media,
    is_video,
    output_paths,
    process_file,
    run_batch,
)
from twinscribe.profiles import profile_for


@pytest.fixture
def models(tmp_path: Path):
    return make_models(tmp_path / "models")


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    folder = tmp_path / "media"
    folder.mkdir()
    return audio.synthetic_wav(folder / "call.wav", 30.0)


def job_for(source: Path, models, tmp_path: Path, **overrides) -> Job:
    fields = dict(source=source, profile=profile_for("standard"), models=models, work_folder=tmp_path / "work",
                  plan=make_plan_for())
    fields.update(overrides)
    return Job(**fields)


# ----- paths and discovery ------------------------------------------------------------------


def test_output_paths_beside_and_in_folder(tmp_path: Path) -> None:
    beside = output_paths(tmp_path / "a" / "call.mp3")
    assert beside.transcript == tmp_path / "a" / "call.transcript.json"
    assert beside.text == tmp_path / "a" / "call.txt"
    assert beside.docx.name == "call.docx" and beside.subtitles.name == "call.srt"
    assert beside.review.name == "call.review.json" and beside.run.name == "call.run.json"
    assert len(beside.all) == 6 and set(beside.as_dict()) == {"transcript", "text", "docx", "subtitles", "review", "run"}
    elsewhere = output_paths(tmp_path / "a" / "call.mp3", tmp_path / "out")
    assert elsewhere.transcript.parent == tmp_path / "out"


def test_output_paths_avoid_stem_collisions(tmp_path: Path) -> None:
    first = tmp_path / "call.mp3"
    first.write_bytes(b"")
    assert output_paths(first).text == tmp_path / "call.txt"
    (tmp_path / "call.txt").write_bytes(b"")                  # a non-media sibling does not count
    assert output_paths(first).subtitles == tmp_path / "call.srt"
    second = tmp_path / "call.wav"
    second.write_bytes(b"")
    assert output_paths(first).text == tmp_path / "call.mp3.txt"
    assert output_paths(second).subtitles == tmp_path / "call.wav.srt"
    assert output_paths(first).transcript != output_paths(second).transcript
    assert output_paths(tmp_path / "other.wav").review == tmp_path / "other.review.json"


def test_same_stem_recordings_keep_separate_outputs(models, tmp_path: Path) -> None:
    folder = tmp_path / "media"
    folder.mkdir()
    wav = audio.synthetic_wav(folder / "call.wav", 3.0)
    (folder / "call.mp4").write_bytes(b"not a video")
    result = process_file(job_for(wav, models, tmp_path), engines=make_engines())
    assert result.outputs.transcript == folder / "call.wav.transcript.json"
    assert result.document["source"]["outputs"] == "call.wav"
    assert "see call.wav.review.json" in result.outputs.text.read_text(encoding="utf-8")
    review = json.loads(result.outputs.review.read_text(encoding="utf-8"))
    assert review["audio"] == "call.wav"


def test_media_extension_rules() -> None:
    assert is_media("x.MP3") and is_media("y.mkv") and not is_media("z.txt") and not is_media("call.transcript.json")
    assert is_video("a.mp4") and is_video("b.AVI") and not is_video("c.wav")


def test_discover_media_walks_folders(tmp_path: Path) -> None:
    (tmp_path / "b.MP3").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    nested = tmp_path / "sub" / "deeper"
    nested.mkdir(parents=True)
    (nested / "a.wav").write_bytes(b"")
    (tmp_path / "sub" / "c.mp4").write_bytes(b"")
    found = discover_media([tmp_path, tmp_path / "b.MP3", tmp_path / "missing.wav"])
    assert [p.name for p in found] == ["b.MP3", "c.mp4", "a.wav"]
    flat = discover_media([tmp_path], recursive=False)
    assert [p.name for p in flat] == ["b.MP3"]
    assert discover_media([tmp_path / "notes.txt"]) == []


# ----- one file ------------------------------------------------------------------------------


def test_process_file_writes_six_outputs(recording: Path, models, tmp_path: Path) -> None:
    reports: list[Progress] = []
    calls: list[str] = []
    result = process_file(
        job_for(recording, models, tmp_path, author="Tester"),
        progress=reports.append,
        engines=make_engines(calls=calls),
    )
    assert calls == ["publisher", "detector", "diarizer"]
    for path in result.outputs.all:
        assert path.is_file(), path
        assert path.parent == recording.parent
    assert result.marks == 2 and result.duration_s == 30.0
    doc = json.loads(result.outputs.transcript.read_text(encoding="utf-8"))
    assert doc["source"]["name"] == "call.wav"
    assert doc["source"]["sha256"] == audio.sha256_of(recording)
    assert doc["source"]["bytes"] == recording.stat().st_size
    assert doc["profile"] == "standard" and doc["source"]["video"] is False
    assert [s["name"] for s in doc["speakers"]] == ["Speaker 1", "Speaker 2"]
    assert len(doc["overview"]["peaks"]) == 1200

    review = json.loads(result.outputs.review.read_text(encoding="utf-8"))
    assert review["schema"] == "twinscribe.review.v1"
    assert review["audio"] == "call.wav"                 # relative, beside the review set
    assert len(review["marks"]) == 2 and review["marks"][0]["detector_text"] == "yes I am here"

    record = json.loads(result.outputs.run.read_text(encoding="utf-8"))
    assert record["schema"] == "twinscribe.runrecord.v1"
    assert record["input_sha256"] == doc["source"]["sha256"]
    assert record["engines"]["publisher"]["model"] == "parakeet-tdt-0.6b-v2-int8"
    assert record["engines"]["detector"]["preset"] == "production"
    assert record["settings"]["profile"] == "standard" and record["settings"]["review"]["min_silence_s"] == 0.8
    assert record["settings"]["detector_backend"] == "ct2" and record["settings"]["detector_model"] == KEY_WHISPER_TURBO
    assert record["settings"]["plan"]["platform"] == "windows-x86_64" and record["settings"]["parallel_engines"] is False
    assert record["settings"]["detector_word_timing"] == "token"
    assert record["failures"] == []
    assert record["audio_s"] == 30.0 and record["transcribe_s"] == pytest.approx(1.5 + 1.5 + 0.9)
    assert set(record["versions"]) >= {"parakeet_tdt_lib", "whisper_ct2_lib", "sherpa_onnx"}

    text = result.outputs.text.read_text(encoding="utf-8")
    assert "[0:00.5] Speaker 1: good morning this is the first call" in text
    assert not list((tmp_path / "work").glob("*")) if (tmp_path / "work").exists() else True

    fractions = [r.fraction for r in reports]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0 and fractions[0] == 0.0
    stage_order = [name for name, _ in STAGES]
    seen = [r.stage for r in reports]
    assert [s for s in stage_order if s in seen] == [s for i, s in enumerate(seen) if s not in seen[:i]]
    assert "decode" not in seen                       # a compliant WAV is fed as it is


def test_detector_placement_follows_the_plan(recording: Path, models, tmp_path: Path) -> None:
    seen: dict[str, dict] = {}
    process_file(job_for(recording, models, tmp_path), engines=make_engines(kwargs_seen=seen))
    assert seen["detector"] == {"device": "cpu", "compute_type": "int8", "device_index": 0}
    assert seen["publisher"] == {"provider": "cpu"} and seen["diarizer"] == {"provider": "cpu"}

    gpu_plan = make_plan_for(cuda=True, sherpa_cuda=True)
    seen.clear()
    process_file(job_for(recording, models, tmp_path, plan=gpu_plan), engines=make_engines(kwargs_seen=seen))
    assert seen["detector"] == {"device": "cuda", "compute_type": "float16", "device_index": 0}
    assert seen["publisher"] == {"provider": "cuda"} and seen["diarizer"] == {"provider": "cuda"}


def test_onnx_detector_is_dispatched_without_ctranslate2(recording: Path, models, tmp_path: Path) -> None:
    calls: list[str] = []
    seen: dict[str, dict] = {}
    result = process_file(
        job_for(recording, models, tmp_path, plan=make_plan_for(ct2=False)),
        engines=make_engines(calls=calls, kwargs_seen=seen),
    )
    assert calls == ["publisher", "detector_onnx", "diarizer"]
    assert seen["detector_onnx"] == {"provider": "cpu"}
    assert result.run_record["settings"]["detector_backend"] == "onnx"
    assert result.run_record["settings"]["detector_model"] == KEY_WHISPER_TURBO_ONNX
    assert result.run_record["settings"]["detector_word_timing"] == "segment"
    assert "greedy" in result.run_record["settings"]["detector_preset"]
    facts = result.document["engines"]["detector"]
    assert facts["engine"] == "whisper_onnx" and facts["word_timing"] == "segment"
    text = result.outputs.text.read_text(encoding="utf-8")
    assert "word times are approximate" in text
    assert result.marks == 2


def test_explicit_detector_key_overrides_the_selection(recording: Path, models, tmp_path: Path) -> None:
    calls: list[str] = []
    result = process_file(
        job_for(recording, models, tmp_path, detector=KEY_WHISPER_TURBO_ONNX),
        engines=make_engines(calls=calls),
    )
    assert calls[1] == "detector_onnx" and result.run_record["settings"]["detector_model"] == KEY_WHISPER_TURBO_ONNX


def test_engines_run_in_parallel_when_the_plan_separates_them(recording: Path, models, tmp_path: Path) -> None:
    reports: list[Progress] = []
    calls: list[str] = []
    result = process_file(
        job_for(recording, models, tmp_path, plan=make_plan_for(cuda=True)),
        progress=reports.append,
        engines=make_engines(calls=calls),
    )
    assert set(calls) == {"publisher", "detector", "diarizer"}
    assert result.run_record["settings"]["parallel_engines"] is True
    fractions = [r.fraction for r in reports]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert any("at the same time" in r.message for r in reports)
    assert result.marks == 2 and result.outputs.transcript.is_file()


def test_missing_backend_library_is_a_clear_error(recording: Path, models, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no library for the onnx detector backend"):
        process_file(
            job_for(recording, models, tmp_path, plan=make_plan_for(sherpa=False), detector=KEY_WHISPER_TURBO_ONNX),
            engines=make_engines(),
        )


def test_process_file_into_another_folder_uses_absolute_audio(recording: Path, models, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = process_file(job_for(recording, models, tmp_path, out_dir=out), engines=make_engines())
    assert result.outputs.transcript.parent == out
    review = json.loads(result.outputs.review.read_text(encoding="utf-8"))
    assert Path(review["audio"]).is_absolute() and Path(review["audio"]).name == "call.wav"


def test_speaker_failure_keeps_the_transcript(recording: Path, models, tmp_path: Path) -> None:
    result = process_file(job_for(recording, models, tmp_path), engines=make_engines(fail_diarizer=True))
    doc = result.document
    assert doc["speaker_failure"] == "RuntimeError: embedding model rejected"
    assert [s["label"] for s in doc["speakers"]] == [None]
    assert len(doc["lines"]) >= 1
    record = result.run_record
    assert len(record["failures"]) == 1 and "speaker labelling" in record["failures"][0]["message"]
    assert "Speaker labelling did not complete" in result.outputs.text.read_text(encoding="utf-8")


def test_no_diarizer_at_all(recording: Path, models, tmp_path: Path) -> None:
    result = process_file(job_for(recording, models, tmp_path), engines=make_engines(no_diarizer=True))
    assert result.document["engines"]["diarization"] is None
    assert result.run_record["failures"] == []


def test_engine_failure_leaves_a_run_record(recording: Path, models, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="publisher exploded"):
        process_file(job_for(recording, models, tmp_path), engines=make_engines(fail_publisher=True))
    paths = output_paths(recording)
    assert paths.run.is_file() and not paths.transcript.exists()
    record = json.loads(paths.run.read_text(encoding="utf-8"))
    assert record["failures"][0]["error_class"] == "RuntimeError"
    assert record["input_sha256"] == audio.sha256_of(recording)


def test_cancellation_stops_early_without_outputs(recording: Path, models, tmp_path: Path) -> None:
    seen: list[str] = []

    def cancel() -> bool:
        return len(seen) >= 3

    def progress(report: Progress) -> None:
        seen.append(report.stage)

    with pytest.raises(Cancelled):
        process_file(job_for(recording, models, tmp_path), progress=progress, cancel=cancel, engines=make_engines())
    assert not output_paths(recording).transcript.exists()
    assert not output_paths(recording).run.exists()


def test_missing_recording(models, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        process_file(job_for(tmp_path / "absent.wav", models, tmp_path), engines=make_engines())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on the search path")
def test_non_compliant_audio_is_decoded_and_the_work_file_removed(models, tmp_path: Path) -> None:
    folder = tmp_path / "media"
    folder.mkdir()
    source = folder / "narrow.wav"
    tone = (0.3 * np.sin(np.linspace(0.0, 2000.0, 8000 * 5))).astype(np.float32)
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes((tone * 32767).astype("<i2").tobytes())
    reports: list[Progress] = []
    result = process_file(job_for(source, models, tmp_path), progress=reports.append, engines=make_engines())
    assert "decode" in {r.stage for r in reports}
    assert result.outputs.transcript.is_file()
    assert list((tmp_path / "work").glob("*.wav")) == []
    kept = process_file(job_for(source, models, tmp_path, keep_audio=True), engines=make_engines())
    assert kept.outputs.transcript.is_file()
    assert len(list((tmp_path / "work").glob("*.16k.wav"))) == 1


# ----- batch ---------------------------------------------------------------------------------


def test_batch_records_every_outcome(recording: Path, models, tmp_path: Path) -> None:
    broken = recording.parent / "broken.mp3"
    broken.write_bytes(b"not audio at all")
    seen: list[tuple[int, bool]] = []
    result = run_batch(
        [recording, broken],
        profile_for("standard"),
        models,
        engines=make_engines(),
        record_dir=tmp_path / "runs",
        on_outcome=lambda index, outcome: seen.append((index, outcome.ok)),
        plan=make_plan_for(),
    )
    assert seen == [(0, True), (1, False)]
    assert len(result.completed) == 1 and len(result.failures) == 1
    assert result.failures[0].source == broken and result.failures[0].error_class
    assert result.record_path is not None and result.record_path.parent == tmp_path / "runs"
    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert record["schema"] == BATCH_SCHEMA and record["profile"] == "standard"
    assert record["plan"]["detector_backend"] == "ct2"
    assert [f["ok"] for f in record["files"]] == [True, False]
    assert record["failures"][0]["path"] == str(broken)
    assert "outputs" in record["files"][0] and record["files"][0]["marks"] == 2


def test_batch_cancellation_marks_the_rest(recording: Path, models, tmp_path: Path) -> None:
    second = audio.synthetic_wav(recording.parent / "second.wav", 5.0)
    third = audio.synthetic_wav(recording.parent / "third.wav", 5.0)
    count = {"reports": 0}

    def progress(index: int, total: int, report: Progress) -> None:
        count["reports"] += 1

    def cancel() -> bool:
        return count["reports"] > 4

    result = run_batch([recording, second, third], profile_for("standard"), models, engines=make_engines(),
                       progress=progress, cancel=cancel, record_dir=tmp_path / "runs", plan=make_plan_for())
    assert [o.cancelled for o in result.outcomes] == [True, True, True]
    assert result.failures == [] and result.completed == []
    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert all(f.get("cancelled") for f in record["files"])
