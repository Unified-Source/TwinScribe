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


def test_digest_reports_progress(tmp_path: Path) -> None:
    import hashlib
    import os

    payload = os.urandom(3 * (1 << 20) + 17)
    target = tmp_path / "big.bin"
    target.write_bytes(payload)
    seen: list[float] = []
    assert audio.sha256_of(target, progress=seen.append) == hashlib.sha256(payload).hexdigest()
    assert seen and seen[-1] == 1.0 and seen == sorted(seen) and all(0.0 <= f <= 1.0 for f in seen)
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    seen = []
    assert audio.sha256_of(empty, progress=seen.append) == hashlib.sha256(b"").hexdigest()
    assert seen == [1.0]


def test_human_size() -> None:
    from twinscribe.pipeline import human_size

    assert human_size(0) == "0 bytes" and human_size(850) == "850 bytes"
    assert human_size(850_000) == "850 KB" and human_size(12_400_000) == "12.4 MB"
    assert human_size(1_200_000_000) == "1.2 GB" and human_size(3 * 10**12) == "3 TB"


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
    wav = audio.synthetic_wav(folder / "call.wav", 30.0)      # as long as the fixture transcript
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
    assert calls == ["publisher", "detector", "tagger", "diarizer"]
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
    assert record["audio_s"] == 30.0 and record["transcribe_s"] == pytest.approx(1.5 + 1.5 + 0.9 + 0.1)
    assert set(record["versions"]) >= {"parakeet_tdt_lib", "whisper_ct2_lib", "sherpa_onnx", "tagger_lib"}
    assert record["engines"]["tagging"]["engine"] == "fake_tagging"
    scene_facts = record["settings"]["scenes"]
    # Five pause windows (the ten-second pause is cut in two) and four utterances were tagged.
    assert scene_facts["count"] == 0 and scene_facts["tagged"] is True and scene_facts["regions_tagged"] == 9
    assert scene_facts["suppressed_publisher_words"] == 0 and scene_facts["suppressed_detector_words"] == 0
    assert doc["scenes"] == [] and doc["non_speech"]["tagged"] is True

    text = result.outputs.text.read_text(encoding="utf-8")
    assert "[0:00.5] Speaker 1: good morning this is the first call" in text
    assert not list((tmp_path / "work").glob("*")) if (tmp_path / "work").exists() else True

    fractions = [r.fraction for r in reports]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0 and fractions[0] == 0.0
    stage_order = [name for name, _ in STAGES]
    seen = [r.stage for r in reports]
    assert [s for s in stage_order if s in seen] == [s for i, s in enumerate(seen) if s not in seen[:i]]
    assert "decode" not in seen                       # a compliant WAV is fed as it is


def test_progress_carries_time_and_loading_messages_and_partials(recording: Path, models, tmp_path: Path) -> None:
    reports: list[Progress] = []
    partials: list[tuple[str, str]] = []
    process_file(
        job_for(recording, models, tmp_path),
        progress=reports.append,
        engines=make_engines(),
        on_partial=lambda role, segment: partials.append((role, segment.text)),
    )
    messages = [r.message for r in reports]
    assert "Loading the published engine" in messages and "Transcribing (published engine)" in messages
    assert messages.index("Loading the published engine") < messages.index("Transcribing (published engine)")
    assert "Loading the second engine" in messages and "Loading the speaker models and labelling speakers" in messages
    assert all(r.elapsed_s >= 0.0 for r in reports) and reports[-1].elapsed_s >= reports[0].elapsed_s
    assert reports[0].eta_s is None
    assert any(r.eta_s is not None and r.eta_s >= 0.0 for r in reports)
    assert partials and all(role == "publisher" for role, _ in partials)
    assert partials[0][1].startswith("good morning")


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


def test_speaker_count_is_an_explicit_opt_in(recording: Path, models, tmp_path: Path) -> None:
    seen: dict[str, dict] = {}
    result = process_file(job_for(recording, models, tmp_path), engines=make_engines(kwargs_seen=seen))
    assert "num_speakers" not in seen["diarizer"]
    assert result.run_record["settings"]["speakers"] is None
    assert result.document["engines"]["diarization"]["settings"]["num_speakers"] is None
    assert "clustering threshold 0.5" in result.outputs.text.read_text(encoding="utf-8")
    assert result.run_record["settings"]["labelling"] == {"smoothed_words": 0, "min_run_words": 2, "min_run_s": 0.6}

    seen.clear()
    fixed = process_file(job_for(recording, models, tmp_path, speakers=2), engines=make_engines(kwargs_seen=seen))
    assert seen["diarizer"]["num_speakers"] == 2
    assert fixed.run_record["settings"]["speakers"] == 2
    assert fixed.document["engines"]["diarization"]["settings"]["num_speakers"] == 2
    assert "speaker count fixed at 2" in fixed.outputs.text.read_text(encoding="utf-8")


def test_flicker_inside_an_utterance_is_smoothed(recording: Path, models, tmp_path: Path) -> None:
    from tests._fixtures import TURNS

    # A turn boundary jitters into the first utterance: its third word (1.3 to 1.8) falls in a
    # sliver of the other speaker while the words on both sides stay with the first.
    jittered = [(0.0, 1.3, "speaker_00"), (1.3, 1.8, "speaker_01"), (1.8, 3.6, "speaker_00")] + TURNS[1:]
    from tests import _fixtures

    original = _fixtures.diarization

    def flickering(threshold=0.5, num_speakers=None):
        return original(jittered, threshold=threshold, num_speakers=num_speakers)

    engines = make_engines()
    from dataclasses import replace

    def diarizer(path, segmentation, embedding, threads=None, threshold=0.5, **kwargs):
        return flickering(threshold=threshold)

    result = process_file(job_for(recording, models, tmp_path), engines=replace(engines, diarizer=diarizer))
    assert result.run_record["settings"]["labelling"]["smoothed_words"] == 1
    first = result.document["lines"][0]
    assert first["speaker"] == "speaker_00" and first["text"] == "good morning this is the first call"


def test_onnx_detector_is_dispatched_without_ctranslate2(recording: Path, models, tmp_path: Path) -> None:
    calls: list[str] = []
    seen: dict[str, dict] = {}
    result = process_file(
        job_for(recording, models, tmp_path, plan=make_plan_for(ct2=False)),
        engines=make_engines(calls=calls, kwargs_seen=seen),
    )
    assert calls == ["publisher", "detector_onnx", "tagger", "diarizer"]
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
    assert set(calls) == {"publisher", "detector", "tagger", "diarizer"}
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


def test_scenes_mark_non_speech_and_set_words_aside(recording: Path, models, tmp_path: Path) -> None:
    from twinscribe.scenes import Event

    # The fixture's third pause (13.9 to 24.0) is heard as music; everything else is speech.
    def events(region: tuple[float, float]) -> list[Event]:
        start, end = region
        if start >= 13.8 and end <= 24.1:
            return [Event("Music", 0.8), Event("Synthesizer", 0.4), Event("Speech", 0.03)]
        return [Event("Speech", 0.9)]

    result = process_file(job_for(recording, models, tmp_path), engines=make_engines(tagger_events=events))
    doc = result.document
    assert [(round(s["start"], 1), round(s["end"], 1), s["kind"]) for s in doc["scenes"]] == [(13.9, 24.0, "music")]
    # The detector's two words inside the music no longer raise a mark; the first pause still does.
    assert result.marks == 1 and doc["marks"][0]["span_start"] == 3.5
    assert doc["non_speech"]["suppressed_detector_words"] == 2 and doc["non_speech"]["suppressed_publisher_words"] == 0
    assert doc["non_speech"]["seconds_by_kind"] == {"music": pytest.approx(10.1)}
    text = result.outputs.text.read_text(encoding="utf-8")
    assert "[0:13.9] (music, no speech, 10 s)" in text and "Without speech: music 10 s" in text
    assert "2 words the second engine placed inside silence, music or noise were left out of the review list" in text
    srt = result.outputs.subtitles.read_text(encoding="utf-8")
    assert "00:00:13,900 --> 00:00:24,000\n[music]" in srt
    record = result.run_record["settings"]["scenes"]
    assert record["count"] == 1 and record["list"][0]["kind"] == "music" and record["suppressed_utterances"] == []


def test_an_utterance_without_speech_is_set_aside_and_recorded(recording: Path, models, tmp_path: Path) -> None:
    from twinscribe.scenes import Event

    # The publisher's first utterance (0.5 to 3.5, seven words) is heard as music with no speech.
    def events(region: tuple[float, float]) -> list[Event]:
        start, end = region
        if abs(start - 0.5) < 0.01 and abs(end - 3.5) < 0.01:
            return [Event("Music", 0.9), Event("Speech", 0.02)]
        return [Event("Speech", 0.9)]

    result = process_file(job_for(recording, models, tmp_path), engines=make_engines(tagger_events=events))
    doc = result.document
    assert [s["kind"] for s in doc["scenes"]] == ["music"] and doc["scenes"][0]["start"] == 0.5
    assert doc["non_speech"]["suppressed_publisher_words"] == 7
    assert sum(s["words"] for s in doc["speakers"]) == 11
    assert "good morning" not in result.outputs.text.read_text(encoding="utf-8")
    assert "7 words the published engine wrote inside music or noise were set aside" in result.outputs.text.read_text(encoding="utf-8")
    suppressed = result.run_record["settings"]["scenes"]["suppressed_utterances"]
    assert len(suppressed) == 1 and suppressed[0]["text"].startswith("good morning") and suppressed[0]["words"] == 7
    review = json.loads(result.outputs.review.read_text(encoding="utf-8"))
    assert len(review["transcript"]) == 11


def test_without_a_tagging_model_the_level_alone_decides(recording: Path, models, tmp_path: Path) -> None:
    from twinscribe.models import KEY_AUDIO_TAGGER, find_models

    for child in (models.root / KEY_AUDIO_TAGGER).iterdir():
        child.unlink()
    (models.root / KEY_AUDIO_TAGGER).rmdir()
    store = find_models(models.root)
    assert not store.has(KEY_AUDIO_TAGGER)
    calls: list[str] = []
    result = process_file(job_for(recording, store, tmp_path), engines=make_engines(calls=calls))
    assert "tagger" not in calls
    doc = result.document
    # The synthetic recording is a loud tone throughout, so every pause of two seconds or more
    # is sound of an unknown kind, and the detector words inside them raise no marks.
    assert doc["non_speech"]["tagged"] is False
    assert [s["kind"] for s in doc["scenes"]] == ["sound"] * 4
    assert result.marks == 0
    assert "sound, no speech" in result.outputs.text.read_text(encoding="utf-8")
    assert result.run_record["settings"]["scenes"]["tagger_model"] is None
    assert "tagging" not in result.run_record["engines"]


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
    partials: list[tuple[int, str]] = []
    result = run_batch(
        [recording, broken],
        profile_for("standard"),
        models,
        engines=make_engines(),
        record_dir=tmp_path / "runs",
        on_outcome=lambda index, outcome: seen.append((index, outcome.ok)),
        plan=make_plan_for(),
        on_partial=lambda index, role, segment: partials.append((index, role)),
    )
    assert seen == [(0, True), (1, False)]
    assert partials and all(entry == (0, "publisher") for entry in partials)
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
