"""Tests for the run record, the machine facts and the atomic JSON writer."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from twinscribe.load import LoadVerdict
from twinscribe.runrecord import (
    RUNRECORD_SCHEMA,
    Failure,
    RunRecord,
    input_digest,
    machine_facts,
    utc_now,
    write_json_atomic,
)

MACHINE = {
    "processor": "test-cpu",
    "logical_cores": 4,
    "memory_gb": 8.0,
    "os": "test-os 1 (1.0)",
    "architecture": "test",
    "python": "3.11.0",
}


def _record(digest: str, **overrides) -> RunRecord:
    fields = dict(
        engines={
            "publisher": {"engine": "transducer", "model": "model-p", "preset": "vad"},
            "detector": {"engine": "whisper", "model": "model-d", "preset": "production"},
        },
        versions={"publisher-lib": "1.13.0", "detector-lib": "1.2.0"},
        settings={"min_silence_s": 0.8, "min_detector_words": 2, "pad_s": 0.4, "beam": 5},
        input_path="sample.wav",
        input_sha256=digest,
        audio_s=120.0,
        load_s=2.5,
        transcribe_s=30.0,
        started_utc="2026-01-01T00:00:00+00:00",
        ended_utc="2026-01-01T00:00:33+00:00",
        load_verdict=LoadVerdict(other_load_cores=0.1, busy=False, on_mains_power=True, threshold_cores=0.5),
        failures=(
            ("bad.wav", "ValueError", "unsupported sample rate"),
            Failure("worse.wav", "OSError", "cannot open"),
        ),
        machine=dict(MACHINE),
    )
    fields.update(overrides)
    return RunRecord(**fields)


# --------------------------------------------------------------- machine facts


def test_machine_facts_shape_and_no_hostname_by_default():
    facts = machine_facts()
    assert set(facts) == {"processor", "logical_cores", "memory_gb", "os", "architecture", "python"}
    assert "hostname" not in facts
    assert facts["logical_cores"] is None or (isinstance(facts["logical_cores"], int) and facts["logical_cores"] > 0)
    assert facts["memory_gb"] is None or facts["memory_gb"] > 0.0
    assert facts["processor"] is None or isinstance(facts["processor"], str)
    assert isinstance(facts["os"], str) and facts["os"]
    assert isinstance(facts["python"], str) and "." in facts["python"]


def test_machine_facts_hostname_only_on_request():
    facts = machine_facts(include_hostname=True)
    assert "hostname" in facts
    assert facts["hostname"] is None or isinstance(facts["hostname"], str)


# --------------------------------------------------------------------- digest


def test_input_digest_matches_hashlib_over_multiple_chunks(tmp_path):
    payload = bytes(range(256)) * 6000
    target = tmp_path / "input.bin"
    target.write_bytes(payload)
    assert input_digest(target) == hashlib.sha256(payload).hexdigest()
    assert input_digest(str(target)) == hashlib.sha256(payload).hexdigest()
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert input_digest(empty) == hashlib.sha256(b"").hexdigest()


def test_input_digest_matches_audio_module_digest(tmp_path):
    audio = pytest.importorskip("twinscribe.audio", reason="audio module not present")
    sha256_of = getattr(audio, "sha256_of", None)
    if sha256_of is None:
        pytest.skip("audio module has no sha256_of")
    target = tmp_path / "input.bin"
    target.write_bytes(b"twinscribe" * 1000)
    assert input_digest(target) == sha256_of(str(target))


# --------------------------------------------------------------------- record


def test_failures_normalise_to_failure_records():
    record = _record("00" * 32)
    assert all(isinstance(failure, Failure) for failure in record.failures)
    assert record.failures[0] == Failure("bad.wav", "ValueError", "unsupported sample rate")
    assert record.failures[1] == Failure("worse.wav", "OSError", "cannot open")
    from_exception = Failure.from_exception("x.wav", ValueError("boom"))
    assert from_exception == Failure("x.wav", "ValueError", "boom")


def test_real_time_factor():
    assert _record("00" * 32).real_time_factor == pytest.approx(0.25)
    assert _record("00" * 32, audio_s=0.0).real_time_factor is None
    assert _record("00" * 32, audio_s=0.0).to_dict()["real_time_factor"] is None


def test_to_dict_shape():
    doc = _record("ab" * 32).to_dict()
    assert doc["schema"] == RUNRECORD_SCHEMA
    assert doc["engines"]["publisher"]["preset"] == "vad"
    assert doc["versions"] == {"publisher-lib": "1.13.0", "detector-lib": "1.2.0"}
    assert doc["input_path"] == "sample.wav"
    assert doc["input_sha256"] == "ab" * 32
    assert (doc["audio_s"], doc["load_s"], doc["transcribe_s"]) == (120.0, 2.5, 30.0)
    assert doc["real_time_factor"] == pytest.approx(0.25)
    assert doc["started_utc"] == "2026-01-01T00:00:00+00:00"
    assert doc["ended_utc"] == "2026-01-01T00:00:33+00:00"
    assert doc["load_verdict"] == {
        "other_load_cores": 0.1,
        "busy": False,
        "on_mains_power": True,
        "threshold_cores": 0.5,
    }
    assert doc["failures"] == [
        {"path": "bad.wav", "error_class": "ValueError", "message": "unsupported sample rate"},
        {"path": "worse.wav", "error_class": "OSError", "message": "cannot open"},
    ]
    assert doc["machine"] == MACHINE
    assert "hostname" not in doc["machine"]


def test_default_machine_facts_carry_no_hostname():
    record = _record("00" * 32, machine=machine_facts())
    assert "hostname" not in record.to_dict()["machine"]
    defaulted = RunRecord(
        engines={},
        versions={},
        settings={},
        input_path="sample.wav",
        input_sha256="00" * 32,
        audio_s=1.0,
        load_s=0.0,
        transcribe_s=0.0,
        started_utc=utc_now(),
        ended_utc=utc_now(),
    )
    assert set(defaulted.machine) == set(machine_facts())
    assert defaulted.load_verdict is None
    assert defaulted.to_dict()["load_verdict"] is None
    assert defaulted.failures == ()


def test_write_then_read_back_matches_and_digest_is_reused(tmp_path):
    payload = b"\x00\x01\x02" * 4096
    source = tmp_path / "sample.wav"
    source.write_bytes(payload)
    digest = input_digest(source)
    assert digest == hashlib.sha256(payload).hexdigest()
    record = _record(digest)
    target = tmp_path / "run.json"
    record.write(target)
    with open(target, encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded == record.to_dict()
    assert loaded["input_sha256"] == digest
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json", "sample.wav"]


# --------------------------------------------------------------- json writer


def test_write_json_atomic_overwrites_and_leaves_no_temporary_files(tmp_path):
    target = tmp_path / "nested" / "doc.json"
    write_json_atomic({"a": 1}, target)
    write_json_atomic({"a": 2}, target)
    with open(target, encoding="utf-8") as handle:
        assert json.load(handle) == {"a": 2}
    assert sorted(p.name for p in (tmp_path / "nested").iterdir()) == ["doc.json"]


def test_write_json_atomic_coerces_numpy_and_tuples(tmp_path):
    doc = {"threshold": np.float32(0.5), "count": np.int64(3), "shape": (1, 2), "labels": {"x"}}
    target = tmp_path / "doc.json"
    write_json_atomic(doc, target)
    with open(target, encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded == {"threshold": 0.5, "count": 3, "shape": [1, 2], "labels": ["x"]}


def test_write_json_atomic_rejects_unserialisable_values(tmp_path):
    with pytest.raises(TypeError):
        write_json_atomic({"bad": object()}, tmp_path / "doc.json")
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------- time


def test_utc_now_is_iso_8601_in_utc():
    stamp = utc_now()
    parsed = datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0
    assert abs((datetime.now(timezone.utc) - parsed).total_seconds()) < 60
