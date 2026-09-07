"""Tests for the model catalogue, the store reader and the lock verification."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from twinscribe import models
from twinscribe.models import (
    CATALOGUE,
    ROLES,
    STATUS_MISMATCH,
    STATUS_MISSING,
    STATUS_UNPINNED,
    STATUS_VERIFIED,
    find_models,
    lock_entry,
    read_lock,
    spec_for,
    specs_for_role,
    verify_store,
    write_lock,
)


def test_catalogue_is_consistent() -> None:
    keys = [spec.key for spec in CATALOGUE]
    assert len(keys) == len(set(keys))
    for spec in CATALOGUE:
        assert spec.role in ROLES
        assert spec.required and spec.sources
        assert spec.licence and spec.credit and spec.title
        targets = {source.target for source in spec.sources}
        assert set(spec.required) <= targets, spec.key
        for source in spec.sources:
            assert source.url.startswith("https://"), source.url
            if source.archive is not None:
                assert source.url.endswith(source.archive)


def test_every_role_has_a_model() -> None:
    for role in ROLES:
        assert specs_for_role(role)
    with pytest.raises(KeyError):
        specs_for_role("conductor")


def test_spec_for_unknown_key_lists_known() -> None:
    with pytest.raises(KeyError) as excinfo:
        spec_for("nothing")
    assert "parakeet-tdt-0.6b-v2-int8" in str(excinfo.value)


def test_find_models_on_empty_root(tmp_path: Path) -> None:
    store = find_models(tmp_path)
    assert store.root == tmp_path
    assert not store.present
    assert set(store.missing) == {spec.key for spec in CATALOGUE}
    with pytest.raises(KeyError) as excinfo:
        store.path("silero-vad")
    assert str(tmp_path / "silero-vad") in excinfo.value.args[0]
    assert "Silero voice activity detector" in excinfo.value.args[0]


def test_find_models_present_and_incomplete(tmp_path: Path) -> None:
    complete = tmp_path / "silero-vad"
    complete.mkdir()
    (complete / "silero_vad.onnx").write_bytes(b"x")
    partial = tmp_path / "parakeet-tdt-0.6b-v2-int8"
    partial.mkdir()
    (partial / "encoder.int8.onnx").write_bytes(b"x")
    store = find_models(tmp_path)
    assert store.has("silero-vad") and store.path("silero-vad") == complete
    assert store.file("silero-vad", "silero_vad.onnx") == complete / "silero_vad.onnx"
    assert not store.has("parakeet-tdt-0.6b-v2-int8")
    assert set(store.incomplete["parakeet-tdt-0.6b-v2-int8"]) == {"decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"}
    described = store.describe()
    assert "silero-vad" in described and "present" in described
    assert "incomplete, lacks" in described and "tokens.txt" in described


def test_default_root_follows_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(models.MODELS_ENV, str(tmp_path / "elsewhere"))
    assert models.default_models_root() == tmp_path / "elsewhere"
    assert find_models().root == tmp_path / "elsewhere"


def test_lock_round_trip_and_verification(tmp_path: Path) -> None:
    spec = spec_for("silero-vad")
    folder = tmp_path / spec.key
    folder.mkdir()
    payload = b"model bytes" * 1000
    target = folder / "silero_vad.onnx"
    target.write_bytes(payload)

    assert read_lock(tmp_path) == {}
    checks = verify_store(tmp_path, keys=("silero-vad",))
    assert [c.status for c in checks] == [STATUS_UNPINNED]

    entry = lock_entry(spec, spec.sources[0], target, revision="v1")
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["bytes"] == len(payload)
    assert entry["url"] == spec.sources[0].url and entry["licence"] == "MIT"
    lock_file = write_lock(tmp_path, {f"{spec.key}/silero_vad.onnx": entry})
    assert lock_file.is_file()
    assert read_lock(tmp_path)[f"{spec.key}/silero_vad.onnx"]["sha256"] == entry["sha256"]

    checks = verify_store(tmp_path, keys=("silero-vad",))
    assert [c.status for c in checks] == [STATUS_VERIFIED]

    target.write_bytes(payload + b"!")
    checks = verify_store(tmp_path, keys=("silero-vad",))
    assert [c.status for c in checks] == [STATUS_MISMATCH]

    target.unlink()
    checks = verify_store(tmp_path, keys=("silero-vad",))
    assert [c.status for c in checks] == [STATUS_MISSING]


def test_unreadable_lock_is_empty(tmp_path: Path) -> None:
    (tmp_path / models.LOCK_FILE).write_text("not json", encoding="utf-8")
    assert read_lock(tmp_path) == {}
    (tmp_path / models.LOCK_FILE).write_text('{"schema": "other", "entries": {}}', encoding="utf-8")
    assert read_lock(tmp_path) == {}
