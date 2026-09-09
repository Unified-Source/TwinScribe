"""Tests for the fetch module, against local file sources: plain files and archive members,
the lock, skipping what is present and pinned, progress reports by stage, cancellation leaving
no half file, the level plans, and the proposed root for a first fetch."""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import pytest

from twinscribe import fetch
from twinscribe.fetch import (
    STAGE_DONE,
    STAGE_DOWNLOAD,
    STAGE_EXTRACT,
    Cancelled,
    Progress,
    extract_members,
    fetch_specs,
    missing_for_levels,
    missing_sources,
    proposed_root,
    specs_for_level,
)
from twinscribe.models import (
    KEY_AUDIO_TAGGER,
    KEY_EMBEDDING,
    KEY_PARAKEET_V2,
    KEY_SEGMENTATION,
    KEY_SILERO_VAD,
    KEY_WHISPER_LARGE,
    KEY_WHISPER_TURBO,
    ModelSpec,
    Source,
    find_models,
    read_lock,
    sha256_file,
)


def _file_url(path: Path) -> str:
    return path.resolve().as_uri()


@pytest.fixture
def upstream(tmp_path: Path) -> dict[str, Path]:
    """A plain file and an archive holding three members under a top folder, as upstream serves them."""
    up = tmp_path / "upstream"
    up.mkdir()
    plain = up / "model.bin"
    plain.write_bytes(b"weights " * 1000)
    inner = up / "pack" / "top"
    inner.mkdir(parents=True)
    (inner / "encoder.onnx").write_bytes(b"encoder " * 2000)
    (inner / "decoder.onnx").write_bytes(b"decoder " * 500)
    (inner / "tokens.txt").write_text("a\nb\n", encoding="utf-8")
    (inner / "README").write_text("not wanted\n", encoding="utf-8")
    archive = up / "pack.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        tar.add(inner, arcname="top")
    return {"plain": plain, "archive": archive}


def _specs(upstream: dict[str, Path]) -> list[ModelSpec]:
    plain = ModelSpec(
        key="plain-model", role="detector", title="A plain model", licence="MIT", credit="tests",
        required=("model.bin",), sources=(Source(url=_file_url(upstream["plain"]), target="model.bin"),), backend="ct2", size_mb=1,
    )
    archive_name = upstream["archive"].name
    url = _file_url(upstream["archive"])
    packed = ModelSpec(
        key="packed-model", role="publisher", title="A packed model", licence="CC BY 4.0", credit="tests",
        required=("encoder.onnx", "decoder.onnx", "tokens.txt"),
        sources=(
            Source(url=url, target="encoder.onnx", archive=archive_name),
            Source(url=url, target="decoder.onnx", archive=archive_name),
            Source(url=url, target="tokens.txt", archive=archive_name),
        ),
        size_mb=1,
    )
    return [plain, packed]


def test_fetch_places_files_records_the_lock_and_reports_by_stage(upstream: dict[str, Path], tmp_path: Path) -> None:
    root = tmp_path / "store"
    seen: list[Progress] = []
    lines: list[str] = []
    lock = fetch_specs(_specs(upstream), root, progress=seen.append, log=lines.append)
    assert (root / "plain-model" / "model.bin").read_bytes() == upstream["plain"].read_bytes()
    assert (root / "packed-model" / "encoder.onnx").stat().st_size == 8 * 2000
    assert (root / "packed-model" / "decoder.onnx").stat().st_size == 8 * 500
    assert (root / "packed-model" / "tokens.txt").read_text(encoding="utf-8") == "a\nb\n"
    assert not (root / "packed-model" / "README").exists()
    assert set(lock) == {"plain-model/model.bin", "packed-model/encoder.onnx", "packed-model/decoder.onnx", "packed-model/tokens.txt"}
    assert lock["plain-model/model.bin"]["sha256"] == sha256_file(root / "plain-model" / "model.bin")
    assert lock["packed-model/tokens.txt"]["archive"] == upstream["archive"].name
    assert read_lock(root) == lock
    # Four files: the plain one downloaded, the archive downloaded once under its own name,
    # extraction reported as a stage, and every placed file reported as done in order.
    done = [p for p in seen if p.stage == STAGE_DONE]
    assert done[0].file == "model.bin" and [p.files_done for p in done] == [1, 2, 3, 4]
    assert sorted(p.file for p in done[1:]) == ["decoder.onnx", "encoder.onnx", "tokens.txt"]
    assert all(p.files_total == 4 and p.fraction is None for p in done)
    downloads = [p for p in seen if p.stage == STAGE_DOWNLOAD]
    assert {p.file for p in downloads} == {"model.bin", upstream["archive"].name}
    assert all(p.fraction is not None and p.fraction <= 1.0 for p in downloads)
    extracting = [p for p in seen if p.stage == STAGE_EXTRACT]
    assert len(extracting) == 1 and extracting[0].file == upstream["archive"].name and extracting[0].files_done == 1
    assert sum(1 for line in lines if "downloading" in line) == 2 and any("extracting 3 file(s)" in line for line in lines)
    assert not list(root.glob("twinscribe-fetch-*")) and not list(root.rglob("*.part"))


def test_present_and_pinned_files_are_skipped(upstream: dict[str, Path], tmp_path: Path) -> None:
    root = tmp_path / "store"
    specs = _specs(upstream)
    fetch_specs(specs, root)
    stamp = (root / "plain-model" / "model.bin").stat().st_mtime_ns
    assert missing_sources(specs[0], root, read_lock(root)) == []
    seen: list[Progress] = []
    fetch_specs(specs, root, progress=seen.append)
    assert seen == [] and (root / "plain-model" / "model.bin").stat().st_mtime_ns == stamp
    # A changed file is fetched again; a changed archive member alone means the archive again, one member out.
    (root / "plain-model" / "model.bin").write_bytes(b"corrupt")
    (root / "packed-model" / "decoder.onnx").write_bytes(b"corrupt")
    assert [s.target for s in missing_sources(specs[0], root, read_lock(root))] == ["model.bin"]
    assert [s.target for s in missing_sources(specs[1], root, read_lock(root))] == ["decoder.onnx"]
    seen = []
    fetch_specs(specs, root, progress=seen.append)
    assert (root / "plain-model" / "model.bin").read_bytes() == upstream["plain"].read_bytes()
    assert (root / "packed-model" / "decoder.onnx").stat().st_size == 8 * 500
    assert [(p.file, p.files_done) for p in seen if p.stage == STAGE_DONE] == [("model.bin", 1), ("decoder.onnx", 2)]
    assert [p.file for p in seen if p.stage == STAGE_DOWNLOAD and p.files_done == 1] == [upstream["archive"].name] * sum(1 for p in seen if p.stage == STAGE_DOWNLOAD and p.files_done == 1)


def test_extract_members_reads_the_archive_once_and_names_what_is_missing(upstream: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "out"
    order: list[str] = []
    extract_members(upstream["archive"], {"tokens.txt": out / "t.txt", "encoder.onnx": out / "e.onnx"}, order.append)
    # Members come in archive order, whatever the order asked for; the unwanted ones are skipped.
    assert sorted(order) == ["encoder.onnx", "tokens.txt"]
    assert (out / "e.onnx").stat().st_size == 8 * 2000 and (out / "t.txt").read_text(encoding="utf-8") == "a\nb\n"
    with pytest.raises(FileNotFoundError, match="absent.bin"):
        extract_members(upstream["archive"], {"absent.bin": out / "a.bin"})
    with pytest.raises(Cancelled):
        extract_members(upstream["archive"], {"decoder.onnx": out / "d.onnx"}, cancel=lambda: True)
    assert not (out / "d.onnx").exists() and not list(out.glob("*.part"))


def test_cancel_leaves_no_half_file(upstream: dict[str, Path], tmp_path: Path) -> None:
    root = tmp_path / "store"
    with pytest.raises(Cancelled):
        fetch_specs(_specs(upstream), root, cancel=lambda: True)
    assert not (root / "plain-model" / "model.bin").exists() and not list(root.rglob("*.part"))
    assert read_lock(root) == {}


def test_download_removes_the_partial_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "out" / "file.bin"
    with pytest.raises(Exception):
        fetch.download((tmp_path / "absent.bin").resolve().as_uri(), target)
    assert not target.exists() and not target.with_name("file.bin.part").exists()


def test_progress_fraction_by_stage() -> None:
    assert Progress("k", "f", 50, 100, 0, 3).fraction == 0.5
    assert Progress("k", "f", 50, 0, 0, 3).fraction is None
    assert Progress("k", "f", 0, 0, 0, 3, STAGE_EXTRACT).fraction is None
    assert Progress("k", "f", 100, 100, 1, 3, STAGE_DONE).fraction is None


def test_level_plans() -> None:
    standard = [spec.key for spec in specs_for_level("standard")]
    assert standard == [KEY_PARAKEET_V2, KEY_WHISPER_TURBO, KEY_SILERO_VAD, KEY_SEGMENTATION, KEY_EMBEDDING, KEY_AUDIO_TAGGER]
    assert KEY_WHISPER_LARGE in [spec.key for spec in specs_for_level("careful")]
    assert all(spec.size_mb > 0 for spec in specs_for_level("careful"))
    with pytest.raises(KeyError):
        specs_for_level("ultimate")


def test_missing_for_levels_follows_the_store(tmp_path: Path) -> None:
    store = find_models(tmp_path)
    keys = [spec.key for spec in missing_for_levels(["standard"], store)]
    assert keys == [KEY_PARAKEET_V2, KEY_WHISPER_TURBO, KEY_SILERO_VAD, KEY_SEGMENTATION, KEY_EMBEDDING, KEY_AUDIO_TAGGER]
    (tmp_path / KEY_SILERO_VAD).mkdir()
    (tmp_path / KEY_SILERO_VAD / "silero_vad.onnx").write_bytes(b"x")
    keys = [spec.key for spec in missing_for_levels(["standard"], find_models(tmp_path))]
    assert KEY_SILERO_VAD not in keys and KEY_PARAKEET_V2 in keys
    both = [spec.key for spec in missing_for_levels(["standard", "careful"], find_models(tmp_path))]
    assert KEY_WHISPER_LARGE in both and both.count(KEY_PARAKEET_V2) == 1


def test_proposed_root_prefers_the_store_then_the_executable_then_the_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TWINSCRIBE_MODELS", raising=False)
    monkeypatch.setenv("TWINSCRIBE_HOME", str(tmp_path / "home"))
    store = find_models(tmp_path / "store")
    assert proposed_root(store) == tmp_path / "store"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "dist" / "twinscribe.exe"))
    (tmp_path / "dist").mkdir()
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"")
    unwritable = find_models(blocker / "store")
    assert proposed_root(unwritable) == (tmp_path / "dist" / "models").resolve()
    monkeypatch.setenv("TWINSCRIBE_MODELS", str(tmp_path / "named"))
    assert proposed_root(find_models(None)) == tmp_path / "named"
