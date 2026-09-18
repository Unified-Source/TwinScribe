"""Tests for the portable folder's tooling: the launchers setting the folders beside them and
unpacking the parts placed there once, a store cut down to a level with a lock to match, and
an assembled folder packed into the parts a release can carry, with nothing in two parts,
nothing shipped that should not be, and a part too large refused."""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import pytest

from twinscribe.models import (
    KEY_AUDIO_TAGGER,
    KEY_PARAKEET_V2,
    KEY_WHISPER_LARGE,
    KEY_WHISPER_TURBO,
    KEY_WHISPER_TURBO_ONNX,
    read_lock,
    spec_for,
    write_lock,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import build_portable  # noqa: E402

CHECKER = KEY_WHISPER_TURBO
PUBLISHER = KEY_PARAKEET_V2


def entry(digest: str) -> dict[str, object]:
    return {"sha256": digest, "bytes": 1, "url": "x", "archive": None, "revision": None, "licence": "MIT", "fetched_utc": "t"}


def test_launchers_set_the_folders_beside_them_and_unpack_parts_once() -> None:
    starts = (
        (build_portable.LAUNCHER_CLI, '"%HERE%python\\python.exe" -m twinscribe %*'),
        (build_portable.LAUNCHER_APP, 'start "" "%HERE%python\\pythonw.exe" -m twinscribe.app %*'),
    )
    for text, start in starts:
        text.encode("ascii")
        assert 'set "TWINSCRIBE_HOME=%HERE%home"' in text
        assert 'set "TWINSCRIBE_MODELS=%HERE%models"' in text
        assert 'for %%Z in ("%HERE%twinscribe-portable-*.zip") do call :unpack "%%~fZ"' in text
        assert 'ren %1 "%~nx1.unpacked"' in text
        assert text.index("if defined UNPACK_FAILED exit /b 1") < text.index(start) < text.index("\n:unpack\n")


def test_the_level_keys_and_the_detector_follow_the_architecture() -> None:
    amd64 = build_portable.level_keys("standard", "amd64")
    assert CHECKER in amd64 and PUBLISHER in amd64 and KEY_AUDIO_TAGGER in amd64
    assert KEY_WHISPER_TURBO_ONNX not in amd64
    arm64 = build_portable.level_keys("standard", "arm64")
    assert KEY_WHISPER_TURBO_ONNX in arm64 and CHECKER not in arm64
    assert build_portable.detector_key("standard", "amd64") == CHECKER
    assert build_portable.detector_key("standard", "arm64") == KEY_WHISPER_TURBO_ONNX


def make_store(root: Path, keys: tuple[str, ...]) -> None:
    entries = {}
    for key in keys:
        name = spec_for(key).required[0]
        path = root / key / name
        path.parent.mkdir(parents=True)
        path.write_bytes(key.encode("ascii"))
        entries[f"{key}/{name}"] = entry(hashlib.sha256(key.encode("ascii")).hexdigest())
    write_lock(root, entries)


def test_copy_models_keeps_the_chosen_folders_and_their_lock_entries_only(tmp_path: Path) -> None:
    source = tmp_path / "store"
    make_store(source, (PUBLISHER, CHECKER, KEY_WHISPER_LARGE))
    target = tmp_path / "models"
    copied = build_portable.copy_models(source, target, (PUBLISHER, CHECKER))
    assert copied == (PUBLISHER, CHECKER)
    assert sorted(p.name for p in target.iterdir() if p.is_dir()) == sorted((PUBLISHER, CHECKER))
    kept = read_lock(target)
    assert {k.split("/", 1)[0] for k in kept} == {PUBLISHER, CHECKER}
    assert len(read_lock(source)) == 3
    with pytest.raises(FileNotFoundError):
        build_portable.copy_models(source, tmp_path / "again", (PUBLISHER, KEY_WHISPER_TURBO_ONNX))


def test_copy_models_without_keys_takes_every_folder(tmp_path: Path) -> None:
    source = tmp_path / "store"
    make_store(source, (PUBLISHER, CHECKER))
    copied = build_portable.copy_models(source, tmp_path / "models", None)
    assert copied == tuple(sorted((PUBLISHER, CHECKER)))
    assert len(read_lock(tmp_path / "models")) == 2


SHIPPED = (
    "python/python.exe",
    "python/python311._pth",
    "Lib/site-packages/numpy/__init__.py",
    "Lib/site-packages/PySide6/Qt6Core.dll",
    "twinscribe/__init__.py",
    "twinscribe/app/main.py",
    f"models/{PUBLISHER}/encoder.int8.onnx",
    "models/models.lock.json",
    "NOTICE",
    "LICENSE",
    "RECORD.txt",
    "twinscribe.cmd",
    "twinscribe-app.cmd",
)
CHECKER_FILES = (f"models/{CHECKER}/model.bin", f"models/{CHECKER}/config.json")
CUDA_FILES = (
    "Lib/site-packages/nvidia/__init__.py",
    "Lib/site-packages/nvidia/cublas/bin/cublas64_12.dll",
    "Lib/site-packages/nvidia/cudnn/bin/cudnn64_9.dll",
    "Lib/site-packages/nvidia_cublas_cu12-12.0.dist-info/RECORD",
)
NOT_SHIPPED = (
    "wheels/numpy-2.0-cp311-cp311-win_amd64.whl",
    "home/settings.json",
    "home/twinscribe.log",
    "twinscribe/__pycache__/__init__.cpython-311.pyc",
    "twinscribe-portable-win64-part2.zip",
    "twinscribe-portable-win64-cuda.zip.unpacked",
)


def make_folder(root: Path, cuda: bool = True, checker: bool = True) -> None:
    names = list(SHIPPED) + list(NOT_SHIPPED)
    if cuda:
        names += CUDA_FILES
    if checker:
        names += CHECKER_FILES
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("ascii") * 40)


def members_of(path: Path) -> dict[str, zipfile.ZipInfo]:
    with zipfile.ZipFile(path) as archive:
        return {info.filename: info for info in archive.infolist()}


def test_pack_splits_the_folder_into_parts_with_nothing_twice_and_nothing_extra(tmp_path: Path) -> None:
    folder = tmp_path / "twinscribe-portable"
    make_folder(folder)
    parts = build_portable.pack(folder, tmp_path / "out", CHECKER)
    assert [p.name for p in parts] == [
        "twinscribe-portable-win64-part1.zip",
        "twinscribe-portable-win64-part2.zip",
        "twinscribe-portable-win64-cuda.zip",
    ]
    part1, part2, cuda = (members_of(p.path) for p in parts)
    top = "twinscribe-portable/"
    assert all(name.startswith(top) for name in part1)
    assert top + "home/" in part1 and part1[top + "home/"].is_dir()
    files1 = {name[len(top):] for name in part1 if not name.endswith("/")}
    assert files1 == set(SHIPPED)
    assert set(part2) == set(CHECKER_FILES)
    assert set(cuda) == set(CUDA_FILES)
    assert not (files1 & set(part2)) and not (files1 & set(cuda)) and not (set(part2) & set(cuda))
    assert files1 | set(part2) | set(cuda) == set(SHIPPED) | set(CHECKER_FILES) | set(CUDA_FILES)
    for name, info in {**part1, **part2, **cuda}.items():
        if info.is_dir():
            continue
        inside = name[len(top):] if name.startswith(top) else name
        expected = zipfile.ZIP_STORED if inside.startswith("models/") else zipfile.ZIP_DEFLATED
        assert info.compress_type == expected, name
    for part in parts:
        assert part.bytes == part.path.stat().st_size
        assert part.sha256 == hashlib.sha256(part.path.read_bytes()).hexdigest()
    assert [p.files for p in parts] == [len(SHIPPED), len(CHECKER_FILES), len(CUDA_FILES)]
    lines = build_portable.describe_parts(parts)
    assert len(lines) == 3 and all(p.sha256 in line for p, line in zip(parts, lines))


def test_pack_without_cuda_packages_writes_two_parts(tmp_path: Path) -> None:
    folder = tmp_path / "twinscribe-portable"
    make_folder(folder, cuda=False)
    parts = build_portable.pack(folder, tmp_path / "out", CHECKER)
    assert [p.name for p in parts] == ["twinscribe-portable-win64-part1.zip", "twinscribe-portable-win64-part2.zip"]
    assert not (tmp_path / "out" / "twinscribe-portable-win64-cuda.zip").exists()


def test_pack_refuses_a_folder_without_the_checker_and_a_part_too_large(tmp_path: Path) -> None:
    folder = tmp_path / "twinscribe-portable"
    make_folder(folder, checker=False)
    with pytest.raises(ValueError, match="part2"):
        build_portable.pack(folder, tmp_path / "out", CHECKER)
    make_folder(folder)
    with pytest.raises(ValueError, match="part1.*over"):
        build_portable.pack(folder, tmp_path / "out", CHECKER, limit=100)


def test_the_plan_names_the_wheel_set_the_level_and_the_parts() -> None:
    steps = build_portable.plan(argparse.Namespace(
        out=Path("p"), arch="amd64", gpu=True, bundle_ffmpeg=True, wheels=Path("w"), models=Path("m"),
        level="standard", ffmpeg=None, pack=True,
    ))
    text = "\n".join(steps)
    assert "already in w" in text and "standard level" in text and CHECKER in text and "part1.zip" in text
