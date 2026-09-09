"""Tests for the folders a frozen build uses: models and home beside its executable when they
exist there, the usual places otherwise."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from twinscribe import models, paths


@pytest.fixture
def frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    folder = tmp_path / "dist" / "twinscribe"
    folder.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(folder / "twinscribe.exe"))
    monkeypatch.delenv(paths.HOME_ENV, raising=False)
    monkeypatch.delenv(models.MODELS_ENV, raising=False)
    return folder


def test_not_frozen_has_no_sibling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert paths.frozen_sibling("models") is None


def test_frozen_sibling_needs_the_folder(frozen: Path) -> None:
    assert paths.frozen_sibling("models") is None
    (frozen / "models").mkdir()
    assert paths.frozen_sibling("models") == (frozen / "models").resolve()


def test_frozen_models_root_beside_the_executable(frozen: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (frozen / "models").mkdir()
    assert models.default_models_root() == (frozen / "models").resolve()
    # The environment variable still wins.
    monkeypatch.setenv(models.MODELS_ENV, str(frozen / "elsewhere"))
    assert models.default_models_root() == frozen / "elsewhere"


def test_frozen_home_beside_the_executable(frozen: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert paths.app_home() == paths.default_app_home()
    (frozen / "home").mkdir()
    assert paths.app_home() == (frozen / "home").resolve()
    monkeypatch.setenv(paths.HOME_ENV, str(frozen / "other"))
    assert paths.app_home() == frozen / "other"
