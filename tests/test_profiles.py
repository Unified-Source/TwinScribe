"""Tests for the quality levels and their dependence on the model store."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._fixtures import make_models
from twinscribe.engines.presets import PARAKEET_PRESETS, WHISPER_PRESETS
from twinscribe.models import (
    KEY_EMBEDDING,
    KEY_PARAKEET_V2,
    KEY_SEGMENTATION,
    KEY_SILERO_VAD,
    KEY_WHISPER_DISTIL,
    KEY_WHISPER_DISTIL_ONNX,
    KEY_WHISPER_LARGE,
    KEY_WHISPER_LARGE_ONNX,
    KEY_WHISPER_TURBO,
    KEY_WHISPER_TURBO_ONNX,
    find_models,
)
from twinscribe.profiles import (
    CHECK_EVERYWHERE,
    CHECK_GAPS,
    DEFAULT_PROFILE,
    PROFILES,
    ModelsMissing,
    available_profiles,
    choose_profile,
    detector_candidates,
    missing_models,
    profile_for,
    resolve_detector,
    select,
)


def test_profiles_are_the_four_levels() -> None:
    assert [p.name for p in PROFILES] == ["quick", "standard", "careful", "laptop"]
    assert DEFAULT_PROFILE == "standard"
    standard = profile_for("standard")
    assert standard.publisher == KEY_PARAKEET_V2 and standard.detector == KEY_WHISPER_TURBO
    assert standard.detector_preset == "production" and standard.publisher_preset == "vad"
    assert profile_for("quick").detector == KEY_WHISPER_DISTIL and profile_for("quick").detector_preset == "quick"
    assert profile_for("careful").detector == KEY_WHISPER_LARGE
    assert standard.detectors == (KEY_WHISPER_TURBO, KEY_WHISPER_TURBO_ONNX)
    for profile in PROFILES:
        assert set(profile.model_keys) >= {KEY_SILERO_VAD, KEY_SEGMENTATION, KEY_EMBEDDING}
        assert set(profile.required_keys) == {profile.publisher, KEY_SILERO_VAD, KEY_SEGMENTATION, KEY_EMBEDDING}
        assert profile.review.min_silence_s == 0.8 and profile.review.min_detector_words == 2 and profile.review.pad_s == 0.4
    laptop = profile_for("laptop")
    assert laptop.checking == CHECK_GAPS and laptop.checking_margin_s == 0.5
    assert laptop.detectors == standard.detectors and laptop.publisher == standard.publisher and laptop.model_keys == standard.model_keys
    assert all(p.checking == CHECK_EVERYWHERE for p in PROFILES if p.name != "laptop")


def test_profile_presets_exist_and_detector_keeps_word_times() -> None:
    for profile in PROFILES:
        assert profile.publisher_preset in PARAKEET_PRESETS
        detector = WHISPER_PRESETS[profile.detector_preset]
        assert detector["word_timestamps"] is True, profile.name


def test_unknown_profile_lists_names() -> None:
    with pytest.raises(KeyError) as excinfo:
        profile_for("ultra")
    assert "standard" in str(excinfo.value)


def _remove(root: Path, *keys: str) -> None:
    for key in keys:
        for child in (root / key).iterdir():
            child.unlink()
        (root / key).rmdir()


def test_availability_follows_the_store(tmp_path: Path) -> None:
    empty = find_models(tmp_path / "empty")
    assert available_profiles(empty) == []
    assert set(missing_models(profile_for("standard"), empty)) == set(profile_for("standard").model_keys)

    full = make_models(tmp_path / "full")
    assert [p.name for p in available_profiles(full)] == ["quick", "standard", "careful", "laptop"]
    assert choose_profile("careful", full).name == "careful"
    assert select("standard", full).detector == KEY_WHISPER_TURBO and select("standard", full).backend == "ct2"

    partial_root = tmp_path / "partial"
    make_models(partial_root)
    _remove(partial_root, KEY_WHISPER_DISTIL, KEY_WHISPER_LARGE, KEY_WHISPER_DISTIL_ONNX, KEY_WHISPER_LARGE_ONNX)
    partial = find_models(partial_root)
    assert [p.name for p in available_profiles(partial)] == ["standard", "laptop"]
    with pytest.raises(ModelsMissing) as excinfo:
        choose_profile("quick", partial)
    assert str(partial_root / KEY_WHISPER_DISTIL) in str(excinfo.value)


def test_detector_falls_back_to_the_onnx_export_without_ctranslate2(tmp_path: Path) -> None:
    full = make_models(tmp_path / "full")
    ct2_only = {"ct2": True, "onnx": False}
    onnx_only = {"ct2": False, "onnx": True}
    assert select("standard", full, onnx_only).detector == KEY_WHISPER_TURBO_ONNX
    assert select("standard", full, onnx_only).backend == "onnx"
    assert select("quick", full, onnx_only).detector == KEY_WHISPER_DISTIL_ONNX
    assert [p.name for p in available_profiles(full, onnx_only)] == ["quick", "standard", "careful", "laptop"]
    assert available_profiles(full, {"ct2": False, "onnx": False}) == []

    root = tmp_path / "ct2_models_only"
    make_models(root)
    _remove(root, KEY_WHISPER_TURBO_ONNX, KEY_WHISPER_DISTIL_ONNX, KEY_WHISPER_LARGE_ONNX)
    store = find_models(root)
    assert [p.name for p in available_profiles(store, ct2_only)] == ["quick", "standard", "careful", "laptop"]
    assert available_profiles(store, onnx_only) == []
    with pytest.raises(ModelsMissing) as excinfo:
        select("standard", store, onnx_only)
    assert str(root / KEY_WHISPER_TURBO_ONNX) in str(excinfo.value)
    assert resolve_detector(profile_for("standard"), store, onnx_only) is None
    assert detector_candidates(profile_for("standard"), full) == [KEY_WHISPER_TURBO, KEY_WHISPER_TURBO_ONNX]
