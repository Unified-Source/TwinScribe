"""Quality levels.

A profile names which catalogue models play each role, the preset each engine runs with and
the thresholds of the review list. The standard profile is the configuration the design
measured; quick and careful swap the detector for a faster or a fuller Whisper conversion.
Every level lists its detector candidates in order of preference: the CTranslate2 conversion
first, the ONNX export for machines without CTranslate2 second. A level is offered only when
its models are present and one of its detectors can run with the libraries installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from twinscribe.models import (
    BACKEND_CT2,
    BACKEND_ONNX,
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
    ModelSet,
    spec_for,
)

ALL_BACKENDS: dict[str, bool] = {BACKEND_CT2: True, BACKEND_ONNX: True}


@dataclass(frozen=True)
class ReviewSettings:
    """Thresholds of the review list; the defaults are the rule the design tested."""

    min_silence_s: float = 0.8
    min_detector_words: int = 2
    pad_s: float = 0.4


CHECK_EVERYWHERE = "everywhere"
CHECK_GAPS = "gaps"


@dataclass(frozen=True)
class Profile:
    """One quality level."""

    name: str
    title: str
    description: str
    publisher: str
    detectors: tuple[str, ...]
    publisher_preset: str = "vad"
    detector_preset: str = "production"
    review: ReviewSettings = field(default_factory=ReviewSettings)
    diarization_threshold: float = 0.5
    checking: str = CHECK_EVERYWHERE
    checking_margin_s: float = 1.0

    @property
    def detector(self) -> str:
        """The preferred detector model."""
        return self.detectors[0]

    @property
    def required_keys(self) -> tuple[str, ...]:
        """The models every backend needs: the publisher, the voice detector, the speaker models."""
        return (self.publisher, KEY_SILERO_VAD, KEY_SEGMENTATION, KEY_EMBEDDING)

    @property
    def model_keys(self) -> tuple[str, ...]:
        """The required models plus the preferred detector."""
        return (self.publisher, self.detector, KEY_SILERO_VAD, KEY_SEGMENTATION, KEY_EMBEDDING)


PROFILES: tuple[Profile, ...] = (
    Profile(
        name="quick",
        title="Quick",
        description="Distil-Whisper as the detector with greedy decoding; the same published engine.",
        publisher=KEY_PARAKEET_V2,
        detectors=(KEY_WHISPER_DISTIL, KEY_WHISPER_DISTIL_ONNX),
        detector_preset="quick",
    ),
    Profile(
        name="standard",
        title="Standard",
        description="The measured configuration: Parakeet published, Whisper large-v3-turbo checking.",
        publisher=KEY_PARAKEET_V2,
        detectors=(KEY_WHISPER_TURBO, KEY_WHISPER_TURBO_ONNX),
    ),
    Profile(
        name="careful",
        title="Careful",
        description="Full-size Whisper large-v3 as the detector; slower, for the hardest audio.",
        publisher=KEY_PARAKEET_V2,
        detectors=(KEY_WHISPER_LARGE, KEY_WHISPER_LARGE_ONNX),
    ),
    Profile(
        name="laptop",
        title="Laptop",
        description="The Standard engines; the checker decodes only where the published engine fell silent, for machines without a graphics device.",
        publisher=KEY_PARAKEET_V2,
        detectors=(KEY_WHISPER_TURBO, KEY_WHISPER_TURBO_ONNX),
        checking=CHECK_GAPS,
    ),
)

DEFAULT_PROFILE = "standard"

_BY_NAME: dict[str, Profile] = {profile.name: profile for profile in PROFILES}


class ModelsMissing(ValueError):
    """A profile was asked for while the store lacks a model it needs."""


@dataclass(frozen=True)
class Selection:
    """A level with the detector that will run: its model key and its backend."""

    profile: Profile
    detector: str
    backend: str


def profile_for(name: str) -> Profile:
    """A profile by name; KeyError lists the known names."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown quality level {name!r}; known levels: {', '.join(_BY_NAME)}") from None


def backend_of(key: str) -> str:
    """The backend a detector model key is for."""
    return spec_for(key).backend


def detector_candidates(profile: Profile, models: ModelSet, backends: Mapping[str, bool] | None = None) -> list[str]:
    """The profile's detectors that are present in the store and whose backend is usable.

    backends maps ct2 and onnx to whether their library is installed; None means both.
    """
    usable = dict(ALL_BACKENDS) if backends is None else dict(backends)
    return [key for key in profile.detectors if models.has(key) and usable.get(backend_of(key), False)]


def resolve_detector(profile: Profile, models: ModelSet, backends: Mapping[str, bool] | None = None) -> str | None:
    """The detector that will run, in order of preference, or None."""
    candidates = detector_candidates(profile, models, backends)
    return candidates[0] if candidates else None


def missing_models(profile: Profile, models: ModelSet, backends: Mapping[str, bool] | None = None) -> tuple[str, ...]:
    """Model keys the profile needs that the store does not hold complete.

    When no detector candidate can run, the preferred detector is named, or the first
    candidate whose backend is usable when the preferred one's is not.
    """
    missing = [key for key in profile.required_keys if not models.has(key)]
    if resolve_detector(profile, models, backends) is None:
        usable = dict(ALL_BACKENDS) if backends is None else dict(backends)
        wanted = next((key for key in profile.detectors if usable.get(backend_of(key), False)), profile.detector)
        missing.append(wanted)
    return tuple(missing)


def available_profiles(models: ModelSet, backends: Mapping[str, bool] | None = None) -> list[Profile]:
    """Profiles whose every model is present and whose detector can run, in defined order."""
    return [profile for profile in PROFILES if not missing_models(profile, models, backends)]


def select(name: str, models: ModelSet, backends: Mapping[str, bool] | None = None) -> Selection:
    """The named profile with its resolved detector, or ModelsMissing naming what is absent."""
    profile = profile_for(name)
    missing = missing_models(profile, models, backends)
    if missing:
        wanted = "; ".join(f"{spec_for(key).title} at {models.root / key}" for key in missing)
        raise ModelsMissing(f"quality level {name!r} needs models that are not present: {wanted}")
    detector = resolve_detector(profile, models, backends)
    assert detector is not None
    return Selection(profile=profile, detector=detector, backend=backend_of(detector))


def choose_profile(name: str, models: ModelSet, backends: Mapping[str, bool] | None = None) -> Profile:
    """The named profile, or ModelsMissing naming the folders the store lacks."""
    return select(name, models, backends).profile
