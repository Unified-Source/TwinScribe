"""Quality levels.

A profile names which catalogue models play each role, the preset each engine runs with and
the thresholds of the review list. The standard profile is the configuration the design
measured; quick and careful swap the detector for a faster or a fuller Whisper conversion,
and are offered only when their models are present in the store.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from twinscribe.models import (
    KEY_EMBEDDING,
    KEY_PARAKEET_V2,
    KEY_SEGMENTATION,
    KEY_SILERO_VAD,
    KEY_WHISPER_DISTIL,
    KEY_WHISPER_LARGE,
    KEY_WHISPER_TURBO,
    ModelSet,
    spec_for,
)


@dataclass(frozen=True)
class ReviewSettings:
    """Thresholds of the review list; the defaults are the rule the design tested."""

    min_silence_s: float = 0.8
    min_detector_words: int = 2
    pad_s: float = 0.4


@dataclass(frozen=True)
class Profile:
    """One quality level."""

    name: str
    title: str
    description: str
    publisher: str
    detector: str
    publisher_preset: str = "vad"
    detector_preset: str = "production"
    review: ReviewSettings = field(default_factory=ReviewSettings)
    diarization_threshold: float = 0.5

    @property
    def model_keys(self) -> tuple[str, ...]:
        """Every model folder the profile needs, engines and speaker labelling alike."""
        return (self.publisher, self.detector, KEY_SILERO_VAD, KEY_SEGMENTATION, KEY_EMBEDDING)


PROFILES: tuple[Profile, ...] = (
    Profile(
        name="quick",
        title="Quick",
        description="Distil-Whisper as the detector with greedy decoding; the same published engine.",
        publisher=KEY_PARAKEET_V2,
        detector=KEY_WHISPER_DISTIL,
        detector_preset="quick",
    ),
    Profile(
        name="standard",
        title="Standard",
        description="The measured configuration: Parakeet published, Whisper large-v3-turbo checking.",
        publisher=KEY_PARAKEET_V2,
        detector=KEY_WHISPER_TURBO,
    ),
    Profile(
        name="careful",
        title="Careful",
        description="Full-size Whisper large-v3 as the detector; slower, for the hardest audio.",
        publisher=KEY_PARAKEET_V2,
        detector=KEY_WHISPER_LARGE,
    ),
)

DEFAULT_PROFILE = "standard"

_BY_NAME: dict[str, Profile] = {profile.name: profile for profile in PROFILES}


class ModelsMissing(ValueError):
    """A profile was asked for while the store lacks a model it needs."""


def profile_for(name: str) -> Profile:
    """A profile by name; KeyError lists the known names."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown quality level {name!r}; known levels: {', '.join(_BY_NAME)}") from None


def missing_models(profile: Profile, models: ModelSet) -> tuple[str, ...]:
    """Model keys the profile needs that the store does not hold complete."""
    return tuple(key for key in profile.model_keys if not models.has(key))


def available_profiles(models: ModelSet) -> list[Profile]:
    """Profiles whose every model is present, in the order they are defined."""
    return [profile for profile in PROFILES if not missing_models(profile, models)]


def choose_profile(name: str, models: ModelSet) -> Profile:
    """The named profile, or ModelsMissing naming the folders the store lacks."""
    profile = profile_for(name)
    missing = missing_models(profile, models)
    if missing:
        wanted = "; ".join(f"{spec_for(key).title} at {models.root / key}" for key in missing)
        raise ModelsMissing(f"quality level {name!r} needs models that are not present: {wanted}")
    return profile
