"""Engine presets as plain dictionaries.

The Whisper presets are keyed by the parameter names of faster-whisper's transcribe call so
they can be passed through unchanged. Two exist because the design measures the Whisper
family at production settings (beam search, no voice filter, every window decoded, word
times on) and also needs a cheap benchmark arm (greedy, voice filter, conditioning on) that
represents the settings most deployments actually run. A third, quick, is production with
greedy decoding; it keeps word times on because the review list is built from them. The
transducer preset describes the Silero voice detector that carves long audio into utterances
for the transducer.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

DEFAULT_TEMPERATURE_LADDER: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

WHISPER_PRESETS: dict[str, dict[str, Any]] = {
    "production": {
        "beam_size": 5,
        "vad_filter": False,
        "no_speech_threshold": 0.7,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "temperature": DEFAULT_TEMPERATURE_LADDER,
    },
    "quick": {
        "beam_size": 1,
        "vad_filter": False,
        "no_speech_threshold": 0.7,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "temperature": DEFAULT_TEMPERATURE_LADDER,
    },
    "benchmark": {
        "beam_size": 1,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
        "word_timestamps": False,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "temperature": DEFAULT_TEMPERATURE_LADDER,
    },
}

PARAKEET_PRESETS: dict[str, dict[str, Any]] = {
    "vad": {
        "vad": "silero",
        "threshold": 0.5,
        "min_silence_duration": 0.5,
        "min_speech_duration": 0.25,
        "max_speech_duration": 20.0,
        "window_size": 512,
        "decoding_method": "greedy_search",
    },
}

CUSTOM_PRESET_NAME = "custom"


def _lookup(table: Mapping[str, Mapping[str, Any]], name: str, family: str) -> dict[str, Any]:
    if name not in table:
        known = ", ".join(sorted(table))
        raise KeyError(f"unknown {family} preset {name!r}; known presets: {known}")
    return copy.deepcopy(dict(table[name]))


def get_whisper_preset(name: str) -> dict[str, Any]:
    """Deep copy of a Whisper preset by name; KeyError lists the known names."""
    return _lookup(WHISPER_PRESETS, name, "whisper")


def get_parakeet_preset(name: str) -> dict[str, Any]:
    """Deep copy of a transducer preset by name; KeyError lists the known names."""
    return _lookup(PARAKEET_PRESETS, name, "parakeet")


def resolve_preset(
    preset: str | Mapping[str, Any],
    table: Mapping[str, Mapping[str, Any]],
    family: str,
) -> tuple[str, dict[str, Any]]:
    """Accept a preset name or an explicit mapping and return (name, settings copy).

    A mapping is recorded under the name "custom" so that a transcript always states which
    preset produced it; a name is looked up in the given table.
    """
    if isinstance(preset, str):
        return preset, _lookup(table, preset, family)
    if isinstance(preset, Mapping):
        return CUSTOM_PRESET_NAME, copy.deepcopy(dict(preset))
    raise TypeError(f"preset must be a name or a mapping, got {type(preset).__name__}")
