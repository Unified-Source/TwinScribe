"""Transcription and diarization engines.

Records and presets are importable without any engine library present. The engine modules
(whisper_ct2, parakeet, diarize) import their libraries lazily inside the functions that
need them, so this package imports cleanly on a machine with neither installed.
"""

from twinscribe.engines.base import (
    Diarization,
    Segment,
    SpeakerTurn,
    Stopwatch,
    Transcript,
    Word,
    default_threads,
)
from twinscribe.engines.presets import (
    PARAKEET_PRESETS,
    WHISPER_PRESETS,
    get_parakeet_preset,
    get_whisper_preset,
)

__all__ = [
    "Diarization",
    "PARAKEET_PRESETS",
    "Segment",
    "SpeakerTurn",
    "Stopwatch",
    "Transcript",
    "WHISPER_PRESETS",
    "Word",
    "default_threads",
    "get_parakeet_preset",
    "get_whisper_preset",
]
