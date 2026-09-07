"""Whisper family on the processor through faster-whisper and CTranslate2 in int8.

This engine is never published. It runs beside the transducer so that every span where it
heard speech and the published engine heard nothing can be marked for review. The model is
loaded from a local directory only; the environment is put into offline mode before the
library is imported so that no hub lookup can happen.
"""

from __future__ import annotations

import copy
import importlib
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from twinscribe.engines.base import Segment, Transcript, Word, default_threads
from twinscribe.engines.presets import WHISPER_PRESETS, resolve_preset

ENGINE_NAME = "whisper_ct2"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}

# Files a CTranslate2 Whisper conversion must contain; checked before the library loads so
# that a wrong directory fails with a message naming the missing file rather than a hub error.
REQUIRED_MODEL_FILES: tuple[str, ...] = ("model.bin", "config.json")

# Names faster-whisper's transcribe call accepts. Presets are validated against this set so
# that a misspelt key fails before the model is loaded.
TRANSCRIBE_PARAMETERS: frozenset[str] = frozenset(
    {
        "language",
        "task",
        "log_progress",
        "beam_size",
        "best_of",
        "patience",
        "length_penalty",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "temperature",
        "compression_ratio_threshold",
        "log_prob_threshold",
        "no_speech_threshold",
        "condition_on_previous_text",
        "prompt_reset_on_temperature",
        "initial_prompt",
        "prefix",
        "suppress_blank",
        "suppress_tokens",
        "without_timestamps",
        "max_initial_timestamp",
        "word_timestamps",
        "prepend_punctuations",
        "append_punctuations",
        "multilingual",
        "vad_filter",
        "vad_parameters",
        "max_new_tokens",
        "chunk_length",
        "clip_timestamps",
        "hallucination_silence_threshold",
        "hotwords",
        "language_detection_threshold",
        "language_detection_segments",
    }
)


def set_offline_env() -> dict[str, str]:
    """Force the hub-related environment variables to their offline values.

    Existing values are overwritten rather than kept, because a stray "0" in the environment
    would otherwise let the library attempt a download. Returns the values now in effect.
    """
    for key, value in OFFLINE_ENV.items():
        os.environ[key] = value
    return {key: os.environ[key] for key in OFFLINE_ENV}


def transcribe_kwargs(preset: Mapping[str, Any]) -> dict[str, Any]:
    """Keyword arguments for the library's transcribe call, derived from a preset.

    The temperature ladder is handed over as a list, and any key the library does not
    accept raises ValueError naming it.
    """
    unknown = sorted(set(preset) - TRANSCRIBE_PARAMETERS)
    if unknown:
        raise ValueError(f"preset keys not accepted by faster-whisper: {', '.join(unknown)}")
    kwargs = copy.deepcopy(dict(preset))
    if "temperature" in kwargs and not isinstance(kwargs["temperature"], (int, float)):
        kwargs["temperature"] = [float(t) for t in kwargs["temperature"]]
    return kwargs


def check_model_dir(model_dir: str | os.PathLike[str]) -> Path:
    """Return the model directory as a Path after confirming the conversion files exist."""
    directory = Path(model_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"whisper model directory not found: {directory}")
    missing = [name for name in REQUIRED_MODEL_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"whisper model directory {directory} lacks {', '.join(missing)}; "
            "a CTranslate2 conversion is required"
        )
    return directory


def _import_faster_whisper() -> Any:
    """Import faster_whisper after putting the environment into offline mode."""
    set_offline_env()
    try:
        return importlib.import_module("faster_whisper")
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is not installed; install the 'engines' extra to use this engine"
        ) from exc


def library_versions() -> dict[str, str]:
    """Versions of faster-whisper and CTranslate2 as installed, for the run record."""
    versions: dict[str, str] = {}
    for module_name, key in (("faster_whisper", "faster_whisper"), ("ctranslate2", "ctranslate2")):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        versions[key] = str(getattr(module, "__version__", "unknown"))
    return versions


def segment_from_library(seg: Any, want_words: bool) -> Segment:
    """Convert one faster-whisper segment into the shared Segment record.

    Word times are copied only when the preset asked for them; the library returns None
    for words otherwise. The quality figures are the per-segment values the library reports
    for its own fallback decisions: average log probability, no-speech probability and the
    zlib compression ratio of the decoded text.
    """
    words: tuple[Word, ...] = ()
    library_words = getattr(seg, "words", None)
    if want_words and library_words:
        words = tuple(
            Word(
                text=str(w.word).strip(),
                start=float(w.start),
                end=float(w.end),
                prob=None if w.probability is None else float(w.probability),
            )
            for w in library_words
        )
    temperature = getattr(seg, "temperature", None)
    quality: dict[str, float | None] = {
        "avg_logprob": _float_or_none(getattr(seg, "avg_logprob", None)),
        "no_speech_prob": _float_or_none(getattr(seg, "no_speech_prob", None)),
        "compression_ratio": _float_or_none(getattr(seg, "compression_ratio", None)),
        "temperature": _float_or_none(temperature),
    }
    return Segment(
        start=float(seg.start),
        end=float(seg.end),
        text=str(seg.text).strip(),
        words=words,
        quality=quality,
    )


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def transcribe(
    audio_path: str | os.PathLike[str],
    model_dir: str | os.PathLike[str],
    preset: str | Mapping[str, Any],
    threads: int | None = None,
    progress: Callable[[float], None] | None = None,
) -> Transcript:
    """Transcribe one file with a local CTranslate2 Whisper conversion on the processor.

    The load timer covers model construction. The transcription timer encloses full
    consumption of the segment generator, because the library decodes lazily and the
    generator is where the work happens. The duration the library reports after voice
    filtering is recorded in extras so that a benchmark run states how much audio it decoded.
    progress, when given, is called after every decoded segment with the fraction of the
    audio reached so far; an exception raised inside it propagates and abandons the run,
    which is how a caller cancels.
    """
    audio = Path(audio_path)
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")
    directory = check_model_dir(model_dir)
    preset_name, settings = resolve_preset(preset, WHISPER_PRESETS, "whisper")
    kwargs = transcribe_kwargs(settings)
    want_words = bool(kwargs.get("word_timestamps", False))
    thread_count = default_threads(threads)

    faster_whisper = _import_faster_whisper()

    load_start = time.perf_counter()
    model = faster_whisper.WhisperModel(
        str(directory),
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=thread_count,
        local_files_only=True,
    )
    load_s = time.perf_counter() - load_start

    transcribe_start = time.perf_counter()
    generator, info = model.transcribe(str(audio), **kwargs)
    audio_s = float(getattr(info, "duration", 0.0) or 0.0)
    collected: list[Segment] = []
    for seg in generator:
        collected.append(segment_from_library(seg, want_words))
        if progress is not None and audio_s > 0.0:
            progress(min(1.0, float(seg.end) / audio_s))
    segments = tuple(collected)
    transcribe_s = time.perf_counter() - transcribe_start
    extras: dict[str, float | int | None] = {
        "duration_after_vad_s": _float_or_none(getattr(info, "duration_after_vad", None)),
        "language_probability": _float_or_none(getattr(info, "language_probability", None)),
        "segments": len(segments),
        "words": sum(len(segment.words) for segment in segments),
        "threads": thread_count,
    }
    return Transcript(
        engine=ENGINE_NAME,
        model=directory.name,
        preset=preset_name,
        segments=segments,
        audio_s=audio_s,
        load_s=load_s,
        transcribe_s=transcribe_s,
        versions=library_versions(),
        extras=extras,
    )
