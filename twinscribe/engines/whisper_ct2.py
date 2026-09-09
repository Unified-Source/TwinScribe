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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from twinscribe.audio import duration_s as wav_duration_s
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


def speech_within(windows: Sequence[tuple[float, float]], speech: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """The parts of the windows that the speech spans cover, merged where they touch: what a
    checker need decode when it checks windows but not the silence inside them."""
    out: list[tuple[float, float]] = []
    for window_start, window_end in sorted(windows):
        for speech_start, speech_end in sorted(speech):
            start, end = max(window_start, speech_start), min(window_end, speech_end)
            if end <= start:
                continue
            if out and start <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], end))
            else:
                out.append((start, end))
    return out


def speech_spans(samples: Any, vad_parameters: Mapping[str, Any] | None) -> list[tuple[float, float]]:
    """The speech the library's voice detector finds in decoded 16 kHz samples, as (start, end)
    seconds, with the same settings the full check uses to skip silence."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    chunks = get_speech_timestamps(samples, VadOptions(**dict(vad_parameters or {})), sampling_rate=16000)
    return [(float(c["start"]) / 16000.0, float(c["end"]) / 16000.0) for c in chunks]


def concatenate_clips(samples: Any, clips: Sequence[tuple[float, float]]) -> tuple[Any, list[tuple[float, float, float]]]:
    """The clips of the samples joined into one stream, and for each piece its start in the
    stream and its start and end in the recording, so times can be restored afterwards."""
    import numpy as np

    parts = []
    pieces: list[tuple[float, float, float]] = []
    cursor = 0.0
    for start, end in clips:
        first, last = int(round(start * 16000)), int(round(end * 16000))
        part = samples[first:last]
        if len(part) == 0:
            continue
        length = len(part) / 16000.0
        pieces.append((cursor, first / 16000.0, first / 16000.0 + length))
        parts.append(part)
        cursor += length
    stream = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)
    return stream, pieces


def restore_time(pieces: Sequence[tuple[float, float, float]], t: float) -> float:
    """A time in the concatenated stream mapped back to the recording."""
    import bisect

    if not pieces:
        return float(t)
    index = max(0, bisect.bisect_right([p[0] for p in pieces], t) - 1)
    stream_start, clip_start, clip_end = pieces[index]
    return min(clip_end, clip_start + max(0.0, t - stream_start))


def restore_segment(segment: Segment, pieces: Sequence[tuple[float, float, float]]) -> Segment:
    """A segment decoded from the stream with every time put back on the recording's clock."""
    from dataclasses import replace

    words = tuple(replace(w, start=restore_time(pieces, w.start), end=restore_time(pieces, w.end)) for w in segment.words)
    return replace(segment, start=restore_time(pieces, segment.start), end=restore_time(pieces, segment.end), words=words)


def stream_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments of the library call for a stream of joined speech clips: no voice filter,
    since the stream holds the speech the detector already chose."""
    return {k: v for k, v in kwargs.items() if k not in ("vad_filter", "vad_parameters", "clip_timestamps")}


def transcribe(    audio_path: str | os.PathLike[str],
    model_dir: str | os.PathLike[str],
    preset: str | Mapping[str, Any],
    threads: int | None = None,
    progress: Callable[[float], None] | None = None,
    device: str = DEVICE,
    compute_type: str = COMPUTE_TYPE,
    device_index: int = 0,
    on_segment: Callable[[Segment], None] | None = None,
    clips: Sequence[tuple[float, float]] | None = None,
    narrow: bool = True,
) -> Transcript:
    """Transcribe one file with a local CTranslate2 Whisper conversion.

    The load timer covers model construction. The transcription timer encloses full
    consumption of the segment generator, because the library decodes lazily and the
    generator is where the work happens. The duration the library reports after voice
    filtering is recorded in extras so that a benchmark run states how much audio it decoded.
    progress, when given, is called after every decoded segment with the fraction of the
    audio reached so far; an exception raised inside it propagates and abandons the run,
    which is how a caller cancels; it is also called with 0.0 once the model has loaded, so a
    caller can tell loading from decoding. on_segment, when given, receives each segment as it
    is decoded. device is "cpu" (the measured configuration, int8) or "cuda" with a compute
    type the device supports; the acceleration plan chooses them and they are recorded in the
    transcript's settings. clips, when given, are the only spans of the audio decoded, as
    (start, end) seconds: narrowed first to the speech the voice detector finds inside them
    unless narrow is False, joined into one stream so the model reads them in its usual
    windows, and every time put back on the recording's clock; an empty list, before or after
    narrowing, decodes nothing and loads no model.
    """
    audio = Path(audio_path)
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")
    directory = check_model_dir(model_dir)
    preset_name, settings = resolve_preset(preset, WHISPER_PRESETS, "whisper")
    kwargs = transcribe_kwargs(settings)
    want_words = bool(kwargs.get("word_timestamps", False))
    thread_count = default_threads(threads)
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be cpu or cuda, got {device!r}")

    if device == "cuda":
        from twinscribe.hardware import register_cuda_libraries

        register_cuda_libraries()
    faster_whisper = _import_faster_whisper()
    windows: list[tuple[float, float]] | None = None
    stream = None
    pieces: list[tuple[float, float, float]] = []
    if clips is not None:
        windows = [(float(start), float(end)) for start, end in clips]
        samples = faster_whisper.decode_audio(str(audio), sampling_rate=16000)
        if narrow and windows:
            clips = speech_within(windows, speech_spans(samples, settings.get("vad_parameters")))
        stream, pieces = concatenate_clips(samples, clips)
        clips = [(clip_start, clip_end) for _, clip_start, clip_end in pieces]
        kwargs = stream_kwargs(kwargs)
    if clips is not None and not clips:
        if progress is not None:
            progress(0.0)
        return Transcript(
            engine=ENGINE_NAME, model=directory.name, preset=preset_name, segments=(), audio_s=float(wav_duration_s(audio)),
            load_s=0.0, transcribe_s=0.0, versions=library_versions(),
            extras={"segments": 0, "words": 0, "threads": thread_count, "windows": len(windows or []),
                    "windows_s": float(sum(end - start for start, end in (windows or []))), "clips": 0, "clipped_s": 0.0},
            settings={"device": device, "device_index": int(device_index), "compute_type": compute_type},
        )

    load_start = time.perf_counter()
    model = faster_whisper.WhisperModel(
        str(directory),
        device=device,
        device_index=int(device_index),
        compute_type=compute_type,
        cpu_threads=thread_count,
        local_files_only=True,
    )
    load_s = time.perf_counter() - load_start
    if progress is not None:
        progress(0.0)

    transcribe_start = time.perf_counter()
    generator, info = model.transcribe(stream if stream is not None else str(audio), **kwargs)
    decoded_s = float(getattr(info, "duration", 0.0) or 0.0)
    audio_s = float(wav_duration_s(audio)) if stream is not None else decoded_s
    collected: list[Segment] = []
    for seg in generator:
        segment = segment_from_library(seg, want_words)
        if stream is not None:
            segment = restore_segment(segment, pieces)
        collected.append(segment)
        if on_segment is not None:
            on_segment(segment)
        if progress is not None and decoded_s > 0.0:
            progress(min(1.0, float(seg.end) / decoded_s))
    segments = tuple(collected)
    transcribe_s = time.perf_counter() - transcribe_start
    extras: dict[str, float | int | None] = {
        "duration_after_vad_s": _float_or_none(getattr(info, "duration_after_vad", None)),
        "language_probability": _float_or_none(getattr(info, "language_probability", None)),
        "segments": len(segments),
        "words": sum(len(segment.words) for segment in segments),
        "threads": thread_count,
        "windows": len(windows) if windows is not None else None,
        "windows_s": float(sum(end - start for start, end in windows)) if windows is not None else None,
        "clips": len(clips) if clips is not None else None,
        "clipped_s": float(sum(end - start for start, end in clips)) if clips is not None else None,
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
        settings={"device": device, "device_index": int(device_index), "compute_type": compute_type},
    )
