"""Transducer engine (NVIDIA Parakeet TDT through sherpa-onnx) with Silero voice detection.

This is the published engine: a transducer emits a token only while consuming audio frames,
so it cannot write into silence. Long audio is carved into utterances by the voice detector
and each utterance is decoded on its own stream. Words are grouped from the recogniser's
tokens and timestamps here, not taken from its text field, because the two can differ by a
token.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from twinscribe.audio import SAMPLE_RATE, read_wav_mono16k
from twinscribe.engines.base import Segment, Transcript, Word, default_threads
from twinscribe.engines.presets import PARAKEET_PRESETS, resolve_preset

ENGINE_NAME = "parakeet_tdt"
MODEL_TYPE = "nemo_transducer"
WORD_MARKER = " "
FEATURE_DIM = 80

# Candidate file names per model part, in order of preference. The int8 export is preferred
# because it is the form the design measured; the fp32 names are accepted as a fallback.
MODEL_FILE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "encoder": ("encoder.int8.onnx", "encoder.onnx"),
    "decoder": ("decoder.int8.onnx", "decoder.onnx"),
    "joiner": ("joiner.int8.onnx", "joiner.onnx"),
    "tokens": ("tokens.txt",),
}

_dll_handles: list[Any] = []


def register_windows_dlls() -> tuple[str, ...]:
    """On Windows, add the library's binary directories to the process DLL search path.

    The compiled extension and the runtime it depends on ship inside the package's lib
    directory; registering that directory before import lets the loader find them. A no-op
    on every other platform, and when the package is not installed. Returns the directories
    registered. The directory handles are kept for the life of the process.
    """
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return ()
    spec = importlib.util.find_spec("sherpa_onnx")
    if spec is None or not spec.submodule_search_locations:
        return ()
    registered: list[str] = []
    for location in spec.submodule_search_locations:
        for candidate in (Path(location) / "lib", Path(location)):
            if candidate.is_dir() and any(candidate.glob("*.dll")):
                _dll_handles.append(os.add_dll_directory(str(candidate)))
                registered.append(str(candidate))
    return tuple(registered)


def import_sherpa_onnx() -> Any:
    """Import sherpa_onnx after registering its binaries on Windows."""
    register_windows_dlls()
    try:
        return importlib.import_module("sherpa_onnx")
    except ImportError as exc:
        raise ImportError(
            "sherpa-onnx is not installed; install the 'engines' extra to use this engine"
        ) from exc


def _call_or_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        result = value() if callable(value) else value
    except Exception:  # noqa: BLE001 - a version probe must never fail a run
        return None
    text = str(result).strip()
    return text or None


def library_versions(module: Any | None = None) -> dict[str, str]:
    """Versions of sherpa-onnx and its bundled onnxruntime, for the run record."""
    versions: dict[str, str] = {}
    if module is None:
        try:
            module = importlib.import_module("sherpa_onnx")
        except ImportError:
            return versions
    version = _call_or_str(getattr(module, "version", None))
    if version is None:
        try:
            version = importlib.metadata.version("sherpa-onnx")
        except importlib.metadata.PackageNotFoundError:
            version = None
    if version is not None:
        versions["sherpa_onnx"] = version
    for attribute, key in (
        ("onnxruntime_version", "onnxruntime"),
        ("git_sha1", "sherpa_onnx_git_sha1"),
    ):
        value = _call_or_str(getattr(module, attribute, None))
        if value is not None:
            versions[key] = value
    return versions


def resolve_model_files(model_dir: str | os.PathLike[str]) -> dict[str, Path]:
    """Locate encoder, decoder, joiner and tokens inside a sherpa-onnx transducer directory.

    Raises FileNotFoundError naming every part that has no candidate file.
    """
    directory = Path(model_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"transducer model directory not found: {directory}")
    found: dict[str, Path] = {}
    missing: list[str] = []
    for part, candidates in MODEL_FILE_CANDIDATES.items():
        for name in candidates:
            path = directory / name
            if path.is_file():
                found[part] = path
                break
        else:
            missing.append(f"{part} (one of {', '.join(candidates)})")
    if missing:
        raise FileNotFoundError(f"transducer model directory {directory} lacks: {'; '.join(missing)}")
    return found


def group_words(
    tokens: Sequence[str],
    timestamps: Sequence[float],
    segment_end: float,
    offset: float = 0.0,
) -> list[Word]:
    """Group per-token output into words using the leading-space word marker.

    A token that begins with the marker starts a new word; the marker is stripped. A token
    that is the marker alone starts an empty word, which is kept only if continuation
    tokens follow it and dropped otherwise. A token with no marker continues the current
    word, or starts one when no word is open. A word starts at its first token's time and
    ends at the next kept word's start, or at the segment end. Times are shifted by offset
    (the utterance's position in the recording). Missing trailing timestamps repeat the last
    known time so that a short timestamp list cannot lose tokens.
    """
    if not tokens:
        return []
    times = _padded_times(timestamps, len(tokens), fallback=0.0)
    groups: list[tuple[str, float]] = []
    current_text: str | None = None
    current_start = 0.0
    for token, stamp in zip(tokens, times):
        if token.startswith(WORD_MARKER) or current_text is None:
            if current_text:
                groups.append((current_text, current_start))
            current_text = token[len(WORD_MARKER):] if token.startswith(WORD_MARKER) else token
            current_start = float(stamp)
        else:
            current_text += token
    if current_text:
        groups.append((current_text, current_start))

    words: list[Word] = []
    for index, (text, start) in enumerate(groups):
        if index + 1 < len(groups):
            end = groups[index + 1][1]
        else:
            end = float(segment_end)
        end = max(end, start)
        words.append(Word(text=text, start=start + offset, end=end + offset, prob=None))
    return words


def _padded_times(timestamps: Sequence[float], count: int, fallback: float) -> list[float]:
    times = [float(t) for t in list(timestamps)[:count]]
    last = times[-1] if times else fallback
    while len(times) < count:
        times.append(last)
    return times


def segment_from_words(words: Sequence[Word], start: float, end: float) -> Segment:
    """Build a Segment whose text is the grouped words joined by single spaces."""
    return Segment(
        start=float(start),
        end=float(end),
        text=" ".join(word.text for word in words),
        words=tuple(words),
        quality={},
    )


def _build_recognizer(
    sherpa_onnx: Any, files: Mapping[str, Path], settings: Mapping[str, Any], threads: int, provider: str = "cpu"
) -> Any:
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(files["encoder"]),
        decoder=str(files["decoder"]),
        joiner=str(files["joiner"]),
        tokens=str(files["tokens"]),
        num_threads=threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=FEATURE_DIM,
        decoding_method=str(settings.get("decoding_method", "greedy_search")),
        model_type=MODEL_TYPE,
        provider=provider,
        debug=False,
    )


def build_vad(sherpa_onnx: Any, vad_model_path: Path, settings: Mapping[str, Any], threads: int) -> Any:
    """The Silero voice detector from a preset; always on the processor, where it is cheap."""
    return _build_vad(sherpa_onnx, vad_model_path, settings, threads)


def _build_vad(sherpa_onnx: Any, vad_model_path: Path, settings: Mapping[str, Any], threads: int) -> Any:
    silero = sherpa_onnx.SileroVadModelConfig(
        model=str(vad_model_path),
        threshold=float(settings["threshold"]),
        min_silence_duration=float(settings["min_silence_duration"]),
        min_speech_duration=float(settings["min_speech_duration"]),
        max_speech_duration=float(settings["max_speech_duration"]),
        window_size=int(settings["window_size"]),
    )
    config = sherpa_onnx.VadModelConfig(
        silero_vad=silero,
        sample_rate=SAMPLE_RATE,
        num_threads=threads,
        provider="cpu",
        debug=False,
    )
    # The internal buffer must hold the longest utterance the detector may emit with room to
    # spare; three times the maximum speech duration, never under sixty seconds.
    buffer_s = max(60.0, 3.0 * float(settings["max_speech_duration"]))
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=buffer_s)


def transcribe(
    audio_path: str | os.PathLike[str],
    model_dir: str | os.PathLike[str],
    vad_model_path: str | os.PathLike[str],
    preset: str | Mapping[str, Any],
    threads: int | None = None,
    progress: Callable[[float], None] | None = None,
    provider: str = "cpu",
) -> Transcript:
    """Transcribe one 16 kHz mono WAV with a local transducer export and Silero VAD.

    The waveform is fed to the detector in windows of the configured size; detected speech
    segments are drained after every window and again after the final flush, and each is
    decoded on its own recogniser stream. A trailing partial window is padded with zeros to a
    full window rather than discarded, so no audio at the end of a file is skipped silently.
    The load timer covers recogniser and detector construction; the transcription timer
    covers the whole feed-and-decode loop, with the decode share recorded in extras.
    progress, when given, is called at most a few hundred times per file with the fraction
    of the waveform fed so far, and once more with 1.0 after the final flush; an exception
    raised inside it propagates and abandons the run, which is how a caller cancels. provider
    names the onnxruntime execution provider for the recogniser ("cpu", or "cuda" with the
    CUDA build of the library, which falls back to the processor with a warning otherwise);
    the voice detector always runs on the processor.
    """
    audio = Path(audio_path)
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")
    vad_path = Path(vad_model_path)
    if not vad_path.is_file():
        raise FileNotFoundError(f"voice detector model not found: {vad_path}")
    files = resolve_model_files(model_dir)
    preset_name, settings = resolve_preset(preset, PARAKEET_PRESETS, "parakeet")
    thread_count = default_threads(threads)
    window = int(settings["window_size"])
    if window < 1:
        raise ValueError(f"window_size must be positive, got {window}")

    samples = read_wav_mono16k(audio)
    audio_s = len(samples) / float(SAMPLE_RATE)

    if provider == "cuda":
        from twinscribe.hardware import register_cuda_libraries

        register_cuda_libraries()
    sherpa_onnx = import_sherpa_onnx()

    load_start = time.perf_counter()
    recognizer = _build_recognizer(sherpa_onnx, files, settings, thread_count, provider)
    vad = _build_vad(sherpa_onnx, vad_path, settings, thread_count)
    load_s = time.perf_counter() - load_start

    segments: list[Segment] = []
    decode_s = 0.0
    speech_s = 0.0

    def drain() -> None:
        nonlocal decode_s, speech_s
        while not vad.empty():
            front = vad.front
            start_s = float(front.start) / SAMPLE_RATE
            utterance = np.asarray(front.samples, dtype=np.float32)
            vad.pop()
            length_s = len(utterance) / float(SAMPLE_RATE)
            speech_s += length_s
            decode_start = time.perf_counter()
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, utterance)
            recognizer.decode_stream(stream)
            result = stream.result
            decode_s += time.perf_counter() - decode_start
            words = group_words(
                list(result.tokens),
                list(result.timestamps),
                segment_end=length_s,
                offset=start_s,
            )
            segments.append(segment_from_words(words, start_s, start_s + length_s))

    transcribe_start = time.perf_counter()
    position = 0
    total = len(samples)
    report_every = max(window, total // 200)
    next_report = report_every
    while position + window <= total:
        vad.accept_waveform(samples[position : position + window])
        position += window
        drain()
        if progress is not None and position >= next_report:
            progress(min(1.0, position / total))
            next_report += report_every
    if position < total:
        tail = np.zeros(window, dtype=np.float32)
        tail[: total - position] = samples[position:]
        vad.accept_waveform(tail)
        drain()
    vad.flush()
    drain()
    if progress is not None:
        progress(1.0)
    transcribe_s = time.perf_counter() - transcribe_start

    extras: dict[str, float | int | None] = {
        "vad_segments": len(segments),
        "vad_speech_s": speech_s,
        "empty_segments": sum(1 for segment in segments if not segment.words),
        "words": sum(len(segment.words) for segment in segments),
        "decode_s": decode_s,
        "vad_s": max(0.0, transcribe_s - decode_s),
        "threads": thread_count,
    }
    return Transcript(
        engine=ENGINE_NAME,
        model=Path(model_dir).name,
        preset=preset_name,
        segments=tuple(segments),
        audio_s=audio_s,
        load_s=load_s,
        transcribe_s=transcribe_s,
        versions=library_versions(sherpa_onnx),
        extras=extras,
        settings={"provider": provider},
    )
