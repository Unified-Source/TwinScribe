"""Whisper through sherpa-onnx, for machines where CTranslate2 is not available (Windows on
ARM today) and for the CUDA or CoreML providers of onnxruntime.

The same detector role as the CTranslate2 wrapper: never published, it marks the spans where it
heard speech and the published engine heard nothing. Whisper decodes at most thirty seconds
per stream, so the recording is cut into windows of up to thirty seconds at the quietest
moment before each limit, and each window is decoded on its own stream. The windows are
deliberately not the voice detector's utterances: the transducer decodes those, and a detector
that saw the same boundaries would fail in the same places. Word times come from token
timestamps when the export carries cross-attention outputs; otherwise the words of each
segment are spread inside the segment timestamps that Whisper's own timestamp tokens give,
clamped to the window and confined to the parts of the segment the voice detector marks as
speech (so that no word is placed in a pause), and the transcript records that its word times
are approximate.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from twinscribe.audio import SAMPLE_RATE, read_wav_mono16k
from twinscribe.engines.base import Segment, Transcript, Word, default_threads
from twinscribe.engines.parakeet import build_vad, group_words, import_sherpa_onnx, library_versions, segment_from_words
from twinscribe.engines.presets import PARAKEET_PRESETS, resolve_preset

ENGINE_NAME = "whisper_onnx"
TIMING_TOKEN = "token"
TIMING_SEGMENT = "segment"
MAX_STREAM_S = 30.0
DEFAULT_PRESET = "greedy"

# Whisper's decoder is greedy in this library; the preset names the windowing: windows of at
# most window_s seconds, cut at the quietest quiet_s stretch inside the last search_s seconds.
WHISPER_ONNX_PRESETS: dict[str, dict[str, Any]] = {
    DEFAULT_PRESET: {
        "decoding_method": "greedy_search",
        "language": "en",
        "task": "transcribe",
        "window_s": 30.0,
        "search_s": 6.0,
        "quiet_s": 0.3,
    },
}

# File name endings per model part, in order of preference; the exports name their files
# after the model, for example turbo-encoder.int8.onnx.
FILE_SUFFIXES: dict[str, tuple[str, ...]] = {
    "encoder": ("-encoder.int8.onnx", "-encoder.onnx"),
    "decoder": ("-decoder.int8.onnx", "-decoder.onnx"),
    "tokens": ("-tokens.txt",),
}

_HOP_S = 0.05
_MIN_WINDOW_FRACTION = 0.5


def resolve_model_files(model_dir: str | os.PathLike[str]) -> dict[str, Path]:
    """Locate encoder, decoder and tokens inside a sherpa-onnx Whisper export by their endings.

    Raises FileNotFoundError naming every part that has no candidate file.
    """
    directory = Path(model_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Whisper ONNX model directory not found: {directory}")
    names = sorted(p.name for p in directory.iterdir() if p.is_file())
    found: dict[str, Path] = {}
    missing: list[str] = []
    for part, suffixes in FILE_SUFFIXES.items():
        for suffix in suffixes:
            match = next((n for n in names if n.endswith(suffix)), None)
            if match is not None:
                found[part] = directory / match
                break
        else:
            missing.append(f"{part} (a file ending in {' or '.join(suffixes)})")
    if missing:
        raise FileNotFoundError(f"Whisper ONNX model directory {directory} lacks: {'; '.join(missing)}")
    return found


def plan_windows(
    samples: np.ndarray,
    rate: int = SAMPLE_RATE,
    window_s: float = 30.0,
    search_s: float = 6.0,
    quiet_s: float = 0.3,
) -> list[tuple[int, int]]:
    """Contiguous windows of at most window_s seconds, each cut at the quietest stretch.

    From a window's start, the cut is placed at the centre of the quietest quiet_s stretch
    (by root mean square, in steps of fifty milliseconds) inside the last search_s seconds
    before the limit, so that a word is not split; a cut is never placed before half the
    window. The last window ends at the end of the audio. Windows cover every sample exactly
    once, in order.
    """
    if window_s <= 0.0 or window_s > MAX_STREAM_S:
        raise ValueError(f"window_s must be positive and at most {MAX_STREAM_S}, got {window_s}")
    if search_s < 0.0 or quiet_s <= 0.0:
        raise ValueError("search_s must not be negative and quiet_s must be positive")
    total = int(len(samples))
    if total == 0:
        return []
    limit = max(1, int(round(window_s * rate)))
    search = int(round(min(search_s, window_s * (1.0 - _MIN_WINDOW_FRACTION)) * rate))
    quiet = max(1, int(round(quiet_s * rate)))
    hop = max(1, int(round(_HOP_S * rate)))
    data = np.asarray(samples, dtype=np.float32)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(total, start + limit)
        if end < total and search > 0:
            region_start = max(start + limit // 2, end - search)
            best_cut = end
            best_level = float("inf")
            position = region_start
            while position + quiet <= end:
                frame = data[position : position + quiet]
                level = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
                if level < best_level:
                    best_level = level
                    best_cut = position + quiet // 2
                position += hop
            end = max(start + 1, min(best_cut, end))
        windows.append((start, end))
        start = end
    return windows


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def speech_regions(
    sherpa_onnx: Any,
    samples: np.ndarray,
    vad_model_path: Path,
    threads: int,
) -> list[tuple[float, float]]:
    """Speech regions of the whole recording in seconds, from the Silero voice detector.

    The detector runs with the transducer's settings but its regions are used only to place
    words, never to choose what is decoded.
    """
    settings = PARAKEET_PRESETS["vad"]
    vad = build_vad(sherpa_onnx, vad_model_path, settings, threads)
    window = int(settings["window_size"])
    regions: list[tuple[float, float]] = []

    def drain() -> None:
        while not vad.empty():
            front = vad.front
            start = float(front.start) / SAMPLE_RATE
            end = start + len(front.samples) / float(SAMPLE_RATE)
            vad.pop()
            regions.append((start, end))

    position = 0
    total = len(samples)
    while position + window <= total:
        vad.accept_waveform(samples[position : position + window])
        position += window
        drain()
    if position < total:
        tail = np.zeros(window, dtype=np.float32)
        tail[: total - position] = samples[position:]
        vad.accept_waveform(tail)
        drain()
    vad.flush()
    drain()
    regions.sort()
    return regions


def _voiced_parts(begin: float, end: float, voiced: Sequence[tuple[float, float]] | None) -> list[tuple[float, float]]:
    """The parts of [begin, end] that lie in voiced regions, or the whole span when none does."""
    if not voiced:
        return [(begin, end)]
    parts = [(max(begin, s), min(end, e)) for s, e in voiced if s < end and e > begin]
    parts = [(s, e) for s, e in parts if e > s]
    return parts if parts else [(begin, end)]


def _spread(tokens: Sequence[str], parts: Sequence[tuple[float, float]]) -> list[Word]:
    """Spread tokens over the concatenation of the parts, weighted by character count."""
    lengths = [e - s for s, e in parts]
    total_len = float(sum(lengths))
    weights = [len(token) + 1 for token in tokens]
    total_weight = float(sum(weights))

    def to_real(virtual: float, for_start: bool) -> float:
        # A position exactly on a boundary belongs to the next part when it starts a word and
        # to the earlier part when it ends one, so a word never straddles a pause.
        remaining = virtual
        for index, ((start, _), length) in enumerate(zip(parts, lengths)):
            last = index == len(parts) - 1
            if remaining < length or (remaining <= length and (not for_start or last)):
                return start + remaining
            remaining -= length
        return parts[-1][1]

    words: list[Word] = []
    cursor = 0.0
    for token, weight in zip(tokens, weights):
        length = total_len * weight / total_weight
        start = to_real(cursor, True)
        end = to_real(cursor + length, False)
        words.append(Word(text=token, start=start, end=max(end, start), prob=None))
        cursor += length
    return words


def words_from_segments(
    texts: Sequence[str],
    starts: Sequence[float],
    durations: Sequence[float],
    window_len_s: float,
    offset: float = 0.0,
    voiced: Sequence[tuple[float, float]] | None = None,
) -> list[Word]:
    """Spread the words of each Whisper segment inside its span, weighted by character count.

    Segment spans are clamped to the window, because a timestamp token can point past the
    end of a short window. When the timing lists do not match the texts, the whole text is
    spread over the whole window. Times are shifted by offset (the position of the window in
    the recording). voiced, when given, holds the speech regions of the recording in absolute
    seconds; the words of a segment are then confined to the parts of its span that are
    voiced, so that a pause between sentences receives no word.
    """
    if not texts:
        return []
    if len(starts) != len(texts) or len(durations) != len(texts):
        spans: list[tuple[str, float, float]] = [(" ".join(texts), 0.0, max(0.0, window_len_s))]
    else:
        spans = []
        for index, (text, start, duration) in enumerate(zip(texts, starts, durations)):
            begin = _clamp(float(start), 0.0, window_len_s)
            end = _clamp(float(start) + float(duration), begin, window_len_s)
            if end <= begin:
                following = float(starts[index + 1]) if index + 1 < len(starts) else window_len_s
                end = _clamp(max(following, begin), begin, window_len_s)
            spans.append((text, begin, end))
    words: list[Word] = []
    for text, begin, end in spans:
        tokens = text.split()
        if not tokens:
            continue
        parts = _voiced_parts(begin + offset, end + offset, voiced)
        words.extend(_spread(tokens, parts))
    return words


def decode_window(
    recognizer: Any,
    samples: np.ndarray,
    offset_s: float,
    voiced: Sequence[tuple[float, float]] | None = None,
) -> tuple[list[Word], bool]:
    """Decode one window; returns its words and whether token timestamps were available."""
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    recognizer.decode_stream(stream)
    result = stream.result
    length_s = len(samples) / float(SAMPLE_RATE)
    tokens = [str(t) for t in result.tokens]
    stamps = [float(t) for t in result.timestamps]
    if tokens and len(stamps) == len(tokens):
        return group_words(tokens, stamps, segment_end=length_s, offset=offset_s), True
    texts = [str(t) for t in getattr(result, "segment_texts", [])]
    if not texts and str(result.text).strip():
        texts = [str(result.text)]
    starts = [float(t) for t in getattr(result, "segment_timestamps", [])]
    durations = [float(t) for t in getattr(result, "segment_durations", [])]
    return words_from_segments(texts, starts, durations, length_s, offset_s, voiced), False


def _build_recognizer(
    sherpa_onnx: Any,
    files: Mapping[str, Path],
    settings: Mapping[str, Any],
    threads: int,
    provider: str,
    token_timestamps: bool,
) -> Any:
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(files["encoder"]),
        decoder=str(files["decoder"]),
        tokens=str(files["tokens"]),
        language=str(settings.get("language", "en")),
        task=str(settings.get("task", "transcribe")),
        num_threads=threads,
        decoding_method=str(settings.get("decoding_method", "greedy_search")),
        debug=False,
        provider=provider,
        tail_paddings=-1,
        enable_token_timestamps=token_timestamps,
        enable_segment_timestamps=True,
    )


def transcribe(
    audio_path: str | os.PathLike[str],
    model_dir: str | os.PathLike[str],
    vad_model_path: str | os.PathLike[str] | None = None,
    preset: str | Mapping[str, Any] = DEFAULT_PRESET,
    threads: int | None = None,
    progress: Callable[[float], None] | None = None,
    provider: str = "cpu",
    on_segment: Callable[[Segment], None] | None = None,
) -> Transcript:
    """Transcribe one 16 kHz mono WAV with a sherpa-onnx Whisper export.

    The recording is cut into windows by plan_windows and each window decoded on its own
    stream. The recogniser is first built asking for token timestamps; if the first decoded
    window returns none (the export has no cross-attention outputs) it is built again without
    them, so the library does not warn on every window, and every word is placed by segment,
    confined to the speech regions the voice detector finds when vad_model_path is given.
    The transcript's settings record which timing was used. progress, when given, is called
    after every window with the fraction of the audio reached; an exception raised inside it
    propagates and abandons the run, which is how a caller cancels. provider names the
    onnxruntime execution provider ("cpu", "cuda" with the CUDA build, "coreml" on macOS).
    """
    audio = Path(audio_path)
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")
    vad_path = Path(vad_model_path) if vad_model_path is not None else None
    if vad_path is not None and not vad_path.is_file():
        raise FileNotFoundError(f"voice detector model not found: {vad_path}")
    files = resolve_model_files(model_dir)
    preset_name, settings = resolve_preset(preset, WHISPER_ONNX_PRESETS, "whisper_onnx")
    thread_count = default_threads(threads)

    samples = read_wav_mono16k(audio)
    audio_s = len(samples) / float(SAMPLE_RATE)
    windows = plan_windows(
        samples,
        SAMPLE_RATE,
        float(settings.get("window_s", 30.0)),
        float(settings.get("search_s", 6.0)),
        float(settings.get("quiet_s", 0.3)),
    )

    if provider == "cuda":
        from twinscribe.hardware import register_cuda_libraries

        register_cuda_libraries()
    sherpa_onnx = import_sherpa_onnx()

    load_start = time.perf_counter()
    recognizer = _build_recognizer(sherpa_onnx, files, settings, thread_count, provider, token_timestamps=True)
    load_s = time.perf_counter() - load_start
    if progress is not None:
        progress(0.0)

    voiced: list[tuple[float, float]] | None = None
    vad_s = 0.0
    if vad_path is not None:
        vad_start = time.perf_counter()
        voiced = speech_regions(sherpa_onnx, samples, vad_path, thread_count)
        vad_s = time.perf_counter() - vad_start

    segments: list[Segment] = []
    timing: str | None = None
    transcribe_start = time.perf_counter()
    for index, (start, end) in enumerate(windows):
        start_s = start / float(SAMPLE_RATE)
        chunk = samples[start:end]
        words, by_token = decode_window(recognizer, chunk, start_s, voiced)
        if timing is None:
            timing = TIMING_TOKEN if by_token else TIMING_SEGMENT
            if not by_token:
                rebuild_start = time.perf_counter()
                recognizer = _build_recognizer(
                    sherpa_onnx, files, settings, thread_count, provider, token_timestamps=False
                )
                load_s += time.perf_counter() - rebuild_start
                words, _ = decode_window(recognizer, chunk, start_s, voiced)
        segment = segment_from_words(words, start_s, end / float(SAMPLE_RATE))
        segments.append(segment)
        if on_segment is not None:
            on_segment(segment)
        if progress is not None:
            progress(min(1.0, end / float(len(samples))))
    if progress is not None and not windows:
        progress(1.0)
    transcribe_s = time.perf_counter() - transcribe_start

    resolved_timing = timing or TIMING_SEGMENT
    extras: dict[str, float | int | None] = {
        "windows": len(windows),
        "empty_windows": sum(1 for segment in segments if not segment.words),
        "words": sum(len(segment.words) for segment in segments),
        "threads": thread_count,
        "token_timestamps": 1 if resolved_timing == TIMING_TOKEN else 0,
        "vad_s": vad_s,
        "voiced_regions": None if voiced is None else len(voiced),
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
        settings={
            "provider": provider,
            "word_timing": resolved_timing,
            "language": str(settings.get("language", "en")),
            "task": str(settings.get("task", "transcribe")),
            "decoding_method": str(settings.get("decoding_method", "greedy_search")),
            "window_s": float(settings.get("window_s", 30.0)),
        },
    )
