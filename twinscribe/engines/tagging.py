"""Audio tagging through sherpa-onnx: for each region of one recording, the sound classes an
AudioSet-trained model hears there with their probabilities.

The model is small and runs on the processor in a few milliseconds per region; it decides
nothing by itself. The scene pass (`twinscribe/scenes.py`) turns its events into silence,
music, noise and sound markers, and into the decision to set aside words the published engine
wrote where there was no speech.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from twinscribe.audio import SAMPLE_RATE, read_wav_mono16k
from twinscribe.engines.base import default_threads
from twinscribe.engines.parakeet import import_sherpa_onnx, library_versions
from twinscribe.scenes import Event

ENGINE_NAME = "ced_audio_tagging"
MODEL_FILE = "model.int8.onnx"
LABELS_FILE = "class_labels_indices.csv"
DEFAULT_TOP_K = 8
# A region shorter than this is not tagged; the model's front end needs a few frames.
MIN_REGION_S = 0.1
REPORT_EVERY = 20


@dataclass(frozen=True)
class Tagging:
    """The tagger's events per region, with timings and what it was asked."""

    engine: str
    model: str
    preset: str
    regions: tuple[tuple[float, float], ...]
    events: tuple[tuple[Event, ...], ...]
    load_s: float
    tag_s: float
    versions: dict[str, str]
    settings: dict[str, object]


def resolve_model_files(model_dir: str | os.PathLike[str]) -> dict[str, Path]:
    """The model and its class labels inside a sherpa-onnx audio tagging folder."""
    directory = Path(model_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"audio tagging model directory not found: {directory}")
    files = {"model": directory / MODEL_FILE, "labels": directory / LABELS_FILE}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"audio tagging model directory {directory} lacks: {', '.join(missing)}")
    return files


def _build_tagger(sherpa_onnx: Any, files: dict[str, Path], threads: int, top_k: int, provider: str) -> Any:
    config = sherpa_onnx.AudioTaggingConfig(
        model=sherpa_onnx.AudioTaggingModelConfig(
            ced=str(files["model"]),
            num_threads=threads,
            debug=False,
            provider=provider,
        ),
        labels=str(files["labels"]),
        top_k=top_k,
    )
    if not config.validate():
        raise RuntimeError("audio tagging configuration failed validation; check the model files")
    return sherpa_onnx.AudioTagging(config)


def tag_regions(
    audio_path: str | os.PathLike[str],
    model_dir: str | os.PathLike[str],
    regions: Sequence[tuple[float, float]],
    threads: int | None = None,
    provider: str = "cpu",
    top_k: int = DEFAULT_TOP_K,
    progress: Callable[[float], None] | None = None,
) -> Tagging:
    """Tag every region (start and end in seconds) of one 16 kHz mono WAV.

    The events of a region are the top_k classes by probability, in descending order; a
    region too short to tag yields no events. progress, when given, is called with the
    fraction of regions done every few regions and once with 1.0 at the end.
    """
    audio = Path(audio_path)
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")
    files = resolve_model_files(model_dir)
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    thread_count = default_threads(threads)
    samples = read_wav_mono16k(audio)
    sherpa_onnx = import_sherpa_onnx()

    load_start = time.perf_counter()
    tagger = _build_tagger(sherpa_onnx, files, thread_count, top_k, provider)
    load_s = time.perf_counter() - load_start

    tag_start = time.perf_counter()
    events: list[tuple[Event, ...]] = []
    for index, (start, end) in enumerate(regions):
        lo = max(0, int(round(float(start) * SAMPLE_RATE)))
        hi = min(len(samples), int(round(float(end) * SAMPLE_RATE)))
        if hi - lo < int(MIN_REGION_S * SAMPLE_RATE):
            events.append(())
        else:
            stream = tagger.create_stream()
            stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(samples[lo:hi], dtype=np.float32))
            found = tagger.compute(stream)
            events.append(tuple(Event(name=str(e.name), prob=float(e.prob)) for e in found))
        if progress is not None and (index + 1) % REPORT_EVERY == 0:
            progress((index + 1) / len(regions))
    tag_s = time.perf_counter() - tag_start
    if progress is not None:
        progress(1.0)

    return Tagging(
        engine=ENGINE_NAME,
        model=Path(model_dir).name,
        preset=f"top-{top_k}",
        regions=tuple((float(s), float(e)) for s, e in regions),
        events=tuple(events),
        load_s=load_s,
        tag_s=tag_s,
        versions=library_versions(sherpa_onnx),
        settings={"provider": provider, "top_k": top_k, "threads": thread_count},
    )
