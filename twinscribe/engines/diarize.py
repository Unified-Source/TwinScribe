"""Speaker labelling through sherpa-onnx: pyannote segmentation for speech regions, a speaker
embedding extractor, and clustering into labels.

Clustering by distance threshold is the default. Forcing a speaker count is an explicit
opt-in because it can hide a failure: asked for four speakers, the clusterer can satisfy the
count with a degenerate cluster a few seconds long while two real speakers are absorbed into
other labels, and a duration-weighted error rate barely moves. With a threshold, a missing
speaker shows up as a missing label, which a per-speaker word count makes visible.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from twinscribe.audio import SAMPLE_RATE, read_wav_mono16k
from twinscribe.engines.base import Diarization, SpeakerTurn, default_threads
from twinscribe.engines.parakeet import import_sherpa_onnx, library_versions

ENGINE_NAME = "sherpa_diarization"
LABEL_PREFIX = "speaker_"
MIN_DURATION_ON_S = 0.3
MIN_DURATION_OFF_S = 0.5
THRESHOLD_CLUSTERING = -1


def label_for(speaker_index: int) -> str:
    """Stable label for a cluster index: speaker_00, speaker_01, and so on."""
    if speaker_index < 0:
        raise ValueError(f"speaker index must be non-negative, got {speaker_index}")
    return f"{LABEL_PREFIX}{int(speaker_index):02d}"


def turns_from_result(records: Iterable[Any]) -> tuple[SpeakerTurn, ...]:
    """Convert library segments (objects with start, end and speaker) into SpeakerTurn records.

    Turns are ordered by start time, then by end, then by label, so that the output is the
    same regardless of the order the library returns them in.
    """
    turns = [
        SpeakerTurn(start=float(r.start), end=float(r.end), label=label_for(int(r.speaker)))
        for r in records
    ]
    turns.sort(key=lambda turn: (turn.start, turn.end, turn.label))
    return tuple(turns)


def seconds_per_label(turns: Iterable[SpeakerTurn]) -> dict[str, float]:
    """Total labelled seconds for each label, keyed in order of first appearance."""
    totals: dict[str, float] = {}
    for turn in turns:
        totals[turn.label] = totals.get(turn.label, 0.0) + turn.duration
    return totals


def _check_model_file(path: str | os.PathLike[str], role: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} model not found: {resolved}")
    return resolved


def _build_config(
    sherpa_onnx: Any,
    segmentation_model: Path,
    embedding_model: Path,
    threads: int,
    num_clusters: int,
    threshold: float,
    provider: str = "cpu",
) -> Any:
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation_model)
            ),
            num_threads=threads,
            debug=False,
            provider=provider,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding_model),
            num_threads=threads,
            debug=False,
            provider=provider,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters,
            threshold=threshold,
        ),
        min_duration_on=MIN_DURATION_ON_S,
        min_duration_off=MIN_DURATION_OFF_S,
    )
    if not config.validate():
        raise RuntimeError("speaker diarization configuration failed validation; check the model files")
    return config


def diarize(
    audio_path: str | os.PathLike[str],
    segmentation_model: str | os.PathLike[str],
    embedding_model: str | os.PathLike[str],
    threads: int | None = None,
    num_speakers: int | None = None,
    threshold: float = 0.5,
    provider: str = "cpu",
) -> Diarization:
    """Label speakers in one 16 kHz mono WAV.

    num_speakers=None (the default) clusters by threshold, so a speaker the embeddings
    cannot separate is missing from the output rather than hidden inside a forced count.
    Pass num_speakers only when the count is known and the consequence described in the
    module docstring is accepted. A smaller threshold yields more clusters. provider names
    the onnxruntime execution provider for both models ("cpu", or "cuda" with the CUDA build
    of the library).
    """
    audio = Path(audio_path)
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")
    segmentation = _check_model_file(segmentation_model, "segmentation")
    embedding = _check_model_file(embedding_model, "embedding")
    if num_speakers is not None and num_speakers < 1:
        raise ValueError(f"num_speakers must be at least 1 when given, got {num_speakers}")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    thread_count = default_threads(threads)
    num_clusters = THRESHOLD_CLUSTERING if num_speakers is None else int(num_speakers)

    samples = read_wav_mono16k(audio)
    audio_s = len(samples) / float(SAMPLE_RATE)

    if provider == "cuda":
        from twinscribe.hardware import register_cuda_libraries

        register_cuda_libraries()
    sherpa_onnx = import_sherpa_onnx()

    load_start = time.perf_counter()
    config = _build_config(sherpa_onnx, segmentation, embedding, thread_count, num_clusters, threshold, provider)
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    load_s = time.perf_counter() - load_start
    expected_rate = int(diarizer.sample_rate)
    if expected_rate != SAMPLE_RATE:
        raise RuntimeError(
            f"diarizer expects {expected_rate} Hz audio but the pipeline supplies {SAMPLE_RATE} Hz"
        )

    diarize_start = time.perf_counter()
    result = diarizer.process(samples)
    records = result.sort_by_start_time()
    diarize_s = time.perf_counter() - diarize_start

    turns = turns_from_result(records)
    settings: dict[str, float | int | None] = {
        "num_speakers": num_speakers,
        "threshold": float(threshold),
        "min_duration_on_s": MIN_DURATION_ON_S,
        "min_duration_off_s": MIN_DURATION_OFF_S,
        "threads": thread_count,
        "labels_found": len({turn.label for turn in turns}),
        "provider": provider,
    }
    return Diarization(
        segments=turns,
        seconds_per_label=seconds_per_label(turns),
        audio_s=audio_s,
        load_s=load_s,
        diarize_s=diarize_s,
        versions=library_versions(sherpa_onnx),
        settings=settings,
    )
