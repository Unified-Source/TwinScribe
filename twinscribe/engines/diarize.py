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
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from twinscribe.audio import SAMPLE_RATE, read_wav_mono16k
from twinscribe.engines.base import Diarization, SpeakerTurn, default_threads
from twinscribe.engines.parakeet import import_sherpa_onnx, library_versions

ENGINE_NAME = "sherpa_diarization"
LABEL_PREFIX = "speaker_"
MIN_DURATION_ON_S = 0.3
MIN_DURATION_OFF_S = 0.5
THRESHOLD_CLUSTERING = -1
# The clustering distance threshold: how far apart two voices must be for the clustering to
# keep them as two speakers; a smaller value yields more labels. The default was measured on
# the bench's two-speaker telephone calls, where it gave the lowest error rate, below a fixed
# count of two. A long recording, or a meeting with several voices, keeps each voice together
# only at the long value, where two similar voices on a short call already merge; the
# measurements are in docs/specs/batch_app.notes.md.
DEFAULT_THRESHOLD = 0.9
LONG_RECORDING_THRESHOLD = 1.2
# Clustering by threshold leaves a tail of labels holding a second or two each at every value
# short of merging real voices. A label holding less speech than this is folded into the large
# label whose voice is nearest. Three seconds is the largest floor that never folded a real
# speaker on the bench's telephone calls, where a caller's labelled speech can be four seconds
# long; the minimum is measured in the diarizer's own turns, which run shorter than the speech
# they cover. A label's voice is embedded from at most FOLD_EMBED_MAX_S of its longest turns.
FOLD_MIN_S = 3.0
FOLD_EMBED_MAX_S = 30.0


def label_for(speaker_index: int) -> str:
    """Stable label for a cluster index: speaker_00, speaker_01, and so on."""
    if speaker_index < 0:
        raise ValueError(f"speaker index must be non-negative, got {speaker_index}")
    return f"{LABEL_PREFIX}{int(speaker_index):02d}"


def turns_from_result(records: Iterable[Any]) -> tuple[SpeakerTurn, ...]:
    """Convert library segments (objects with start, end and speaker) into SpeakerTurn records.

    Turns are ordered by start time, then by end, then by label, so that the output is the
    same regardless of the order the library returns them in. The library's cluster numbers
    can have holes (a cluster it discarded keeps its number); they are renumbered by order of
    first appearance so that the labels read speaker_00, speaker_01 and so on.
    """
    ordered = sorted(list(records), key=lambda r: (float(r.start), float(r.end), int(r.speaker)))
    numbering: dict[int, int] = {}
    for record in ordered:
        numbering.setdefault(int(record.speaker), len(numbering))
    turns = [
        SpeakerTurn(start=float(r.start), end=float(r.end), label=label_for(numbering[int(r.speaker)]))
        for r in ordered
    ]
    turns.sort(key=lambda turn: (turn.start, turn.end, turn.label))
    return tuple(turns)


def seconds_per_label(turns: Iterable[SpeakerTurn]) -> dict[str, float]:
    """Total labelled seconds for each label, keyed in order of first appearance."""
    totals: dict[str, float] = {}
    for turn in turns:
        totals[turn.label] = totals.get(turn.label, 0.0) + turn.duration
    return totals


def renumber(turns: Iterable[SpeakerTurn]) -> tuple[SpeakerTurn, ...]:
    """The turns sorted by time with their labels renumbered by order of first appearance."""
    ordered = sorted(turns, key=lambda t: (t.start, t.end, t.label))
    numbering: dict[str, str] = {}
    for turn in ordered:
        numbering.setdefault(turn.label, label_for(len(numbering)))
    return tuple(SpeakerTurn(start=t.start, end=t.end, label=numbering[t.label]) for t in ordered)


def label_embedding(
    samples: np.ndarray,
    turns: Sequence[SpeakerTurn],
    embed: Callable[[np.ndarray], np.ndarray],
    sample_rate: int = SAMPLE_RATE,
    max_seconds: float = FOLD_EMBED_MAX_S,
) -> np.ndarray | None:
    """One unit-length voice embedding for a label, from at most max_seconds of its longest
    turns joined in that order; None when the turns hold no audio or the embedding is empty."""
    pieces: list[np.ndarray] = []
    budget = float(max_seconds)
    for turn in sorted(turns, key=lambda t: (-t.duration, t.start)):
        if budget <= 0.0:
            break
        first = max(0, int(round(turn.start * sample_rate)))
        last = min(len(samples), int(round(min(turn.end, turn.start + budget) * sample_rate)))
        if last <= first:
            continue
        pieces.append(samples[first:last])
        budget -= (last - first) / float(sample_rate)
    if not pieces:
        return None
    vector = np.asarray(embed(np.concatenate(pieces)), dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size == 0 or not np.isfinite(norm) or norm <= 0.0:
        return None
    return vector / norm


def fold_small_labels(
    samples: np.ndarray,
    turns: Sequence[SpeakerTurn],
    embed: Callable[[np.ndarray], np.ndarray],
    min_seconds: float,
    sample_rate: int = SAMPLE_RATE,
    max_embed_s: float = FOLD_EMBED_MAX_S,
) -> tuple[tuple[SpeakerTurn, ...], int]:
    """Fold every label holding less than min_seconds of speech into the large label (one
    holding min_seconds or more) whose voice is nearest: the cosine similarity of one embedding
    per label, each from at most max_embed_s of the label's longest turns. Returns the turns,
    renumbered by first appearance, and the number of labels folded. Nothing is folded when
    min_seconds is zero, when no label reaches it, or when every label does; a small label
    whose turns hold no audio keeps its label.
    """
    if min_seconds < 0.0:
        raise ValueError(f"min_seconds must not be negative, got {min_seconds}")
    seconds = seconds_per_label(turns)
    large = [label for label, total in seconds.items() if total >= min_seconds]
    small = [label for label in seconds if label not in large]
    if min_seconds <= 0.0 or not large or not small:
        return tuple(turns), 0
    by_label: dict[str, list[SpeakerTurn]] = {}
    for turn in turns:
        by_label.setdefault(turn.label, []).append(turn)
    voices: dict[str, np.ndarray] = {}
    for label in large:
        vector = label_embedding(samples, by_label[label], embed, sample_rate, max_embed_s)
        if vector is not None:
            voices[label] = vector
    if not voices:
        return tuple(turns), 0
    target: dict[str, str] = {}
    for label in small:
        vector = label_embedding(samples, by_label[label], embed, sample_rate, max_embed_s)
        if vector is None:
            continue
        target[label] = max(voices, key=lambda name: float(vector @ voices[name]))
    if not target:
        return tuple(turns), 0
    folded = [SpeakerTurn(start=t.start, end=t.end, label=target.get(t.label, t.label)) for t in turns]
    return renumber(folded), len(target)


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
    threshold: float = DEFAULT_THRESHOLD,
    provider: str = "cpu",
    fold_min_s: float = FOLD_MIN_S,
) -> Diarization:
    """Label speakers in one 16 kHz mono WAV.

    num_speakers=None (the default) clusters by threshold, so a speaker the embeddings
    cannot separate is missing from the output rather than hidden inside a forced count.
    Pass num_speakers only when the count is known and the consequence described in the
    module docstring is accepted. A smaller threshold yields more clusters; the default and
    the value for a long recording are the module constants. When clustering by threshold,
    labels holding less than fold_min_s of speech are folded into the nearest large label by
    voice (fold_small_labels); zero leaves every label, and a given count skips the fold.
    provider names the onnxruntime execution provider for the models ("cpu", or "cuda" with
    the CUDA build of the library).
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
    if fold_min_s < 0.0:
        raise ValueError(f"fold_min_s must not be negative, got {fold_min_s}")
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
    labels_before = len({turn.label for turn in turns})
    folded = 0
    fold_s = 0.0
    if num_speakers is None and fold_min_s > 0.0:
        fold_start = time.perf_counter()
        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(embedding), num_threads=thread_count, debug=False, provider=provider
            )
        )

        def embed(piece: np.ndarray) -> np.ndarray:
            stream = extractor.create_stream()
            stream.accept_waveform(SAMPLE_RATE, piece)
            stream.input_finished()
            return np.asarray(extractor.compute(stream), dtype=np.float32)

        turns, folded = fold_small_labels(samples, turns, embed, fold_min_s)
        fold_s = time.perf_counter() - fold_start
        diarize_s += fold_s
    settings: dict[str, float | int | None] = {
        "num_speakers": num_speakers,
        "threshold": float(threshold),
        "min_duration_on_s": MIN_DURATION_ON_S,
        "min_duration_off_s": MIN_DURATION_OFF_S,
        "threads": thread_count,
        "labels_found": len({turn.label for turn in turns}),
        "labels_before_fold": labels_before,
        "fold_min_s": float(fold_min_s) if num_speakers is None else None,
        "labels_folded": folded,
        "fold_s": round(fold_s, 3),
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


def diarize_in_child(
    audio_path: str | os.PathLike[str],
    segmentation_model: str | os.PathLike[str],
    embedding_model: str | os.PathLike[str],
    threads: int | None = None,
    num_speakers: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    provider: str = "cpu",
    fold_min_s: float = FOLD_MIN_S,
) -> Diarization:
    """`diarize` run in a child process: the library holds the interpreter lock for the whole
    call, so a window whose thread shares the process would stop answering for as long as the
    labelling takes."""
    from twinscribe.engines.isolate import run_in_child_process

    return run_in_child_process(
        diarize, os.fspath(audio_path), os.fspath(segmentation_model), os.fspath(embedding_model),
        threads=threads, num_speakers=num_speakers, threshold=threshold, provider=provider,
        fold_min_s=fold_min_s,
    )
