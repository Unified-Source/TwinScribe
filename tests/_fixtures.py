"""Synthetic fixtures shared by the pipeline, output and application tests: word lists with
known gaps, speaker turns, transcripts built from them, a fake model store and fake engines."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from twinscribe.engines.base import Diarization, Segment, SpeakerTurn, Transcript, Word
from twinscribe.scenes import Event, Scene, seconds_by_kind, words_outside
from twinscribe.hardware import ARCH_X86_64, OS_WINDOWS, Gpu, Libraries, Machine, Plan, make_plan
from twinscribe.labelling import build_lines, label_words
from twinscribe.models import CATALOGUE, ModelSet, find_models
from twinscribe.outputs.transcript_doc import build_document
from twinscribe.pipeline import Engines
from twinscribe.review import build_review

AUDIO_S = 30.0

# Published words. Gaps: 3.5 to 6.0, 7.6 to 12.0, 13.9 to 24.0, then words to 26.0.
PUBLISHED = [
    ("good", 0.5, 0.9), ("morning", 0.9, 1.3), ("this", 1.3, 1.8), ("is", 1.8, 2.2),
    ("the", 2.2, 2.6), ("first", 2.6, 3.0), ("call", 3.0, 3.5),
    ("you", 6.0, 6.4), ("can", 6.4, 6.8), ("start", 6.8, 7.2), ("now", 7.2, 7.6),
    ("thank", 12.0, 12.5), ("you", 12.5, 13.0), ("for", 13.0, 13.4), ("waiting", 13.4, 13.9),
    ("hold", 24.0, 24.5), ("on", 24.5, 25.0), ("goodbye", 25.0, 26.0),
]
# What the detector heard in addition: four words in the first gap, two in the third.
DETECTOR_EXTRA = [
    ("yes", 3.9, 4.2), ("I", 4.2, 4.4), ("am", 4.4, 4.7), ("here", 4.7, 5.2),
    ("all", 20.5, 20.9), ("right", 20.9, 21.4),
]
TURNS = [(0.0, 3.6, "speaker_00"), (5.8, 7.7, "speaker_01"), (11.9, 14.0, "speaker_00"), (23.9, 26.1, "speaker_01")]


def words(spec: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(text=t, start=s, end=e, prob=None) for t, s, e in spec]


def turns(spec: list[tuple[float, float, str]] = TURNS) -> tuple[SpeakerTurn, ...]:
    return tuple(SpeakerTurn(start=s, end=e, label=label) for s, e, label in spec)


SEGMENT_GAP_S = 1.0


def segments_from_words(ws: list[Word], gap_s: float = SEGMENT_GAP_S) -> tuple[Segment, ...]:
    """Consecutive words become one segment; a pause longer than gap_s starts the next, as the
    voice detector's utterances would."""
    segments: list[Segment] = []
    current: list[Word] = []
    for word in ws:
        if current and word.start - current[-1].end > gap_s:
            segments.append(Segment(current[0].start, current[-1].end, " ".join(w.text for w in current), tuple(current), {}))
            current = []
        current.append(word)
    if current:
        segments.append(Segment(current[0].start, current[-1].end, " ".join(w.text for w in current), tuple(current), {}))
    return tuple(segments)


def transcript(engine: str, model: str, preset: str, ws: list[Word], audio_s: float = AUDIO_S) -> Transcript:
    segments = segments_from_words(ws)
    return Transcript(
        engine=engine,
        model=model,
        preset=preset,
        segments=segments,
        audio_s=audio_s,
        load_s=0.2,
        transcribe_s=1.5,
        versions={f"{engine}_lib": "1.0"},
        extras={},
    )


def published_transcript(model: str = "parakeet-tdt-0.6b-v2-int8") -> Transcript:
    return transcript("parakeet_tdt", model, "vad", words(PUBLISHED))


def detector_transcript(model: str = "whisper-large-v3-turbo-ct2") -> Transcript:
    return transcript("whisper_ct2", model, "production", sorted(words(PUBLISHED + DETECTOR_EXTRA), key=lambda w: w.start))


def diarization(
    spec: list[tuple[float, float, str]] = TURNS, threshold: float = 0.5, num_speakers: int | None = None
) -> Diarization:
    result = turns(spec)
    seconds: dict[str, float] = {}
    for turn in result:
        seconds[turn.label] = seconds.get(turn.label, 0.0) + turn.duration
    return Diarization(
        segments=result,
        seconds_per_label=seconds,
        audio_s=AUDIO_S,
        load_s=0.1,
        diarize_s=0.9,
        versions={"sherpa_onnx": "1.13"},
        settings={"threshold": threshold, "num_speakers": num_speakers},
    )


def make_models(root: Path) -> ModelSet:
    """A model store with a one-byte stand-in for every required file of every catalogue entry."""
    for spec in CATALOGUE:
        folder = root / spec.key
        folder.mkdir(parents=True, exist_ok=True)
        for name in spec.required:
            (folder / name).write_bytes(b"x")
    return find_models(root)


def make_plan_for(
    ct2: bool = True,
    sherpa: bool = True,
    cuda: bool = False,
    sherpa_cuda: bool = False,
    preference: str = "auto",
) -> Plan:
    """A plan from a fictional machine: x86-64 Windows, 16 cores, optionally an NVIDIA device."""
    machine = Machine(os=OS_WINDOWS, arch=ARCH_X86_64, machine_raw="AMD64", cores=16, memory_gb=32.0, processor="test")
    versions: dict[str, str] = {}
    if sherpa:
        versions["sherpa-onnx"] = "1.13.7+cuda" if sherpa_cuda else "1.13.7"
    if ct2:
        versions["ctranslate2"] = "4.8.2"
        versions["faster-whisper"] = "1.2.1"
    libraries = Libraries(faster_whisper=ct2, ctranslate2=ct2, sherpa_onnx=sherpa, onnxruntime=sherpa, versions=versions)
    gpus = (Gpu(name="Test GPU", memory_mb=8192, driver="1.0"),) if cuda else ()
    return make_plan(
        machine, libraries, gpus, cuda_count=1 if cuda else 0, preference=preference,
        ct2_types_cpu=frozenset({"int8", "float32"}), ct2_types_cuda=frozenset({"float16"}),
    )


def make_engines(
    *,
    fail_publisher: bool = False,
    fail_diarizer: bool = False,
    no_diarizer: bool = False,
    calls: list[str] | None = None,
    kwargs_seen: dict[str, dict] | None = None,
    tagger_events: Callable[[tuple[float, float]], list[Event]] | None = None,
    no_tagger: bool = False,
) -> Engines:
    """Engines that return the fixtures above, report progress and optionally fail.

    calls records the order of engine calls; kwargs_seen records the keyword arguments each
    engine received, keyed by engine name. The fake tagger hears speech everywhere unless
    tagger_events maps a region to the events to return for it.
    """

    def note(name: str, kwargs: dict) -> None:
        if calls is not None:
            calls.append(name)
        if kwargs_seen is not None:
            kwargs_seen[name] = dict(kwargs)

    def publisher(path, model_dir, vad, preset, threads=None, progress=None, on_segment=None, **kwargs):
        note("publisher", kwargs)
        if fail_publisher:
            raise RuntimeError("publisher exploded")
        result = transcript("parakeet_tdt", Path(model_dir).name, preset, words(PUBLISHED))
        if progress is not None:
            progress(0.0)
        if on_segment is not None:
            for segment in result.segments:
                on_segment(segment)
        if progress is not None:
            progress(0.5)
            progress(1.0)
        return result

    def detector(path, model_dir, preset, threads=None, progress=None, on_segment=None, **kwargs):
        note("detector", kwargs)
        result = transcript("whisper_ct2", Path(model_dir).name, preset, sorted(words(PUBLISHED + DETECTOR_EXTRA), key=lambda w: w.start))
        if progress is not None:
            progress(0.0)
        if on_segment is not None:
            for segment in result.segments:
                on_segment(segment)
        if progress is not None:
            progress(0.3)
            progress(0.9)
        return result

    def detector_onnx(path, model_dir, vad, preset, threads=None, progress=None, on_segment=None, **kwargs):
        note("detector_onnx", kwargs)
        if progress is not None:
            progress(0.0)
            progress(0.5)
            progress(1.0)
        result = transcript("whisper_onnx", Path(model_dir).name, preset, sorted(words(PUBLISHED + DETECTOR_EXTRA), key=lambda w: w.start))
        return Transcript(
            engine=result.engine, model=result.model, preset=result.preset, segments=result.segments,
            audio_s=result.audio_s, load_s=result.load_s, transcribe_s=result.transcribe_s, versions=result.versions,
            extras=result.extras, settings={"provider": kwargs.get("provider", "cpu"), "word_timing": "segment"},
        )

    def diarizer(path, segmentation, embedding, threads=None, threshold=0.5, **kwargs):
        note("diarizer", kwargs)
        if fail_diarizer:
            raise RuntimeError("embedding model rejected")
        return diarization(threshold=threshold, num_speakers=kwargs.get("num_speakers"))

    def tagger(path, model_dir, regions, threads=None, provider="cpu", top_k=8, progress=None, **kwargs):
        note("tagger", kwargs)
        events = []
        for region in regions:
            if tagger_events is not None:
                events.append(tuple(tagger_events((float(region[0]), float(region[1])))))
            else:
                events.append((Event("Speech", 0.9),))
        if progress is not None:
            progress(1.0)
        return SimpleNamespace(
            engine="fake_tagging", model=Path(model_dir).name, preset=f"top-{top_k}", regions=tuple(regions),
            events=tuple(events), load_s=0.05, tag_s=0.1, versions={"tagger_lib": "1.0"},
            settings={"top_k": top_k, "provider": provider},
        )

    return Engines(
        publisher=publisher,
        detector=detector,
        diarizer=None if no_diarizer else diarizer,
        detector_onnx=detector_onnx,
        tagger=None if no_tagger else tagger,
    )


def make_document(
    source_name: str = "call.wav",
    with_speakers: bool = True,
    profile: str = "standard",
    scenes: tuple[Scene, ...] = (),
) -> dict[str, Any]:
    """A transcript document built from the fixtures, as the pipeline would build it; scenes,
    when given, are written in with their totals and the detector words inside them left out
    of the review list."""
    published = published_transcript()
    detector = detector_transcript()
    diar = diarization() if with_speakers else None
    labels = label_words(published.words, diar.segments if diar is not None else ())
    lines = build_lines(published.words, labels)
    detector_words = words_outside(detector.words, scenes)
    marks = build_review(published.words, detector_words, AUDIO_S)
    non_speech = {
        "seconds_by_kind": seconds_by_kind(scenes),
        "suppressed_publisher_words": 0,
        "suppressed_detector_words": len(detector.words) - len(detector_words),
        "tagged": bool(scenes),
    }
    return build_document(
        scenes=scenes,
        non_speech=non_speech,
        source_name=source_name,
        source_sha256="ab" * 32,
        source_bytes=960044,
        video=False,
        duration_s=AUDIO_S,
        profile=profile,
        publisher=published,
        detector=detector,
        diarization=diar,
        lines=lines,
        marks=marks,
        overview=[0, 50, 100, 50] * 300,
        produced_utc="2026-01-01T00:00:00+00:00",
    )
