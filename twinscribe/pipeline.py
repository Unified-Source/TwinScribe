"""Per-file pipeline and batch driver.

For one recording: digest it, decode it to 16 kHz mono, run the published engine, run the
detector, label speakers, build the review list, and write the outputs beside the recording:
the transcript document, plain text, Word, subtitles, the review set and the run record. A
batch runs the pipeline over many recordings, records every failure and carries on. Nothing
here touches the network; the engines load from local folders only.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from twinscribe import __version__, audio, load
from twinscribe.engines.base import Diarization, Segment, Transcript
from twinscribe.engines.presets import PARAKEET_PRESETS, WHISPER_PRESETS
from twinscribe.engines.whisper_onnx import DEFAULT_PRESET as ONNX_DETECTOR_PRESET
from twinscribe.engines.whisper_onnx import WHISPER_ONNX_PRESETS
from twinscribe.hardware import BACKEND_ONNX, DEVICE_AUTO, Plan, current_plan
from twinscribe.history import HISTORY_FILE, append_entry, entry_from_result, history_path
from twinscribe.labelling import DEFAULT_MIN_RUN_S, DEFAULT_MIN_RUN_WORDS, build_lines, label_words, smooth_labels
from twinscribe.models import KEY_AUDIO_TAGGER, KEY_EMBEDDING, KEY_SEGMENTATION, KEY_SILERO_VAD, ModelSet, spec_for
from twinscribe.outputs import render_all
from twinscribe.outputs.transcript_doc import build_document, overview_peaks, write_document
from twinscribe.paths import runs_dir, work_dir
from twinscribe.profiles import Profile, select as select_level
from twinscribe.review import build_review, review_set, write_review_set
from twinscribe.runrecord import Failure, RunRecord, utc_now, write_json_atomic
from twinscribe.scenes import Analysis, analyse, seconds_by_kind, without_segments, words_outside

BATCH_SCHEMA = "twinscribe.batch.v1"

AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma",
        ".aiff", ".aif", ".amr", ".mp2", ".wv", ".ape",
    }
)
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".wmv", ".webm", ".mpg", ".mpeg",
        ".3gp", ".ts", ".mts", ".m2ts", ".flv", ".asf", ".vob",
    }
)
MEDIA_EXTENSIONS: frozenset[str] = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# Stage weights for the overall progress figure; they sum to one.
STAGES: tuple[tuple[str, float], ...] = (
    ("digest", 0.02),
    ("decode", 0.05),
    ("publisher", 0.32),
    ("detector", 0.38),
    ("scenes", 0.03),
    ("speakers", 0.15),
    ("outputs", 0.05),
)
STAGE_TITLES: dict[str, str] = {
    "digest": "Reading the file",
    "decode": "Decoding the audio",
    "publisher": "Transcribing (published engine)",
    "detector": "Checking (second engine)",
    "scenes": "Marking silence, music and noise",
    "speakers": "Labelling speakers",
    "outputs": "Writing the outputs",
}
# What a stage says before its engine has reported once: the model is being loaded.
LOADING_TITLES: dict[str, str] = {
    "publisher": "Loading the published engine",
    "detector": "Loading the second engine",
    "speakers": "Loading the speaker models and labelling speakers",
}
PARALLEL_TITLE = "Transcribing and checking at the same time"
ETA_MIN_FRACTION = 0.05


class Cancelled(RuntimeError):
    """Raised inside a run when the cancel check of the caller returns True."""


@dataclass(frozen=True)
class Progress:
    """One progress report for a file.

    stage names the stage, fraction is the overall fraction of the file done, message is what
    is happening now, elapsed_s the seconds since the file started, and eta_s a smoothed
    estimate of the seconds left (None until enough has been done to estimate).
    """

    stage: str
    fraction: float
    message: str
    elapsed_s: float = 0.0
    eta_s: float | None = None


ProgressFn = Callable[[Progress], None]
CancelFn = Callable[[], bool]
PartialFn = Callable[[str, Segment], None]


@dataclass(frozen=True)
class OutputPaths:
    """Where the outputs of one recording go."""

    transcript: Path
    text: Path
    docx: Path
    subtitles: Path
    review: Path
    run: Path

    @property
    def all(self) -> tuple[Path, ...]:
        return (self.transcript, self.text, self.docx, self.subtitles, self.review, self.run)

    def as_dict(self) -> dict[str, str]:
        return {
            "transcript": str(self.transcript),
            "text": str(self.text),
            "docx": str(self.docx),
            "subtitles": str(self.subtitles),
            "review": str(self.review),
            "run": str(self.run),
        }


def has_sibling_with_same_stem(source: Path) -> bool:
    """True when another recording with the same stem sits beside `source` (call.mp3 by call.wav)."""
    try:
        entries = list(source.parent.iterdir())
    except OSError:
        return False
    stem = os.path.normcase(source.stem)
    name = os.path.normcase(source.name)
    return any(
        os.path.normcase(entry.name) != name
        and os.path.normcase(entry.stem) == stem
        and is_media(entry)
        and entry.is_file()
        for entry in entries
    )


def output_base(source: str | os.PathLike[str]) -> str:
    """The name the outputs of a recording share: its stem, or its full name when another
    recording with the same stem sits beside it, so that the two never overwrite each other."""
    src = Path(source)
    return src.name if has_sibling_with_same_stem(src) else src.stem


def output_paths(source: str | os.PathLike[str], out_dir: str | os.PathLike[str] | None = None) -> OutputPaths:
    """Output paths for a recording: beside it, or in out_dir when given, named by `output_base`."""
    src = Path(source)
    folder = Path(out_dir) if out_dir is not None else src.parent
    stem = output_base(src)
    return OutputPaths(
        transcript=folder / f"{stem}.transcript.json",
        text=folder / f"{stem}.txt",
        docx=folder / f"{stem}.docx",
        subtitles=folder / f"{stem}.srt",
        review=folder / f"{stem}.review.json",
        run=folder / f"{stem}.run.json",
    )


def is_media(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS


def is_video(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def discover_media(paths: Iterable[str | os.PathLike[str]], recursive: bool = True) -> list[Path]:
    """Media files among the given paths; folders are listed, recursively by default.

    The result is sorted and without duplicates.
    """
    found: dict[str, Path] = {}
    for given in paths:
        path = Path(given)
        if path.is_dir():
            candidates = path.rglob("*") if recursive else path.glob("*")
            for candidate in candidates:
                if candidate.is_file() and is_media(candidate):
                    found.setdefault(os.path.normcase(str(candidate.resolve())), candidate)
        elif path.is_file() and is_media(path):
            found.setdefault(os.path.normcase(str(path.resolve())), path)
    return sorted(found.values(), key=lambda p: (os.path.normcase(str(p.parent)), os.path.normcase(p.name)))


@dataclass(frozen=True)
class Job:
    """One recording to process, with everything the pipeline needs to know.

    plan is the acceleration plan (probed from the machine when None); detector names the
    detector model to run (resolved from the profile and the plan when None); preference is
    auto, cpu or cuda and matters only when the plan is probed here; speakers is the speaker
    count when it is known, an explicit opt-in (None clusters by threshold, so a speaker the
    models cannot separate is missing from the labels rather than hidden inside another).
    """

    source: Path
    profile: Profile
    models: ModelSet
    out_dir: Path | None = None
    threads: int | None = None
    author: str = ""
    keep_audio: bool = False
    work_folder: Path | None = None
    plan: Plan | None = None
    detector: str | None = None
    preference: str = DEVICE_AUTO
    speakers: int | None = None


@dataclass(frozen=True)
class Engines:
    """The engine calls, so that tests can stand synthetic engines in for the real ones.

    detector is the CTranslate2 detector, detector_onnx the sherpa-onnx one; tagger is the
    audio tagger behind the scene pass, used only when its model is present.
    """

    publisher: Callable[..., Transcript]
    detector: Callable[..., Transcript]
    diarizer: Callable[..., Diarization] | None
    detector_onnx: Callable[..., Transcript] | None = None
    tagger: Callable[..., Any] | None = None


def default_engines() -> Engines:
    """The real engine wrappers; their libraries import lazily when first called."""
    from twinscribe.engines import diarize, parakeet, tagging, whisper_ct2, whisper_onnx

    return Engines(
        publisher=parakeet.transcribe,
        detector=whisper_ct2.transcribe,
        diarizer=diarize.diarize,
        detector_onnx=whisper_onnx.transcribe,
        tagger=tagging.tag_regions,
    )


@dataclass(frozen=True)
class FileResult:
    """What one successful run produced."""

    source: Path
    outputs: OutputPaths
    document: dict[str, Any]
    run_record: dict[str, Any]
    elapsed_s: float

    @property
    def marks(self) -> int:
        return int(self.document.get("review", {}).get("marks", 0))

    @property
    def duration_s(self) -> float:
        return float(self.document.get("duration_s", 0.0))


class _Reporter:
    """Turns per-stage fractions into overall progress and checks for cancellation.

    In parallel mode the publisher and detector stages, which are adjacent, are reported as
    one combined stretch so that the overall fraction never runs backwards while the two
    engines report in turn.
    """

    def __init__(self, progress: ProgressFn | None, cancel: CancelFn | None) -> None:
        self._progress = progress
        self._cancel = cancel
        self._offsets: dict[str, tuple[float, float]] = {}
        self._parallel = False
        self._fractions: dict[str, float] = {"publisher": 0.0, "detector": 0.0}
        self._started = time.perf_counter()
        self._eta: float | None = None
        start = 0.0
        for name, weight in STAGES:
            self._offsets[name] = (start, weight)
            start += weight

    def set_parallel(self, parallel: bool) -> None:
        self._parallel = bool(parallel)

    def check(self) -> None:
        if self._cancel is not None and self._cancel():
            raise Cancelled("cancelled")

    def _estimate(self, overall: float, elapsed: float) -> float | None:
        """Seconds left, from the pace so far, smoothed; None until enough is done to say."""
        if overall < ETA_MIN_FRACTION or elapsed <= 0.0:
            return None
        raw = elapsed * (1.0 - overall) / overall
        self._eta = raw if self._eta is None else 0.7 * self._eta + 0.3 * raw
        return max(0.0, self._eta)

    def report(self, stage: str, fraction: float, message: str | None = None) -> None:
        self.check()
        clamped = min(1.0, max(0.0, fraction))
        if self._parallel and stage in self._fractions:
            self._fractions[stage] = clamped
            publisher_offset, publisher_weight = self._offsets["publisher"]
            _, detector_weight = self._offsets["detector"]
            combined = publisher_weight * self._fractions["publisher"] + detector_weight * self._fractions["detector"]
            overall = min(1.0, max(0.0, publisher_offset + combined))
            text = message or PARALLEL_TITLE
        else:
            offset, weight = self._offsets[stage]
            overall = min(1.0, max(0.0, offset + weight * clamped))
            text = message or STAGE_TITLES[stage]
        elapsed = time.perf_counter() - self._started
        if self._progress is not None:
            self._progress(
                Progress(stage=stage, fraction=overall, message=text, elapsed_s=elapsed, eta_s=self._estimate(overall, elapsed))
            )

    def stage_fn(self, stage: str) -> Callable[[float], None]:
        def inner(fraction: float) -> None:
            self.report(stage, fraction)

        return inner


def _decode_source(job: Job, reporter: _Reporter, digest: str) -> tuple[Path, bool]:
    """The 16 kHz mono WAV to feed the engines, and whether it is a temporary file."""
    source = job.source
    if source.suffix.lower() == ".wav":
        try:
            audio.duration_s(source)
            return source, False
        except audio.AudioFormatError:
            pass
    folder = job.work_folder if job.work_folder is not None else work_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{source.stem}.{digest[:12]}.16k.wav"
    reporter.report("decode", 0.0, f"Decoding the audio ({human_size(source.stat().st_size)} to read)")
    audio.decode_to_wav(source, target)
    reporter.report("decode", 1.0)
    return target, True


def human_size(size: int) -> str:
    """A file size for a message: 850 KB, 12.4 MB, 1.2 GB."""
    value = float(size)
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if value < 1000.0 or unit == "TB":
            return f"{int(value)} {unit}" if unit == "bytes" else f"{value:.1f} {unit}".replace(".0 ", " ")
        value /= 1000.0
    return f"{value:.1f} TB"


def _preset_settings(profile: Profile, backend: str | None = None) -> dict[str, Any]:
    if backend == BACKEND_ONNX:
        detector_preset = {ONNX_DETECTOR_PRESET: WHISPER_ONNX_PRESETS.get(ONNX_DETECTOR_PRESET)}
    else:
        detector_preset = {profile.detector_preset: WHISPER_PRESETS.get(profile.detector_preset)}
    return {
        "profile": profile.name,
        "publisher_preset": {profile.publisher_preset: PARAKEET_PRESETS.get(profile.publisher_preset)},
        "detector_preset": detector_preset,
        "review": {
            "min_silence_s": profile.review.min_silence_s,
            "min_detector_words": profile.review.min_detector_words,
            "pad_s": profile.review.pad_s,
        },
        "diarization_threshold": profile.diarization_threshold,
        "version": __version__,
    }


def process_file(
    job: Job,
    progress: ProgressFn | None = None,
    cancel: CancelFn | None = None,
    engines: Engines | None = None,
    on_partial: PartialFn | None = None,
) -> FileResult:
    """Run the whole pipeline over one recording and write its outputs.

    Speaker labelling that fails does not lose the transcript: the failure is recorded in the
    run record and the document, and every word is left unlabelled. Any other failure
    propagates after a run record naming it has been written beside the recording. on_partial,
    when given, receives every segment the published engine produces as it produces it, with
    the role "publisher"; the detector's segments are never passed on, because its text is
    never shown.
    """
    started_utc = utc_now()
    started = time.perf_counter()
    reporter = _Reporter(progress, cancel)
    engine_set = engines if engines is not None else default_engines()
    source = Path(job.source)
    if not source.is_file():
        raise FileNotFoundError(f"recording not found: {source}")
    paths = output_paths(source, job.out_dir)
    paths.transcript.parent.mkdir(parents=True, exist_ok=True)
    wav: Path | None = None
    temporary = False
    try:
        reporter.report("digest", 0.0)
        digest = audio.sha256_of(source, progress=reporter.stage_fn("digest"))
        size = source.stat().st_size
        reporter.report("digest", 1.0)

        wav, temporary = _decode_source(job, reporter, digest)
        models = job.models
        profile = job.profile

        plan = job.plan if job.plan is not None else current_plan(job.preference, job.threads)
        if job.detector is not None:
            detector_key = job.detector
            backend = spec_for(detector_key).backend
        else:
            selection = select_level(profile.name, models, plan.backends)
            detector_key, backend = selection.detector, selection.backend
        placement = plan.placement_for(backend)
        if placement is None:
            raise RuntimeError(f"no library for the {backend} detector backend is installed")
        parallel = plan.parallel_for(backend)
        vad_model = models.file(KEY_SILERO_VAD, "silero_vad.onnx")

        def publisher_segment(segment: Segment) -> None:
            if on_partial is not None:
                on_partial("publisher", segment)

        def run_publisher() -> Transcript:
            return engine_set.publisher(
                wav,
                models.path(profile.publisher),
                vad_model,
                profile.publisher_preset,
                threads=job.threads,
                progress=reporter.stage_fn("publisher"),
                provider=plan.publisher.provider,
                on_segment=publisher_segment if on_partial is not None else None,
            )

        def run_detector() -> Transcript:
            if backend == BACKEND_ONNX:
                if engine_set.detector_onnx is None:
                    raise RuntimeError("the ONNX detector is not available in this engine set")
                return engine_set.detector_onnx(
                    wav,
                    models.path(detector_key),
                    vad_model,
                    ONNX_DETECTOR_PRESET,
                    threads=job.threads,
                    progress=reporter.stage_fn("detector"),
                    provider=placement.provider,
                )
            return engine_set.detector(
                wav,
                models.path(detector_key),
                profile.detector_preset,
                threads=job.threads,
                progress=reporter.stage_fn("detector"),
                device=placement.device,
                compute_type=placement.compute_type or "auto",
                device_index=placement.index,
            )

        before = load.snapshot()
        if parallel:
            reporter.set_parallel(True)
            reporter.report("publisher", 0.0, "Loading both engines")
            with ThreadPoolExecutor(max_workers=2) as pool:
                publisher_future = pool.submit(run_publisher)
                detector_future = pool.submit(run_detector)
                published = publisher_future.result()
                detector = detector_future.result()
            reporter.set_parallel(False)
            reporter.report("detector", 1.0)
        else:
            reporter.report("publisher", 0.0, LOADING_TITLES["publisher"])
            published = run_publisher()
            reporter.report("publisher", 1.0)
            reporter.report("detector", 0.0, LOADING_TITLES["detector"])
            detector = run_detector()
            reporter.report("detector", 1.0)

        reporter.report("scenes", 0.0)
        samples = audio.read_wav_mono16k(wav)
        audio_s = float(published.audio_s) if published.audio_s > 0 else float(detector.audio_s)
        tagging_result: Any = None
        tag_fn: Callable[[Sequence[tuple[float, float]]], Any] | None = None
        if engine_set.tagger is not None and models.has(KEY_AUDIO_TAGGER):
            tagger_dir = models.path(KEY_AUDIO_TAGGER)

            def tag_with_model(regions: Sequence[tuple[float, float]]) -> Any:
                nonlocal tagging_result
                tagging_result = engine_set.tagger(
                    wav, tagger_dir, regions, threads=job.threads, progress=reporter.stage_fn("scenes")
                )
                return tagging_result.events

            tag_fn = tag_with_model
        analysis: Analysis = analyse(samples, published.segments, audio_s, tag_fn)
        published_kept = without_segments(published, analysis.suppressed)
        detector_words = words_outside(detector.words, analysis.scenes)
        reporter.report("scenes", 1.0)

        reporter.report("speakers", 0.0, LOADING_TITLES["speakers"])
        diarization: Diarization | None = None
        speaker_failure: str | None = None
        failures: list[Failure] = []
        if engine_set.diarizer is not None:
            try:
                diarization = engine_set.diarizer(
                    wav,
                    models.file(KEY_SEGMENTATION, "model.onnx"),
                    models.file(KEY_EMBEDDING, "nemo_en_titanet_large.onnx"),
                    threads=job.threads,
                    threshold=profile.diarization_threshold,
                    provider=plan.diarizer.provider,
                    **({"num_speakers": int(job.speakers)} if job.speakers is not None else {}),
                )
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - recorded; the transcript is still published
                speaker_failure = f"{type(exc).__name__}: {exc}"
                failures.append(Failure(str(source), type(exc).__name__, f"speaker labelling: {exc}"))
        after = load.snapshot()
        reporter.report("speakers", 0.6)

        words = published_kept.words
        labels = label_words(words, diarization.segments if diarization is not None else ())
        labels, smoothed_words = smooth_labels(words, labels, published_kept.segments)
        lines = build_lines(words, labels)
        marks = build_review(
            words,
            detector_words,
            audio_s,
            min_silence_s=profile.review.min_silence_s,
            min_detector_words=profile.review.min_detector_words,
            pad_s=profile.review.pad_s,
        )
        overview = overview_peaks(samples)
        del samples
        reporter.report("speakers", 1.0)

        reporter.report("outputs", 0.0)
        non_speech = {
            "seconds_by_kind": seconds_by_kind(analysis.scenes),
            "suppressed_publisher_words": sum(len(segment.words) for segment in analysis.suppressed),
            "suppressed_detector_words": len(detector.words) - len(detector_words),
            "tagged": analysis.tagged,
        }
        document = build_document(
            source_name=source.name,
            source_sha256=digest,
            source_bytes=size,
            video=is_video(source),
            output_base=paths.transcript.name[: -len(".transcript.json")],
            duration_s=audio_s,
            profile=profile.name,
            publisher=published,
            detector=detector,
            diarization=diarization,
            lines=lines,
            marks=marks,
            overview=overview,
            produced_utc=started_utc,
            speaker_failure=speaker_failure,
            scenes=analysis.scenes,
            non_speech=non_speech,
        )
        write_document(document, paths.transcript)
        render_all(document, paths.text, paths.docx, paths.subtitles, author=job.author)
        audio_reference = source.name if job.out_dir is None else str(source.resolve())
        write_review_set(review_set(published_kept, detector, audio_reference, marks), paths.review)
        reporter.report("outputs", 0.7)

        versions: dict[str, str] = {}
        versions.update(published.versions)
        versions.update(detector.versions)
        if diarization is not None:
            versions.update(diarization.versions)
        if tagging_result is not None:
            versions.update(tagging_result.versions)
        engine_facts = {
            "publisher": {"engine": published.engine, "model": published.model, "preset": published.preset},
            "detector": {"engine": detector.engine, "model": detector.model, "preset": detector.preset},
            "diarization": {
                "engine": "sherpa_diarization",
                "model": f"{KEY_SEGMENTATION} + {KEY_EMBEDDING}",
                "preset": f"threshold {profile.diarization_threshold}",
            },
        }
        if tagging_result is not None:
            engine_facts["tagging"] = {
                "engine": str(tagging_result.engine),
                "model": str(tagging_result.model),
                "preset": str(tagging_result.preset),
            }
        load_s = (
            published.load_s
            + detector.load_s
            + (diarization.load_s if diarization is not None else 0.0)
            + (float(tagging_result.load_s) if tagging_result is not None else 0.0)
        )
        transcribe_s = (
            published.transcribe_s
            + detector.transcribe_s
            + (diarization.diarize_s if diarization is not None else 0.0)
            + (float(tagging_result.tag_s) if tagging_result is not None else 0.0)
        )
        run_settings = _preset_settings(profile, backend) | {
            "threads": job.threads,
            "plan": plan.to_dict(),
            "detector_backend": backend,
            "detector_model": detector_key,
            "detector_word_timing": str(detector.settings.get("word_timing", "token")),
            "publisher_engine_settings": dict(published.settings),
            "detector_engine_settings": dict(detector.settings),
            "parallel_engines": parallel,
            "speakers": job.speakers,
            "labelling": {
                "smoothed_words": smoothed_words,
                "min_run_words": DEFAULT_MIN_RUN_WORDS,
                "min_run_s": DEFAULT_MIN_RUN_S,
            },
            "scenes": dict(analysis.settings)
            | dict(non_speech)
            | {
                "count": len(analysis.scenes),
                "regions_tagged": analysis.regions_tagged,
                "tagger_model": KEY_AUDIO_TAGGER if tagging_result is not None else None,
                "list": [
                    {"start": s.start, "end": s.end, "kind": s.kind, "label": s.label, "probability": s.probability}
                    for s in analysis.scenes
                ],
                "suppressed_utterances": [
                    {"start": u.start, "end": u.end, "words": len(u.words), "text": u.text} for u in analysis.suppressed
                ],
            },
        }
        record = RunRecord(
            engines=engine_facts,
            versions=versions,
            settings=run_settings,
            input_path=str(source),
            input_sha256=digest,
            audio_s=audio_s,
            load_s=load_s,
            transcribe_s=transcribe_s,
            started_utc=started_utc,
            ended_utc=utc_now(),
            load_verdict=load.verdict(before, after),
            failures=tuple(failures),
        )
        record.write(paths.run)
        reporter.report("outputs", 1.0, "Done")
        return FileResult(
            source=source,
            outputs=paths,
            document=document,
            run_record=record.to_dict(),
            elapsed_s=time.perf_counter() - started,
        )
    except Cancelled:
        raise
    except Exception as exc:
        _write_failure_record(job, paths, started_utc, exc)
        raise
    finally:
        if wav is not None and temporary and not job.keep_audio:
            try:
                wav.unlink()
            except OSError:
                pass


def _write_failure_record(job: Job, paths: OutputPaths, started_utc: str, exc: BaseException) -> None:
    """Best effort: leave a run record naming the failure beside the recording."""
    try:
        digest = audio.sha256_of(job.source) if job.source.is_file() else ""
    except OSError:
        digest = ""
    try:
        record = RunRecord(
            engines={},
            versions={},
            settings=_preset_settings(job.profile) | {"threads": job.threads},
            input_path=str(job.source),
            input_sha256=digest,
            audio_s=0.0,
            load_s=0.0,
            transcribe_s=0.0,
            started_utc=started_utc,
            ended_utc=utc_now(),
            load_verdict=None,
            failures=(Failure.from_exception(str(job.source), exc),),
        )
        record.write(paths.run)
    except Exception:  # noqa: BLE001 - the original error is what the caller must see
        pass


@dataclass
class Outcome:
    """How one recording of a batch fared."""

    source: Path
    ok: bool
    result: FileResult | None = None
    error_class: str | None = None
    error: str | None = None
    elapsed_s: float = 0.0
    cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"path": str(self.source), "ok": self.ok, "elapsed_s": self.elapsed_s}
        if self.result is not None:
            entry["outputs"] = self.result.outputs.as_dict()
            entry["duration_s"] = self.result.duration_s
            entry["marks"] = self.result.marks
        if self.error is not None:
            entry["error_class"] = self.error_class
            entry["error"] = self.error
        if self.cancelled:
            entry["cancelled"] = True
        return entry


BatchProgressFn = Callable[[int, int, Progress], None]
BatchPartialFn = Callable[[int, str, Segment], None]
OutcomeFn = Callable[[int, "Outcome"], None]


@dataclass
class BatchResult:
    """Every outcome of a batch and where its record was written."""

    outcomes: list[Outcome] = field(default_factory=list)
    record_path: Path | None = None

    @property
    def failures(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.ok and not o.cancelled]

    @property
    def completed(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.ok]


def run_batch(
    sources: Sequence[str | os.PathLike[str]],
    profile: Profile,
    models: ModelSet,
    out_dir: str | os.PathLike[str] | None = None,
    threads: int | None = None,
    author: str = "",
    keep_audio: bool = False,
    progress: BatchProgressFn | None = None,
    cancel: CancelFn | None = None,
    engines: Engines | None = None,
    record_dir: str | os.PathLike[str] | None = None,
    on_outcome: OutcomeFn | None = None,
    plan: Plan | None = None,
    preference: str = DEVICE_AUTO,
    on_partial: BatchPartialFn | None = None,
    speakers: int | None = None,
    history_path: str | os.PathLike[str] | None = None,
) -> BatchResult:
    """Process recordings one after another, never stopping for a failure.

    A cancellation stops the batch; the recordings not reached are recorded as cancelled. The
    batch record lists every recording with its outputs or its error. on_outcome, when given,
    is called with the index and the outcome as soon as each recording finishes; on_partial
    with the index, the role and each segment the published engine produces. The acceleration
    plan is probed once for the batch when not given. speakers is the speaker count when it is
    known, applied to every recording of the batch; None clusters by threshold.
    """
    started_utc = utc_now()
    result = BatchResult()
    total = len(sources)
    stopped = False
    batch_plan = plan if plan is not None else current_plan(preference, threads)
    # The history sits beside the batch records: under the given folder's parent, else the home.
    if history_path is not None:
        history_target = Path(history_path)
    elif record_dir is not None:
        history_target = Path(record_dir).parent / HISTORY_FILE
    else:
        history_target = globals()['history_path']()

    def record(outcome: Outcome, index: int) -> None:
        result.outcomes.append(outcome)
        if on_outcome is not None:
            on_outcome(index, outcome)

    for index, given in enumerate(sources):
        source = Path(given)
        if stopped:
            record(Outcome(source=source, ok=False, cancelled=True), index)
            continue
        job = Job(
            source=source,
            profile=profile,
            models=models,
            out_dir=Path(out_dir) if out_dir is not None else None,
            threads=threads,
            author=author,
            keep_audio=keep_audio,
            plan=batch_plan,
            preference=preference,
            speakers=speakers,
        )
        started = time.perf_counter()

        def file_progress(p: Progress, _index: int = index) -> None:
            if progress is not None:
                progress(_index, total, p)

        def file_partial(role: str, segment: Segment, _index: int = index) -> None:
            if on_partial is not None:
                on_partial(_index, role, segment)

        try:
            file_result = process_file(
                job, progress=file_progress, cancel=cancel, engines=engines,
                on_partial=file_partial if on_partial is not None else None,
            )
            try:
                append_entry(entry_from_result(file_result), history_target)
            except (OSError, ValueError, TypeError):
                pass  # the history is a convenience; a run never fails for it
            record(Outcome(source=source, ok=True, result=file_result, elapsed_s=time.perf_counter() - started), index)
        except Cancelled:
            record(Outcome(source=source, ok=False, cancelled=True, elapsed_s=time.perf_counter() - started), index)
            stopped = True
        except Exception as exc:  # noqa: BLE001 - every failure is recorded and the batch goes on
            record(
                Outcome(
                    source=source,
                    ok=False,
                    error_class=type(exc).__name__,
                    error=str(exc),
                    elapsed_s=time.perf_counter() - started,
                ),
                index,
            )
    folder = Path(record_dir) if record_dir is not None else runs_dir()
    stamp = started_utc.replace(":", "").replace("-", "").replace("+0000", "Z")
    result.record_path = folder / f"batch_{stamp}.json"
    write_json_atomic(
        {
            "schema": BATCH_SCHEMA,
            "version": __version__,
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "profile": profile.name,
            "models_root": str(models.root),
            "speakers": speakers,
            "plan": batch_plan.to_dict(),
            "files": [o.to_dict() for o in result.outcomes],
            "failures": [
                {"path": str(o.source), "error_class": o.error_class, "message": o.error} for o in result.failures
            ],
        },
        result.record_path,
    )
    return result
