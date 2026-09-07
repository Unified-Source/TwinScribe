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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from twinscribe import __version__, audio, load
from twinscribe.engines.base import Diarization, Transcript
from twinscribe.engines.presets import PARAKEET_PRESETS, WHISPER_PRESETS
from twinscribe.labelling import build_lines, label_words
from twinscribe.models import KEY_EMBEDDING, KEY_SEGMENTATION, KEY_SILERO_VAD, ModelSet
from twinscribe.outputs import render_all
from twinscribe.outputs.transcript_doc import build_document, overview_peaks, write_document
from twinscribe.paths import runs_dir, work_dir
from twinscribe.profiles import Profile
from twinscribe.review import build_review, review_set, write_review_set
from twinscribe.runrecord import Failure, RunRecord, utc_now, write_json_atomic

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
    ("publisher", 0.33),
    ("detector", 0.40),
    ("speakers", 0.15),
    ("outputs", 0.05),
)
STAGE_TITLES: dict[str, str] = {
    "digest": "Reading the file",
    "decode": "Decoding the audio",
    "publisher": "Transcribing (published engine)",
    "detector": "Checking (second engine)",
    "speakers": "Labelling speakers",
    "outputs": "Writing the outputs",
}


class Cancelled(RuntimeError):
    """Raised inside a run when the cancel check of the caller returns True."""


@dataclass(frozen=True)
class Progress:
    """One progress report: the stage, the overall fraction for the file and a message."""

    stage: str
    fraction: float
    message: str


ProgressFn = Callable[[Progress], None]
CancelFn = Callable[[], bool]


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


def output_paths(source: str | os.PathLike[str], out_dir: str | os.PathLike[str] | None = None) -> OutputPaths:
    """Output paths for a recording: beside it, or in out_dir when given, named by its stem."""
    src = Path(source)
    folder = Path(out_dir) if out_dir is not None else src.parent
    stem = src.stem
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
                    found.setdefault(str(candidate.resolve()).lower(), candidate)
        elif path.is_file() and is_media(path):
            found.setdefault(str(path.resolve()).lower(), path)
    return sorted(found.values(), key=lambda p: (str(p.parent).lower(), p.name.lower()))


@dataclass(frozen=True)
class Job:
    """One recording to process, with everything the pipeline needs to know."""

    source: Path
    profile: Profile
    models: ModelSet
    out_dir: Path | None = None
    threads: int | None = None
    author: str = ""
    keep_audio: bool = False
    work_folder: Path | None = None


@dataclass(frozen=True)
class Engines:
    """The three engine calls, so that tests can stand synthetic engines in for the real ones."""

    publisher: Callable[..., Transcript]
    detector: Callable[..., Transcript]
    diarizer: Callable[..., Diarization] | None


def default_engines() -> Engines:
    """The real engine wrappers; their libraries import lazily when first called."""
    from twinscribe.engines import diarize, parakeet, whisper_ct2

    return Engines(publisher=parakeet.transcribe, detector=whisper_ct2.transcribe, diarizer=diarize.diarize)


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
    """Turns per-stage fractions into overall progress and checks for cancellation."""

    def __init__(self, progress: ProgressFn | None, cancel: CancelFn | None) -> None:
        self._progress = progress
        self._cancel = cancel
        self._offsets: dict[str, tuple[float, float]] = {}
        start = 0.0
        for name, weight in STAGES:
            self._offsets[name] = (start, weight)
            start += weight

    def check(self) -> None:
        if self._cancel is not None and self._cancel():
            raise Cancelled("cancelled")

    def report(self, stage: str, fraction: float, message: str | None = None) -> None:
        self.check()
        offset, weight = self._offsets[stage]
        overall = min(1.0, max(0.0, offset + weight * min(1.0, max(0.0, fraction))))
        if self._progress is not None:
            self._progress(Progress(stage=stage, fraction=overall, message=message or STAGE_TITLES[stage]))

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
    reporter.report("decode", 0.0)
    audio.decode_to_wav(source, target)
    reporter.report("decode", 1.0)
    return target, True


def _preset_settings(profile: Profile) -> dict[str, Any]:
    return {
        "profile": profile.name,
        "publisher_preset": {profile.publisher_preset: PARAKEET_PRESETS.get(profile.publisher_preset)},
        "detector_preset": {profile.detector_preset: WHISPER_PRESETS.get(profile.detector_preset)},
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
) -> FileResult:
    """Run the whole pipeline over one recording and write its outputs.

    Speaker labelling that fails does not lose the transcript: the failure is recorded in the
    run record and the document, and every word is left unlabelled. Any other failure
    propagates after a run record naming it has been written beside the recording.
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
        digest = audio.sha256_of(source)
        size = source.stat().st_size
        reporter.report("digest", 1.0)

        wav, temporary = _decode_source(job, reporter, digest)
        models = job.models
        profile = job.profile

        before = load.snapshot()
        reporter.report("publisher", 0.0)
        published = engine_set.publisher(
            wav,
            models.path(profile.publisher),
            models.file(KEY_SILERO_VAD, "silero_vad.onnx"),
            profile.publisher_preset,
            threads=job.threads,
            progress=reporter.stage_fn("publisher"),
        )
        reporter.report("publisher", 1.0)

        reporter.report("detector", 0.0)
        detector = engine_set.detector(
            wav,
            models.path(profile.detector),
            profile.detector_preset,
            threads=job.threads,
            progress=reporter.stage_fn("detector"),
        )
        reporter.report("detector", 1.0)

        reporter.report("speakers", 0.0)
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
                )
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - recorded; the transcript is still published
                speaker_failure = f"{type(exc).__name__}: {exc}"
                failures.append(Failure(str(source), type(exc).__name__, f"speaker labelling: {exc}"))
        after = load.snapshot()
        reporter.report("speakers", 0.6)

        words = published.words
        audio_s = float(published.audio_s) if published.audio_s > 0 else float(detector.audio_s)
        labels = label_words(words, diarization.segments if diarization is not None else ())
        lines = build_lines(words, labels)
        marks = build_review(
            words,
            detector.words,
            audio_s,
            min_silence_s=profile.review.min_silence_s,
            min_detector_words=profile.review.min_detector_words,
            pad_s=profile.review.pad_s,
        )
        samples = audio.read_wav_mono16k(wav)
        overview = overview_peaks(samples)
        del samples
        reporter.report("speakers", 1.0)

        reporter.report("outputs", 0.0)
        document = build_document(
            source_name=source.name,
            source_sha256=digest,
            source_bytes=size,
            video=is_video(source),
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
        )
        write_document(document, paths.transcript)
        render_all(document, paths.text, paths.docx, paths.subtitles, author=job.author)
        audio_reference = source.name if job.out_dir is None else str(source.resolve())
        write_review_set(review_set(published, detector, audio_reference, marks), paths.review)
        reporter.report("outputs", 0.7)

        versions: dict[str, str] = {}
        versions.update(published.versions)
        versions.update(detector.versions)
        if diarization is not None:
            versions.update(diarization.versions)
        engine_facts = {
            "publisher": {"engine": published.engine, "model": published.model, "preset": published.preset},
            "detector": {"engine": detector.engine, "model": detector.model, "preset": detector.preset},
            "diarization": {
                "engine": "sherpa_diarization",
                "model": f"{KEY_SEGMENTATION} + {KEY_EMBEDDING}",
                "preset": f"threshold {profile.diarization_threshold}",
            },
        }
        load_s = published.load_s + detector.load_s + (diarization.load_s if diarization is not None else 0.0)
        transcribe_s = (
            published.transcribe_s
            + detector.transcribe_s
            + (diarization.diarize_s if diarization is not None else 0.0)
        )
        record = RunRecord(
            engines=engine_facts,
            versions=versions,
            settings=_preset_settings(profile) | {"threads": job.threads},
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
) -> BatchResult:
    """Process recordings one after another, never stopping for a failure.

    A cancellation stops the batch; the recordings not reached are recorded as cancelled. The
    batch record lists every recording with its outputs or its error. on_outcome, when given,
    is called with the index and the outcome as soon as each recording finishes.
    """
    started_utc = utc_now()
    result = BatchResult()
    total = len(sources)
    stopped = False

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
        )
        started = time.perf_counter()

        def file_progress(p: Progress, _index: int = index) -> None:
            if progress is not None:
                progress(_index, total, p)

        try:
            file_result = process_file(job, progress=file_progress, cancel=cancel, engines=engines)
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
            "files": [o.to_dict() for o in result.outcomes],
            "failures": [
                {"path": str(o.source), "error_class": o.error_class, "message": o.error} for o in result.failures
            ],
        },
        result.record_path,
    )
    return result
