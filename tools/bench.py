"""Run the bench: build the items the corpora store can provide, run each engine variant on
them with a contention verdict around every timed cell, score against the references, and
write the results document and the report.

Every engine output is kept as a JSON file under the output folder, so a run can be resumed
after an interruption and re-scored without decoding again (--rescore).

Usage:
    python tools/bench.py --corpora ROOT --models ROOT --out FOLDER [--items PATTERN ...]
        [--variants NAME ...] [--device auto|cpu|cuda] [--threads N] [--rescore]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from twinscribe import load  # noqa: E402
from twinscribe.bench.corpora import CATALOGUE, find_corpora  # noqa: E402
from twinscribe.bench.items import Item, ItemSpec, build_item, list_items, safe_name  # noqa: E402
from twinscribe.bench.report import KIND_DIARIZATION, KIND_REVIEW, KIND_TRANSCRIPT, render_report  # noqa: E402
from twinscribe.bench.scoring import diarization_scores, review_scores, transcript_scores  # noqa: E402
from twinscribe.engines.base import Diarization, SpeakerTurn, Transcript, Word  # noqa: E402
from twinscribe.hardware import BACKEND_CT2, DEVICE_AUTO, PREFERENCES, current_plan  # noqa: E402
from twinscribe.models import (  # noqa: E402
    KEY_EMBEDDING,
    KEY_PARAKEET_V2,
    KEY_SEGMENTATION,
    KEY_SILERO_VAD,
    KEY_WHISPER_DISTIL,
    KEY_WHISPER_LARGE,
    KEY_WHISPER_TURBO,
    find_models,
)
from twinscribe.profiles import ReviewSettings  # noqa: E402
from twinscribe.review import checking_windows  # noqa: E402
from twinscribe.runrecord import machine_facts, utc_now, write_json_atomic  # noqa: E402

RESULTS_SCHEMA = "twinscribe.bench.v1"
PUBLISHER_PRESET = "vad"
DETECTOR_PRESET = "production"
DIARIZATION_THRESHOLD = 0.5

VARIANT_PARAKEET = "parakeet"
VARIANT_TURBO = "whisper-turbo"
VARIANT_DISTIL = "whisper-distil"
VARIANT_LARGE = "whisper-large"
VARIANT_SPEAKERS = "speakers"
VARIANT_SPEAKERS_COUNT = "speakers-count"
VARIANT_REVIEW = "review"
VARIANT_TURBO_GAPS = "whisper-turbo-gaps"
VARIANT_REVIEW_GAPS = "review-gaps"
CHECKING_MARGIN_S = 0.5

TRANSCRIPT_VARIANTS: dict[str, str] = {
    VARIANT_TURBO: KEY_WHISPER_TURBO,
    VARIANT_DISTIL: KEY_WHISPER_DISTIL,
    VARIANT_LARGE: KEY_WHISPER_LARGE,
    VARIANT_TURBO_GAPS: KEY_WHISPER_TURBO,
}
DEFAULT_VARIANTS: tuple[str, ...] = (
    VARIANT_PARAKEET, VARIANT_TURBO, VARIANT_DISTIL, VARIANT_SPEAKERS, VARIANT_SPEAKERS_COUNT, VARIANT_REVIEW,
)
ALL_VARIANTS: tuple[str, ...] = DEFAULT_VARIANTS[:3] + (VARIANT_LARGE, VARIANT_TURBO_GAPS) + DEFAULT_VARIANTS[3:] + (VARIANT_REVIEW_GAPS,)


def _words_to_dicts(words: list[Word]) -> list[dict[str, Any]]:
    return [{"text": w.text, "start": w.start, "end": w.end, "prob": w.prob} for w in words]


def _words_from_dicts(rows: list[dict[str, Any]]) -> list[Word]:
    return [Word(text=str(r["text"]), start=float(r["start"]), end=float(r["end"]), prob=r.get("prob")) for r in rows]


def _turns_from_dicts(rows: list[dict[str, Any]]) -> list[SpeakerTurn]:
    return [SpeakerTurn(start=float(r["start"]), end=float(r["end"]), label=str(r["label"])) for r in rows]


def _verdict_dict(before: Any, after: Any) -> dict[str, Any]:
    verdict = load.verdict(before, after)
    return {
        "other_load_cores": verdict.other_load_cores,
        "busy": verdict.busy,
        "on_mains_power": verdict.on_mains_power,
        "threshold_cores": verdict.threshold_cores,
    }


def _transcript_record(transcript: Transcript, verdict: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": KIND_TRANSCRIPT,
        "engine": {"engine": transcript.engine, "model": transcript.model, "preset": transcript.preset, "settings": dict(transcript.settings)},
        "versions": dict(transcript.versions),
        "extras": dict(transcript.extras),
        "timing": {"audio_s": transcript.audio_s, "load_s": transcript.load_s, "transcribe_s": transcript.transcribe_s},
        "verdict": verdict,
        "words": _words_to_dicts(transcript.words),
        "text": transcript.text,
    }


def _diarization_record(result: Diarization, verdict: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": KIND_DIARIZATION,
        "engine": {"engine": "sherpa_diarization", "settings": dict(result.settings)},
        "versions": dict(result.versions),
        "timing": {"audio_s": result.audio_s, "load_s": result.load_s, "transcribe_s": result.diarize_s},
        "verdict": verdict,
        "turns": [{"start": t.start, "end": t.end, "label": t.label} for t in result.segments],
    }


class Runner:
    """Runs variants on items, caching every engine output as JSON under the output folder."""

    def __init__(self, models: Any, plan: Any, out: Path, threads: int | None, rescore: bool, checking_margin_s: float = CHECKING_MARGIN_S) -> None:
        self.checking_margin_s = checking_margin_s
        self.models = models
        self.plan = plan
        self.out = out
        self.threads = threads
        self.rescore = rescore

    def _path(self, item: Item, variant: str) -> Path:
        return self.out / "engines" / safe_name(item.spec.id) / f"{variant}.json"

    def _cached(self, item: Item, variant: str) -> dict[str, Any] | None:
        path = self._path(item, variant)
        if not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _store(self, item: Item, variant: str, record: dict[str, Any]) -> dict[str, Any]:
        write_json_atomic(record, self._path(item, variant))
        return record

    def run_publisher(self, item: Item) -> dict[str, Any] | None:
        cached = self._cached(item, VARIANT_PARAKEET)
        if cached is not None or self.rescore:
            return cached
        from twinscribe.engines import parakeet

        before = load.snapshot()
        transcript = parakeet.transcribe(
            item.wav, self.models.path(KEY_PARAKEET_V2), self.models.file(KEY_SILERO_VAD, "silero_vad.onnx"),
            PUBLISHER_PRESET, threads=self.threads, provider=self.plan.publisher.provider,
        )
        return self._store(item, VARIANT_PARAKEET, _transcript_record(transcript, _verdict_dict(before, load.snapshot())))

    def run_detector(self, item: Item, variant: str) -> dict[str, Any] | None:
        cached = self._cached(item, variant)
        if cached is not None or self.rescore:
            return cached
        key = TRANSCRIPT_VARIANTS[variant]
        if not self.models.has(key):
            print(f"    {variant}: model {key} absent; skipped")
            return None
        placement = self.plan.placement_for(BACKEND_CT2)
        if placement is None:
            print(f"    {variant}: no CTranslate2 detector library; skipped")
            return None
        from twinscribe.engines import whisper_ct2

        clips = None
        if variant == VARIANT_TURBO_GAPS:
            # The checker decodes only where the publisher fell silent; the publisher's record must exist.
            publisher = self._cached(item, VARIANT_PARAKEET)
            if publisher is None:
                print(f"    {variant}: the publisher has not run for this item; skipped")
                return None
            clips = checking_windows(
                _words_from_dicts(publisher["words"]), float(publisher["timing"]["audio_s"]), ReviewSettings().min_silence_s, self.checking_margin_s,
            )
        before = load.snapshot()
        transcript = whisper_ct2.transcribe(
            item.wav, self.models.path(key), DETECTOR_PRESET, threads=self.threads,
            device=placement.device, compute_type=placement.compute_type or "auto", device_index=placement.index,
            clips=clips,
        )
        return self._store(item, variant, _transcript_record(transcript, _verdict_dict(before, load.snapshot())))

    def run_speakers(self, item: Item, variant: str) -> dict[str, Any] | None:
        cached = self._cached(item, variant)
        if cached is not None or self.rescore:
            return cached
        from twinscribe.engines import diarize

        count = len(item.reference.speakers) if variant == VARIANT_SPEAKERS_COUNT else None
        before = load.snapshot()
        result = diarize.diarize(
            item.wav, self.models.file(KEY_SEGMENTATION, "model.onnx"), self.models.file(KEY_EMBEDDING, "nemo_en_titanet_large.onnx"),
            threads=self.threads, threshold=DIARIZATION_THRESHOLD, provider=self.plan.diarizer.provider,
            **({"num_speakers": count} if count is not None else {}),
        )
        return self._store(item, variant, _diarization_record(result, _verdict_dict(before, load.snapshot())))


def score_item(item: Item, records: dict[str, dict[str, Any]], variants: tuple[str, ...]) -> dict[str, Any]:
    """Score every record of an item; the review variant needs the publisher and the turbo
    detector and is scored from their words."""
    out: dict[str, Any] = {}
    for variant, record in records.items():
        if record is None:
            continue
        entry = {k: v for k, v in record.items() if k not in ("words", "turns", "text")}
        if record["kind"] == KIND_TRANSCRIPT:
            entry["scores"] = transcript_scores(item.reference, _words_from_dicts(record["words"]))
        elif record["kind"] == KIND_DIARIZATION:
            entry["scores"] = diarization_scores(item.reference, _turns_from_dicts(record["turns"]))
        out[variant] = entry
    if VARIANT_REVIEW in variants and records.get(VARIANT_PARAKEET) and records.get(VARIANT_TURBO):
        published = _words_from_dicts(records[VARIANT_PARAKEET]["words"])
        detector = _words_from_dicts(records[VARIANT_TURBO]["words"])
        out[VARIANT_REVIEW] = {
            "kind": KIND_REVIEW,
            "engine": {"publisher": VARIANT_PARAKEET, "detector": VARIANT_TURBO},
            "scores": review_scores(item.reference, published, detector),
        }
    if VARIANT_REVIEW_GAPS in variants and records.get(VARIANT_PARAKEET) and records.get(VARIANT_TURBO_GAPS):
        published = _words_from_dicts(records[VARIANT_PARAKEET]["words"])
        detector = _words_from_dicts(records[VARIANT_TURBO_GAPS]["words"])
        out[VARIANT_REVIEW_GAPS] = {
            "kind": KIND_REVIEW,
            "engine": {"publisher": VARIANT_PARAKEET, "detector": VARIANT_TURBO_GAPS},
            "scores": review_scores(item.reference, published, detector),
        }
    return out


def select_items(specs: list[ItemSpec], patterns: list[str] | None) -> list[ItemSpec]:
    if not patterns:
        return specs
    return [spec for spec in specs if any(fnmatch.fnmatchcase(spec.id, pattern) for pattern in patterns)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the engines over the bench corpora and score them.")
    parser.add_argument("--corpora", type=Path, default=None, help="corpora root folder")
    parser.add_argument("--models", type=Path, default=None, help="models root folder")
    parser.add_argument("--out", type=Path, required=True, help="output folder for engine outputs, results and report")
    parser.add_argument("--work", type=Path, default=None, help="folder for the decoded items (default: <out>/work)")
    parser.add_argument("--items", nargs="*", default=None, help="item id patterns (fnmatch), default all")
    parser.add_argument("--variants", nargs="*", default=None, choices=list(ALL_VARIANTS), help="variants to run")
    parser.add_argument("--device", default=DEVICE_AUTO, choices=list(PREFERENCES))
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--rescore", action="store_true", help="score cached engine outputs only; run nothing")
    parser.add_argument("--checking-margin", type=float, default=CHECKING_MARGIN_S, help="context margin in seconds round the checking windows of the targeted arm")
    args = parser.parse_args(argv)

    store = find_corpora(args.corpora)
    print("corpora root:", store.root)
    print(store.describe())
    models = find_models(args.models)
    plan = current_plan(args.device, args.threads)
    for line in plan.describe():
        print("  " + line)
    variants = tuple(args.variants) if args.variants else DEFAULT_VARIANTS
    specs = select_items(list_items(store), args.items)
    if not specs:
        print("no items match; nothing to run", file=sys.stderr)
        return 2
    out: Path = args.out
    work = args.work if args.work is not None else out / "work"
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner(models, plan, out, args.threads, args.rescore, args.checking_margin)

    results: dict[str, Any] = {
        "schema": RESULTS_SCHEMA,
        "meta": {
            "produced_utc": utc_now(),
            "machine": machine_facts(),
            "plan": {"lines": plan.describe(), **plan.to_dict()},
            "device": args.device,
            "threads": args.threads,
            "versions": {},
            "credits": [f"{spec.title}: {spec.credit} ({spec.licence}; {spec.revision})" for spec in CATALOGUE if store.has(spec.key)],
        },
        "items": {},
    }
    started = time.perf_counter()
    for index, spec in enumerate(specs, 1):
        print(f"[{index}/{len(specs)}] {spec.id}")
        try:
            item = build_item(spec, store, work)
        except Exception as exc:  # noqa: BLE001 - one item must not end the bench
            print(f"    build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            results["items"][spec.id] = {"corpus": spec.corpus, "kind": spec.kind, "error": f"{type(exc).__name__}: {exc}"}
            continue
        records: dict[str, dict[str, Any] | None] = {}
        try:
            if VARIANT_PARAKEET in variants or VARIANT_REVIEW in variants:
                records[VARIANT_PARAKEET] = runner.run_publisher(item)
            for variant in TRANSCRIPT_VARIANTS:
                if variant in variants or (variant == VARIANT_TURBO and VARIANT_REVIEW in variants):
                    records[variant] = runner.run_detector(item, variant)
            for variant in (VARIANT_SPEAKERS, VARIANT_SPEAKERS_COUNT):
                if variant in variants:
                    records[variant] = runner.run_speakers(item, variant)
        except Exception as exc:  # noqa: BLE001 - recorded; the next item still runs
            print(f"    engine failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            results["items"].setdefault(spec.id, {})["error"] = f"{type(exc).__name__}: {exc}"
        present = {k: v for k, v in records.items() if v is not None}
        for record in present.values():
            results["meta"]["versions"].update(record.get("versions") or {})
        results["items"][spec.id] = {
            **results["items"].get(spec.id, {}),
            "corpus": spec.corpus,
            "kind": spec.kind,
            "audio_s": item.reference.audio_s,
            "speakers": list(item.reference.speakers),
            "ref_words_norm": len(item.reference.words),
            "word_times": item.reference.word_times,
            "notes": list(item.reference.notes),
            "reference": item.reference.to_dict(),
            "variants": score_item(item, present, variants),
        }
        for variant, entry in results["items"][spec.id]["variants"].items():
            scores = entry.get("scores") or {}
            if entry["kind"] == KIND_TRANSCRIPT:
                norm = scores["norm"]
                print(f"    {variant:<16} WER norm {100 * (norm['wer'] or 0):.1f}%  ({norm['sub']}/{norm['del']}/{norm['ins']} of {norm['n_ref']})  "
                      f"decode {entry['timing']['transcribe_s']:.1f} s  busy={entry['verdict'].get('busy')}")
            elif entry["kind"] == KIND_DIARIZATION:
                d = scores.get("der_0.25") or {}
                print(f"    {variant:<16} labels {scores.get('labels_found')}  DER {100 * (d.get('der') or 0):.1f}%  JER {100 * (scores.get('jer') or 0):.1f}%")
            else:
                print(f"    {variant:<16} marks {scores.get('marks')}  on speech {scores.get('marks_on_speech', '-')}")
        write_json_atomic(results, out / "results.json")
    results["meta"]["elapsed_s"] = time.perf_counter() - started
    write_json_atomic(results, out / "results.json")
    report = render_report(results)
    (out / "report.md").write_text(report, encoding="utf-8")
    print(f"\nresults: {out / 'results.json'}\nreport:  {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
