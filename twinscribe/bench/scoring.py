"""Scores for one item: an engine's words against the reference, a diarizer's turns against the
reference speakers, and the two-engine review list against the reference words. Every
function returns plain dictionaries so that a results file can be written and read back
without the records.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from twinscribe.bench.references import WORD_TIMES_ANNOTATED, WORD_TIMES_SPREAD, Reference, RefWord
from twinscribe.engines.base import SpeakerTurn, Word
from twinscribe.metrics import MAX_SPEAKERS, cer_chunked, der, jer, per_speaker_recall, timestamp_offsets, word_errors
from twinscribe.profiles import ReviewSettings
from twinscribe.review import build_review, evaluate_review
from twinscribe.text import tokenize_norm, tokenize_raw

FOLDED_LABEL = "other"


def _tokens_with_times(words: Sequence[Word | RefWord]) -> tuple[list[str], list[tuple[float, float]]]:
    """Raw tokens of every word, each carrying its word's span; a word that tokenises to
    several tokens gives each of them the same span."""
    tokens: list[str] = []
    times: list[tuple[float, float]] = []
    for word in words:
        if word.start is None or word.end is None:
            continue
        for token in tokenize_raw(word.text):
            tokens.append(token)
            times.append((float(word.start), float(word.end)))
    return tokens, times


def transcript_scores(reference: Reference, words: Sequence[Word]) -> dict[str, Any]:
    """Word error rate on raw and normalised tokens with the substitution, deletion and
    insertion split, chunked character error rate, per-speaker recall on normalised tokens,
    and onset offsets where both sides carry word times the annotation vouches for."""
    hypothesis_text = " ".join(word.text for word in words)
    out: dict[str, Any] = {}
    for name, tokenize in (("raw", tokenize_raw), ("norm", tokenize_norm)):
        ref_tokens, hyp_tokens = tokenize(reference.text), tokenize(hypothesis_text)
        errors, ops = word_errors(ref_tokens, hyp_tokens)
        out[name] = {
            "n_ref": errors.n_ref,
            "n_hyp": errors.n_hyp,
            "hits": errors.hits,
            "sub": errors.sub,
            "del": errors.dele,
            "ins": errors.ins,
            "wer": errors.wer,
            "cer": cer_chunked(ref_tokens, hyp_tokens, ops),
        }
    ref_tokens: list[str] = []
    speakers: list[str] = []
    for segment in reference.segments:
        tokens = tokenize_norm(segment.text)
        ref_tokens.extend(tokens)
        speakers.extend([segment.speaker] * len(tokens))
    recall = per_speaker_recall(ref_tokens, speakers, tokenize_norm(hypothesis_text))
    out["per_speaker"] = {
        speaker: {"n_tokens": r.n_tokens, "recovered": r.recovered, "substituted": r.substituted, "deleted": r.deleted, "recall": r.recall}
        for speaker, r in recall.items()
    }
    out["timing"] = None
    if reference.word_times == WORD_TIMES_ANNOTATED and words:
        ref_timed, ref_times = _tokens_with_times(reference.words)
        hyp_timed, hyp_times = _tokens_with_times(words)
        if ref_timed and hyp_timed:
            _, ops = word_errors(ref_timed, hyp_timed)
            timing = timestamp_offsets(ops, ref_times, hyp_times)
            out["timing"] = {"n": timing.n, "median_abs_s": timing.median_abs_s, "p90_abs_s": timing.p90_abs_s}
    return out


def fold_labels(turns: Sequence[SpeakerTurn], limit: int = MAX_SPEAKERS) -> tuple[list[tuple[float, float, str]], int, str | None]:
    """Turns as (start, end, label) with at most `limit` labels: the largest by time are kept
    and the rest folded into one, which can only be mapped once anyway. Returns the segments,
    the number of labels found, and a note when folding happened."""
    seconds: dict[str, float] = {}
    for turn in turns:
        seconds[turn.label] = seconds.get(turn.label, 0.0) + turn.duration
    found = len(seconds)
    if found <= limit:
        return [(t.start, t.end, t.label) for t in turns], found, None
    keep = set(sorted(seconds, key=lambda label: -seconds[label])[: limit - 1])
    folded = [(t.start, t.end, t.label if t.label in keep else FOLDED_LABEL) for t in turns]
    return folded, found, f"{found} labels found; the {found - len(keep)} smallest by time folded into one for the mapping"


def diarization_scores(reference: Reference, turns: Sequence[SpeakerTurn]) -> dict[str, Any]:
    """Diarization error rate with its split at collars 0.25 s and 0, overlap included, and
    the Jaccard error rate beside it; labels beyond the mapping's limit are folded."""
    ref_segments = [(s.start, s.end, s.speaker) for s in reference.segments if s.end > s.start]
    hyp_segments, found, note = fold_labels([t for t in turns if t.end > t.start])
    out: dict[str, Any] = {"labels_found": found, "note": note}
    if not ref_segments or not hyp_segments:
        out.update({"der_0.25": None, "der_0": None, "jer": None})
        return out
    for collar in (0.25, 0.0):
        result = der(ref_segments, hyp_segments, collar=collar)
        out[f"der_{collar:g}"] = {
            "der": result.der,
            "miss": result.miss,
            "false_alarm": result.false_alarm,
            "confusion": result.confusion,
            "scored_ref_s": result.scored_ref_s,
            "n_ref_speakers": result.n_ref_speakers,
            "n_hyp_speakers": result.n_hyp_speakers,
            "mapping": dict(result.mapping),
            "unmatched_ref": list(result.unmatched_ref),
        }
    out["jer"] = jer(ref_segments, hyp_segments)
    return out


def review_scores(
    reference: Reference,
    published: Sequence[Word],
    detector: Sequence[Word],
    settings: ReviewSettings | None = None,
) -> dict[str, Any]:
    """The review list from the published and detector words at the design's rule, scored
    against the reference words; approximate when the reference word times are spread."""
    rule = settings if settings is not None else ReviewSettings()
    marks = build_review(
        list(published), list(detector), reference.audio_s,
        min_silence_s=rule.min_silence_s, min_detector_words=rule.min_detector_words, pad_s=rule.pad_s,
    )
    reference_words = reference.timed_words()
    out: dict[str, Any] = {
        "marks": len(marks),
        "audio_to_review_s": sum(max(0.0, m.end - m.start) for m in marks),
        "evaluated": bool(reference_words),
        "approximate": reference.word_times == WORD_TIMES_SPREAD,
    }
    if not reference_words:
        return out
    result = evaluate_review(marks, list(published), reference_words, reference.audio_s)
    out.update(
        {
            "marks_on_speech": result.marks_on_speech,
            "precision": result.precision,
            "dropped_words": result.dropped_words,
            "dropped_covered": result.dropped_covered,
            "recall": result.recall,
            "audio_to_review_s": result.audio_to_review_s,
            "audio_to_review_fraction": result.audio_to_review_fraction,
        }
    )
    return out


def pooled_word_errors(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts summed over rows of transcript scores (one tokenisation), with the pooled rate."""
    totals = {"n_ref": 0, "n_hyp": 0, "hits": 0, "sub": 0, "del": 0, "ins": 0, "items": 0}
    for row in rows:
        for key in ("n_ref", "n_hyp", "hits", "sub", "del", "ins"):
            totals[key] += int(row.get(key, 0))
        totals["items"] += 1
    n_ref = totals["n_ref"]
    totals["wer"] = None if n_ref == 0 else (totals["sub"] + totals["del"] + totals["ins"]) / n_ref
    return totals
