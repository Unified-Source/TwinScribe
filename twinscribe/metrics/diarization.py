"""Diarization error rate and Jaccard error rate on a merged boundary timeline.

DER is duration-weighted and lets a short speaker vanish at almost no cost; JER weights every
reference speaker equally, so both are reported and the mapping is exposed so that a speaker
whose label never lands on any hypothesis time is named rather than hidden in a total.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable, Sequence

Seg = tuple[float, float, str]

MAX_SPEAKERS = 6


@dataclass(frozen=True)
class DerResult:
    """Diarization error rate and its split, plus the speaker mapping that produced it.

    `der`, `miss`, `false_alarm` and `confusion` are fractions of `scored_ref_s`, the total
    reference speaker-time inside the scored region (overlapped speech counts once per
    speaker). `mapping` holds reference label -> hypothesis label for the mapped speakers;
    `unmatched_ref` names reference speakers that received no hypothesis label, or a label
    they share no scored time with.
    """

    der: float
    miss: float
    false_alarm: float
    confusion: float
    scored_ref_s: float
    n_ref_speakers: int
    n_hyp_speakers: int
    mapping: dict[str, str]
    unmatched_ref: tuple[str, ...]


@dataclass(frozen=True)
class _Timeline:
    """Elementary intervals with the speaker sets active on each and a scored flag."""

    bounds: list[float]  # sorted boundaries; interval k is [bounds[k], bounds[k + 1])
    ref_active: list[set[str]]
    hyp_active: list[set[str]]
    scored: list[bool]

    def durations(self) -> list[float]:
        return [self.bounds[k + 1] - self.bounds[k] for k in range(len(self.bounds) - 1)]


def _check(segs: Iterable[Seg], side: str) -> list[Seg]:
    out: list[Seg] = []
    for start, end, label in segs:
        if end < start:
            raise ValueError(f"{side} segment ends before it starts: {(start, end, label)!r}")
        out.append((float(start), float(end), str(label)))
    return out


def _labels(segs: Sequence[Seg]) -> list[str]:
    return sorted({label for _, _, label in segs})


def _build_timeline(ref: Sequence[Seg], hyp: Sequence[Seg], collar: float, skip_overlap: bool) -> _Timeline:
    """Merge every boundary from both sides (and the collar edges) into elementary intervals."""
    points: set[float] = set()
    for start, end, _ in ref:
        points.update((start, end, start - collar, start + collar, end - collar, end + collar))
    for start, end, _ in hyp:
        points.update((start, end))
    bounds = sorted(points)
    n = max(len(bounds) - 1, 0)
    ref_active: list[set[str]] = [set() for _ in range(n)]
    hyp_active: list[set[str]] = [set() for _ in range(n)]

    def paint(segs: Sequence[Seg], active: list[set[str]]) -> None:
        for start, end, label in segs:
            lo = bisect_left(bounds, start)
            hi = bisect_left(bounds, end)
            for k in range(lo, hi):
                active[k].add(label)

    paint(ref, ref_active)
    paint(hyp, hyp_active)

    scored = [True] * n
    if collar > 0:
        for start, end, _ in ref:
            for b in (start, end):
                lo = bisect_left(bounds, b - collar)
                hi = bisect_left(bounds, b + collar)
                for k in range(lo, hi):
                    scored[k] = False
    if skip_overlap:
        for k in range(n):
            if len(ref_active[k]) > 1:
                scored[k] = False
    return _Timeline(bounds, ref_active, hyp_active, scored)


def _overlap_matrix(tl: _Timeline, ref_labels: Sequence[str], hyp_labels: Sequence[str]) -> dict[tuple[str, str], float]:
    """Scored time on which reference speaker r and hypothesis speaker h are both active."""
    overlap = {(r, h): 0.0 for r in ref_labels for h in hyp_labels}
    for k, d in enumerate(tl.durations()):
        if not tl.scored[k] or d <= 0:
            continue
        for r in tl.ref_active[k]:
            for h in tl.hyp_active[k]:
                overlap[(r, h)] += d
    return overlap


def _best_mapping(ref_labels: Sequence[str], hyp_labels: Sequence[str], overlap: dict[tuple[str, str], float]) -> dict[str, str]:
    """Exhaustive search over injective partial mappings maximising total overlap.

    Ties are broken towards fewer mapped pairs, so a reference speaker that shares no time
    with any free hypothesis label is left unmapped rather than paired arbitrarily, and then
    by label order so that the result is deterministic.
    """
    if len(ref_labels) > MAX_SPEAKERS or len(hyp_labels) > MAX_SPEAKERS:
        raise ValueError(
            f"exhaustive speaker mapping supports at most {MAX_SPEAKERS} speakers per side; "
            f"got {len(ref_labels)} reference and {len(hyp_labels)} hypothesis speakers"
        )
    best_key: tuple[float, int] | None = None
    best: dict[str, str] = {}

    def search(idx: int, used: set[str], current: dict[str, str], total: float) -> None:
        nonlocal best_key, best
        if idx == len(ref_labels):
            key = (total, -len(current))
            if best_key is None or key > best_key:
                best_key = key
                best = dict(current)
            return
        r = ref_labels[idx]
        # Leave r unmapped.
        search(idx + 1, used, current, total)
        for h in hyp_labels:
            if h in used:
                continue
            current[r] = h
            used.add(h)
            search(idx + 1, used, current, total + overlap[(r, h)])
            used.discard(h)
            del current[r]

    search(0, set(), {}, 0.0)
    return best


def _fraction(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return numerator / denominator
    return 0.0 if numerator == 0 else math.inf


def der(ref: list[Seg], hyp: list[Seg], collar: float = 0.25, skip_overlap: bool = False) -> DerResult:
    """Diarization error rate with the miss, false alarm and confusion split.

    Method: every boundary from both sides, plus the edges of the collar around every
    reference boundary, forms a merged timeline. On each elementary interval the reference
    speakers and hypothesis speakers are counted. Intervals within `collar` seconds of a
    reference segment start or end are excluded from scoring on both sides; with
    `skip_overlap` the intervals where more than one reference speaker is active are
    excluded as well. Per scored interval of duration d with n_ref reference and n_hyp
    hypothesis speakers, of which n_correct reference speakers have their mapped label
    active in the hypothesis: miss = d * max(n_ref - n_hyp, 0), false alarm =
    d * max(n_hyp - n_ref, 0), confusion = d * (min(n_ref, n_hyp) - n_correct). All three
    are divided by the scored reference speaker-time. The speaker mapping is the injective
    partial mapping from reference labels to hypothesis labels that minimises the total,
    found by exhaustive search (at most six speakers per side; `ValueError` beyond).

    Rates are 0.0 when there is neither scored reference speech nor error, and `math.inf`
    when there is error but no scored reference speech.
    """
    if collar < 0:
        raise ValueError("collar must be non-negative")
    ref = _check(ref, "reference")
    hyp = _check(hyp, "hypothesis")
    ref_labels = _labels(ref)
    hyp_labels = _labels(hyp)

    tl = _build_timeline(ref, hyp, collar, skip_overlap)
    overlap = _overlap_matrix(tl, ref_labels, hyp_labels)
    mapping = _best_mapping(ref_labels, hyp_labels, overlap)

    # Confusion is accumulated per interval under the chosen mapping rather than as the
    # difference of two totals, so every term is a non-negative integer times a duration
    # and identical inputs give exactly zero.
    scored_ref = 0.0
    miss = 0.0
    fa = 0.0
    confusion = 0.0
    for k, d in enumerate(tl.durations()):
        if not tl.scored[k] or d <= 0:
            continue
        ref_active = tl.ref_active[k]
        hyp_active = tl.hyp_active[k]
        n_r = len(ref_active)
        n_h = len(hyp_active)
        n_correct = sum(1 for r in ref_active if mapping.get(r) in hyp_active)
        scored_ref += d * n_r
        miss += d * max(n_r - n_h, 0)
        fa += d * max(n_h - n_r, 0)
        confusion += d * (min(n_r, n_h) - n_correct)

    unmatched = tuple(r for r in ref_labels if r not in mapping or overlap[(r, mapping[r])] <= 0)
    return DerResult(
        der=_fraction(miss + fa + confusion, scored_ref),
        miss=_fraction(miss, scored_ref),
        false_alarm=_fraction(fa, scored_ref),
        confusion=_fraction(confusion, scored_ref),
        scored_ref_s=scored_ref,
        n_ref_speakers=len(ref_labels),
        n_hyp_speakers=len(hyp_labels),
        mapping=mapping,
        unmatched_ref=unmatched,
    )


def jer(ref: list[Seg], hyp: list[Seg], collar: float = 0.25) -> float:
    """Jaccard error rate: mean over reference speakers of 1 - intersection / union.

    Each reference speaker is paired with its hypothesis speaker under the mapping chosen by
    `der` at the same collar with overlap included; the intersection and union are measured
    on the scored intervals only. A reference speaker left unmapped counts as 1.0. Returns
    0.0 when the reference has no speakers.
    """
    if collar < 0:
        raise ValueError("collar must be non-negative")
    ref = _check(ref, "reference")
    hyp = _check(hyp, "hypothesis")
    ref_labels = _labels(ref)
    hyp_labels = _labels(hyp)
    if not ref_labels:
        return 0.0

    tl = _build_timeline(ref, hyp, collar, skip_overlap=False)
    overlap = _overlap_matrix(tl, ref_labels, hyp_labels)
    mapping = _best_mapping(ref_labels, hyp_labels, overlap)

    ref_time = {r: 0.0 for r in ref_labels}
    hyp_time = {h: 0.0 for h in hyp_labels}
    for k, d in enumerate(tl.durations()):
        if not tl.scored[k] or d <= 0:
            continue
        for r in tl.ref_active[k]:
            ref_time[r] += d
        for h in tl.hyp_active[k]:
            hyp_time[h] += d

    total = 0.0
    for r in ref_labels:
        h = mapping.get(r)
        if h is None:
            total += 1.0
            continue
        inter = overlap[(r, h)]
        union = ref_time[r] + hyp_time[h] - inter
        total += (1.0 - inter / union) if union > 0 else 1.0
    return total / len(ref_labels)
