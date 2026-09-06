"""Word timestamp offsets between an engine's word times and reference word times, and a
count of reference words that fall inside long hypothesis gaps.

Only words the alignment matched are compared, and only where the engine produced its own
times; a gap count is the timing view of the transducer's silent failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .align import Op


@dataclass(frozen=True)
class TimingResult:
    """Onset offset statistics over matched words.

    `offsets_s` holds the signed offsets (hypothesis start minus reference start) in
    alignment order so that a report can plot or bootstrap them; `median_abs_s` and
    `p90_abs_s` summarise their absolute values and are None when `n` is 0.
    """

    n: int
    median_abs_s: float | None
    p90_abs_s: float | None
    offsets_s: tuple[float, ...]


def timestamp_offsets(ops: list[Op], ref_times: list[tuple[float, float]], hyp_times: list[tuple[float, float]]) -> TimingResult:
    """Signed onset offsets over "equal" ops; count, median and 90th percentile of |offset|.

    `ref_times[i]` and `hyp_times[j]` are (start, end) for reference token i and hypothesis
    token j. The 90th percentile is numpy's default linear interpolation between order
    statistics.
    """
    offsets = [hyp_times[op.hyp][0] - ref_times[op.ref][0] for op in ops if op.kind == "equal"]
    if not offsets:
        return TimingResult(0, None, None, ())
    magnitude = np.abs(np.asarray(offsets, dtype=float))
    return TimingResult(
        n=len(offsets),
        median_abs_s=float(np.median(magnitude)),
        p90_abs_s=float(np.percentile(magnitude, 90)),
        offsets_s=tuple(float(x) for x in offsets),
    )


def _covered(times: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge word spans into disjoint covered intervals, sorted by start."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(times):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def hypothesis_gaps(ref_times: Sequence[tuple[float, float]], hyp_times: Sequence[tuple[float, float]], min_gap: float = 2.0) -> list[tuple[float, float]]:
    """Spans of at least `min_gap` seconds with no hypothesis word, within the annotated extent.

    The extent runs from the earliest start to the latest end over both sides; the lead
    before the first hypothesis word and the tail after the last are gaps like any other
    and are subject to the same minimum length. With no hypothesis words the whole extent
    is one gap.
    """
    if min_gap < 0:
        raise ValueError("min_gap must be non-negative")
    spans = list(ref_times) + list(hyp_times)
    if not spans:
        return []
    extent_start = min(s for s, _ in spans)
    extent_end = max(e for _, e in spans)
    edges = [extent_start]
    for start, end in _covered(hyp_times):
        edges.extend((start, end))
    edges.append(extent_end)
    gaps = []
    for k in range(0, len(edges), 2):
        gap_start, gap_end = edges[k], edges[k + 1]
        if gap_end - gap_start >= min_gap and gap_end > gap_start:
            gaps.append((gap_start, gap_end))
    return gaps


def words_in_gaps(ref_times: list[tuple[float, float]], hyp_times: list[tuple[float, float]], min_gap: float = 2.0) -> int:
    """Count reference words whose midpoint lies inside a hypothesis gap of at least `min_gap`.

    Gaps are those of `hypothesis_gaps`, so the unannotated lead and tail count when they
    are long enough. A midpoint exactly on a gap edge is inside the gap.
    """
    gaps = hypothesis_gaps(ref_times, hyp_times, min_gap)
    if not gaps:
        return 0
    count = 0
    for start, end in ref_times:
        mid = (start + end) / 2.0
        if any(g_start <= mid <= g_end for g_start, g_end in gaps):
            count += 1
    return count
