"""The transcript document: everything a run of one file produced that a reader or the
application needs, in one JSON file with schema twinscribe.transcript.v1.

It carries the published words grouped into speaker lines, the speaker summary and display
names, the review marks without the detector's text (that text lives only in the review set),
an overview of the waveform for the timeline, and the engine facts. The other renderers read
this document and nothing else.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from twinscribe import __version__
from twinscribe.engines.base import Diarization, Transcript
from twinscribe.labelling import Line, default_names, summarise_speakers
from twinscribe.review import Mark
from twinscribe.runrecord import write_json_atomic

TRANSCRIPT_SCHEMA = "twinscribe.transcript.v1"
OVERVIEW_BINS = 1200
OVERVIEW_SCALE = 100


def clock(seconds: float, tenths: bool = True) -> str:
    """m:ss.t below an hour, h:mm:ss.t from an hour on; without the tenth when asked."""
    total_tenths = max(0, int(round(seconds * 10.0)))
    whole, tenth = divmod(total_tenths, 10)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    body = f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
    return f"{body}.{tenth}" if tenths else body


def overview_peaks(samples: np.ndarray, bins: int = OVERVIEW_BINS) -> list[int]:
    """Peak level per bin on a 0 to OVERVIEW_SCALE scale, for drawing the waveform.

    The waveform is cut into bins of equal length and the largest absolute sample of each is
    kept, scaled so that the loudest bin reads OVERVIEW_SCALE. Silence gives zeros.
    """
    if bins < 1:
        raise ValueError("bins must be at least 1")
    data = np.asarray(samples, dtype=np.float32)
    if data.size == 0:
        return [0] * bins
    edges = np.linspace(0, data.size, bins + 1, dtype=np.int64)
    magnitude = np.abs(data)
    peaks = np.zeros(bins, dtype=np.float32)
    for index in range(bins):
        lo, hi = int(edges[index]), int(edges[index + 1])
        if hi > lo:
            peaks[index] = float(magnitude[lo:hi].max())
    top = float(peaks.max())
    if top <= 0.0:
        return [0] * bins
    return [int(round(OVERVIEW_SCALE * value / top)) for value in peaks]


def _engine_facts(transcript: Transcript) -> dict[str, str]:
    return {"engine": transcript.engine, "model": transcript.model, "preset": transcript.preset}


def _union_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    total = 0.0
    current: tuple[float, float] | None = None
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if current is not None and start <= current[1]:
            current = (current[0], max(current[1], end))
        else:
            if current is not None:
                total += current[1] - current[0]
            current = (start, end)
    if current is not None:
        total += current[1] - current[0]
    return total


def build_document(
    *,
    source_name: str,
    source_sha256: str,
    source_bytes: int,
    video: bool,
    duration_s: float,
    profile: str,
    publisher: Transcript,
    detector: Transcript,
    diarization: Diarization | None,
    lines: Sequence[Line],
    marks: Sequence[Mark],
    overview: Sequence[int],
    produced_utc: str,
    names: Mapping[str, str] | None = None,
    speaker_failure: str | None = None,
) -> dict[str, Any]:
    """Assemble the transcript document from the parts of a run."""
    counts = summarise_speakers(lines)
    assigned = dict(default_names(c.label for c in counts))
    if names:
        assigned.update({str(k): str(v) for k, v in names.items() if k in assigned})
    diarization_facts: dict[str, Any] | None = None
    if diarization is not None:
        diarization_facts = {
            "engine": "sherpa_diarization",
            "labels_found": len(diarization.labels),
            "settings": dict(diarization.settings),
            "seconds_per_label": {k: float(v) for k, v in diarization.seconds_per_label.items()},
        }
    review_seconds = _union_seconds((m.start, m.end) for m in marks)
    return {
        "schema": TRANSCRIPT_SCHEMA,
        "version": __version__,
        "produced_utc": produced_utc,
        "profile": profile,
        "source": {
            "name": source_name,
            "sha256": source_sha256,
            "bytes": int(source_bytes),
            "video": bool(video),
        },
        "duration_s": float(duration_s),
        "engines": {
            "publisher": _engine_facts(publisher),
            "detector": _engine_facts(detector),
            "diarization": diarization_facts,
        },
        "speaker_failure": speaker_failure,
        "speakers": [
            {
                "label": c.label,
                "name": assigned.get(c.label) if c.label is not None else None,
                "words": c.words,
                "seconds": float(c.seconds),
            }
            for c in counts
        ],
        "lines": [
            {
                "start": float(line.start),
                "end": float(line.end),
                "speaker": line.speaker,
                "text": line.text,
                "words": [{"s": float(w.start), "e": float(w.end), "w": w.text} for w in line.words],
            }
            for line in lines
        ],
        "marks": [
            {
                "start": float(m.start),
                "end": float(m.end),
                "span_start": float(m.span_start),
                "span_end": float(m.span_end),
                "detector_words": int(m.detector_words),
            }
            for m in marks
        ],
        "review": {
            "marks": len(marks),
            "seconds": float(review_seconds),
            "fraction": float(review_seconds / duration_s) if duration_s > 0.0 else 0.0,
        },
        "overview": {"scale": OVERVIEW_SCALE, "peaks": [int(p) for p in overview]},
    }


def write_document(doc: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    """Write the document atomically as UTF-8 JSON."""
    write_json_atomic(dict(doc), path)


def load_document(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and check a transcript document; ValueError on the wrong schema."""
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict) or doc.get("schema") != TRANSCRIPT_SCHEMA:
        raise ValueError(f"{Path(path).name} is not a {TRANSCRIPT_SCHEMA} document")
    return doc


def speaker_names(doc: Mapping[str, Any]) -> dict[str, str]:
    """Display names by label, from the speaker table of the document."""
    names: dict[str, str] = {}
    for entry in doc.get("speakers", []):
        label = entry.get("label")
        if label is not None:
            names[str(label)] = str(entry.get("name") or label)
    return names


def set_speaker_name(doc: Mapping[str, Any], label: str, name: str) -> dict[str, Any]:
    """A copy of the document with one speaker renamed; unknown labels raise KeyError."""
    updated = json.loads(json.dumps(dict(doc)))
    clean = " ".join(str(name).split())
    if not clean:
        raise ValueError("a speaker name must not be empty")
    for entry in updated.get("speakers", []):
        if entry.get("label") == label:
            entry["name"] = clean
            return updated
    raise KeyError(f"no speaker labelled {label!r}")


def speaker_count(doc: Mapping[str, Any]) -> int:
    """Number of labelled speakers."""
    return sum(1 for entry in doc.get("speakers", []) if entry.get("label") is not None)
