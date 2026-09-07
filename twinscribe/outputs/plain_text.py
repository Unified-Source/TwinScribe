"""Plain-text rendering of a transcript document: a header with the engines and the speaker
summary, then one timestamped, speaker-labelled line per line of transcript.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from twinscribe.labelling import UNLABELLED_NAME
from twinscribe.outputs.transcript_doc import clock, speaker_names

DRAFT_NOTICE = "A transcript is a draft until it has been verified against the recording."


def _engine_line(facts: Mapping[str, Any] | None) -> str:
    if not facts:
        return "none"
    parts = [str(facts.get("engine") or ""), str(facts.get("model") or "")]
    text = " ".join(p for p in parts if p)
    preset = facts.get("preset")
    return f"{text} ({preset})" if preset else text


def header_lines(doc: Mapping[str, Any]) -> list[str]:
    """The header block: file, duration, engines, speaker summary, review pointer."""
    source = doc.get("source", {})
    engines = doc.get("engines", {})
    speakers = doc.get("speakers", [])
    labelled = [s for s in speakers if s.get("label") is not None]
    noun = "speaker" if len(labelled) == 1 else "speakers"
    lines = [
        str(source.get("name", "")),
        f"Duration {clock(float(doc.get('duration_s', 0.0)), tenths=False)}   |   "
        f"{len(labelled)} {noun}   |   produced {doc.get('produced_utc', '')}   |   "
        f"twinscribe {doc.get('version', '')}, quality level {doc.get('profile', '')}",
        f"Published engine: {_engine_line(engines.get('publisher'))}",
        f"Checked against: {_engine_line(engines.get('detector'))}; its text is never published",
    ]
    diarization = engines.get("diarization")
    if diarization:
        threshold = (diarization.get("settings") or {}).get("threshold")
        detail = f", clustering threshold {threshold}" if threshold is not None else ""
        lines.append(f"Speaker labels: {diarization.get('engine', '')}{detail}")
    failure = doc.get("speaker_failure")
    if failure:
        lines.append(f"Speaker labelling did not complete: {failure}")
    lines.append("")
    lines.append("Speakers")
    names = speaker_names(doc)
    for entry in speakers:
        label = entry.get("label")
        name = names.get(label, label) if label is not None else UNLABELLED_NAME
        words = int(entry.get("words", 0))
        word_noun = "word" if words == 1 else "words"
        lines.append(f"  {name:<24} {words:>7,} {word_noun:<5} {clock(float(entry.get('seconds', 0.0)))}")
    if not speakers:
        lines.append("  (no words were published)")
    lines.append("")
    review = doc.get("review", {})
    marks = int(review.get("marks", 0))
    stem = str(source.get("name", "")).rsplit(".", 1)[0]
    if marks:
        span_noun = "span" if marks == 1 else "spans"
        share = 100.0 * float(review.get("fraction", 0.0))
        lines.append(
            f"Review list: {marks} {span_noun} where speech may be missing, {share:.1f}% of the recording; "
            f"see {stem}.review.json."
        )
    else:
        lines.append("Review list: no span where speech may be missing was found.")
    lines.append(DRAFT_NOTICE)
    return lines


def transcript_lines(doc: Mapping[str, Any]) -> list[str]:
    """One text line per transcript line, with a blank line between speakers."""
    names = speaker_names(doc)
    out: list[str] = []
    previous: object = object()
    for line in doc.get("lines", []):
        label = line.get("speaker")
        name = names.get(label, label) if label is not None else UNLABELLED_NAME
        if out and label != previous:
            out.append("")
        out.append(f"[{clock(float(line.get('start', 0.0)))}] {name}: {line.get('text', '')}")
        previous = label
    return out


def render_text(doc: Mapping[str, Any]) -> str:
    """The whole plain-text file."""
    parts = header_lines(doc) + ["", "Transcript", ""] + transcript_lines(doc)
    return "\n".join(parts).rstrip("\n") + "\n"
