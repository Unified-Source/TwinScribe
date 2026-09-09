"""A listener's resolutions applied to a transcript document.

Words typed after listening to a marked span enter the transcript as a line of their own,
carrying the listener as their source, so that every renderer can show them as such; spans
resolved as silent are recorded on their marks. The engine's words are never altered or
removed, and the review list keeps the engine's own account of every gap, so applying a
session again, after a resolution changed, first takes the earlier listener lines out.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twinscribe.runrecord import utc_now

SOURCE_LISTENER = "listener"
SESSION_SCHEMA = "twinscribe.review-session.v1"
STATUS_OPEN = "open"
STATUS_NOTHING = "nothing"
STATUS_TEXT = "text"
LISTENER_SUFFIX = "heard on review"


def read_resolutions(path: str | os.PathLike[str], expected: int) -> list[dict[str, Any]] | None:
    """The resolutions of a review session file, when it has one entry per mark; else None."""
    target = Path(path)
    if not target.is_file():
        return None
    try:
        with open(target, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("schema") != SESSION_SCHEMA:
        return None
    marks = doc.get("marks")
    if not isinstance(marks, list) or len(marks) != expected:
        return None
    out: list[dict[str, Any]] = []
    for entry in marks:
        if not isinstance(entry, Mapping):
            return None
        out.append({"status": str(entry.get("status", STATUS_OPEN)), "note": str(entry.get("note", "") or "")})
    return out


def listener_words(note: str, start: float, end: float) -> list[dict[str, Any]]:
    """The typed words spread evenly across the span, each marked as the listener's."""
    tokens = note.split()
    if not tokens:
        return []
    span = max(0.0, float(end) - float(start))
    step = span / len(tokens)
    return [
        {"s": float(start) + i * step, "e": float(start) + (i + 1) * step, "w": token, "src": SOURCE_LISTENER}
        for i, token in enumerate(tokens)
    ]


def infer_speaker(lines: Sequence[Mapping[str, Any]], start: float, end: float) -> str | None:
    """The speaker a gap most plausibly belongs to: the one speaking both before and after it;
    None when the speakers differ or there is no neighbour on either side."""
    before = [line for line in lines if float(line.get("end", 0.0)) <= start + 1e-6 and line.get("src") != SOURCE_LISTENER]
    after = [line for line in lines if float(line.get("start", 0.0)) >= end - 1e-6 and line.get("src") != SOURCE_LISTENER]
    if not before or not after:
        return None
    previous = max(before, key=lambda line: float(line.get("end", 0.0))).get("speaker")
    following = min(after, key=lambda line: float(line.get("start", 0.0))).get("speaker")
    return previous if previous is not None and previous == following else None


def is_listener_line(line: Mapping[str, Any]) -> bool:
    return line.get("src") == SOURCE_LISTENER


def strip_listener(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of the document without listener lines, resolutions or the applied record."""
    out = json.loads(json.dumps(dict(doc)))
    out["lines"] = [line for line in out.get("lines", []) if not is_listener_line(line)]
    for mark in out.get("marks", []):
        mark.pop("resolution", None)
    out.pop("review_applied", None)
    return out


def _recount_speakers(doc: dict[str, Any]) -> None:
    """Words and seconds per label from the lines, keeping the entries' order and names."""
    words: dict[Any, int] = {}
    seconds: dict[Any, float] = {}
    order: list[Any] = []
    for line in doc.get("lines", []):
        label = line.get("speaker")
        if label not in words:
            order.append(label)
            words[label] = 0
            seconds[label] = 0.0
        words[label] += len(line.get("words", []))
        seconds[label] += max(0.0, float(line.get("end", 0.0)) - float(line.get("start", 0.0)))
    existing = {entry.get("label"): entry for entry in doc.get("speakers", [])}
    speakers: list[dict[str, Any]] = []
    for entry in doc.get("speakers", []):
        label = entry.get("label")
        speakers.append({**entry, "words": words.get(label, 0), "seconds": float(seconds.get(label, 0.0))})
    for label in order:
        if label not in existing:
            speakers.append({"label": label, "name": None, "words": words[label], "seconds": float(seconds[label])})
    doc["speakers"] = speakers


def apply_resolutions(doc: Mapping[str, Any], resolutions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A new document with the resolutions applied; `resolutions` has one entry per mark."""
    marks = list(doc.get("marks", []))
    if len(resolutions) != len(marks):
        raise ValueError(f"{len(resolutions)} resolutions for {len(marks)} marks")
    out = strip_listener(doc)
    lines: list[dict[str, Any]] = list(out.get("lines", []))
    n_text = n_nothing = n_open = n_words = 0
    for mark, resolution in zip(out["marks"], resolutions):
        status = str(resolution.get("status", STATUS_OPEN))
        note = str(resolution.get("note", "") or "").strip()
        if status == STATUS_TEXT and note:
            span_start = float(mark.get("span_start", mark.get("start", 0.0)))
            span_end = float(mark.get("span_end", mark.get("end", span_start)))
            if span_end <= span_start:
                span_end = span_start + 0.5
            words = listener_words(note, span_start, span_end)
            lines.append(
                {
                    "start": span_start,
                    "end": span_end,
                    "speaker": infer_speaker(lines, span_start, span_end),
                    "text": " ".join(w["w"] for w in words),
                    "words": words,
                    "src": SOURCE_LISTENER,
                }
            )
            mark["resolution"] = {"status": STATUS_TEXT, "note": note}
            n_text += 1
            n_words += len(words)
        elif status == STATUS_NOTHING:
            mark["resolution"] = {"status": STATUS_NOTHING, "note": ""}
            n_nothing += 1
        else:
            n_open += 1
    lines.sort(key=lambda line: (float(line.get("start", 0.0)), float(line.get("end", 0.0))))
    out["lines"] = lines
    _recount_speakers(out)
    out["review_applied"] = {
        "applied_utc": utc_now(),
        "text": n_text,
        "nothing": n_nothing,
        "open": n_open,
        "listener_words": n_words,
    }
    return out


def reviewed_note(doc: Mapping[str, Any]) -> str | None:
    """One sentence for a header when a listener's review was applied, else None."""
    applied = doc.get("review_applied")
    if not isinstance(applied, Mapping):
        return None
    checked = int(applied.get("text", 0)) + int(applied.get("nothing", 0))
    if checked == 0:
        return None
    span_noun = "span" if checked == 1 else "spans"
    parts = [f"Reviewed by a listener: {checked} {span_noun} checked"]
    text = int(applied.get("text", 0))
    if text:
        parts.append(
            f"{text} {'carries' if text == 1 else 'carry'} words typed after listening, shown as {LISTENER_SUFFIX}"
        )
    open_count = int(applied.get("open", 0))
    if open_count:
        parts.append(f"{open_count} still open")
    return "; ".join(parts) + "."


def apply_session(
    transcript_path: str | os.PathLike[str],
    session_path: str | os.PathLike[str],
    author: str = "",
    render: bool = True,
) -> dict[str, Any]:
    """Apply a review session to the transcript document beside it and render the outputs
    again; returns the revised document. ValueError when the session does not fit the marks."""
    from twinscribe.outputs import render_all
    from twinscribe.outputs.transcript_doc import load_document, write_document
    from twinscribe.pipeline import output_paths

    transcript = Path(transcript_path)
    doc = load_document(transcript)
    resolutions = read_resolutions(session_path, len(doc.get("marks", [])))
    if resolutions is None:
        raise ValueError(f"the session at {Path(session_path).name} does not match the transcript's marks")
    revised = apply_resolutions(doc, resolutions)
    write_document(revised, transcript)
    if render:
        stem = transcript.name[: -len(".transcript.json")] if transcript.name.endswith(".transcript.json") else transcript.stem
        paths = output_paths(transcript.parent / stem, transcript.parent)
        render_all(revised, paths.text, paths.docx, paths.subtitles, author=author)
    return revised
