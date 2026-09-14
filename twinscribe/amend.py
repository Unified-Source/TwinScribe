"""A listener's resolutions applied to a transcript document.

Words typed after listening to a marked span enter the transcript as a line of their own,
carrying the listener as their source, so that every renderer can show them as such; spans
resolved as silent are recorded on their marks. The engine's words are never altered or
removed: a line the gap falls inside is split around it, every word as it was, so that the
listener's words appear where they were said. The review list keeps the engine's own account
of every gap, so applying a session again, after a resolution changed, first takes the
earlier listener lines out.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twinscribe.runrecord import utc_now

# The outputs package imports this module for the listener's mark, so the document module is
# imported where it is used, and this module can be imported first without a cycle.

SOURCE_LISTENER = "listener"
SESSION_SCHEMA = "twinscribe.review-session.v1"
STATUS_OPEN = "open"
STATUS_NOTHING = "nothing"
STATUS_TEXT = "text"
LISTENER_SUFFIX = "heard on review"
# Word times come from decimal seconds; comparisons carry a tolerance.
_EPSILON = 1e-6


def resolution_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The status, note and speaker of one session entry or recorded resolution.

    The speaker key is present only when the entry names one: a label, or an empty string for
    no speaker; absent, the speaker is inferred when the resolution is applied.
    """
    out: dict[str, Any] = {"status": str(entry.get("status", STATUS_OPEN)), "note": str(entry.get("note", "") or "")}
    speaker = entry.get("speaker")
    if speaker is not None:
        out["speaker"] = str(speaker)
    return out


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
        out.append(resolution_entry(entry))
    return out


def resolutions_from_document(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One resolution per mark from what the document itself records, for a review set whose
    session file is gone (an exported folder, a tidied one): the marks carry the status, the
    words and the speaker they were given, so a pass goes on from them and never removes
    listener lines it did not account for."""
    out: list[dict[str, Any]] = []
    for mark in doc.get("marks", []):
        resolution = mark.get("resolution") if isinstance(mark, Mapping) else None
        if isinstance(resolution, Mapping):
            out.append(resolution_entry(resolution))
        else:
            out.append({"status": STATUS_OPEN, "note": ""})
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


def is_listener_line(line: Mapping[str, Any]) -> bool:
    return line.get("src") == SOURCE_LISTENER


def containing_line(lines: Sequence[Mapping[str, Any]], start: float, end: float) -> Mapping[str, Any] | None:
    """The engine line a gap falls inside: it starts before the gap and ends after it."""
    for line in lines:
        if is_listener_line(line):
            continue
        if float(line.get("start", 0.0)) < start - _EPSILON and float(line.get("end", 0.0)) > end + _EPSILON:
            return line
    return None


def infer_speaker(lines: Sequence[Mapping[str, Any]], start: float, end: float) -> str | None:
    """The speaker a gap most plausibly belongs to: the speaker of the line the gap falls
    inside (the same voice is on both sides of that pause), else the one speaking both
    immediately before and immediately after it; None when the speakers differ or there is no
    neighbour on either side."""
    inside = containing_line(lines, start, end)
    if inside is not None:
        return inside.get("speaker")
    before = [line for line in lines if float(line.get("end", 0.0)) <= start + _EPSILON and not is_listener_line(line)]
    after = [line for line in lines if float(line.get("start", 0.0)) >= end - _EPSILON and not is_listener_line(line)]
    if not before or not after:
        return None
    previous = max(before, key=lambda line: float(line.get("end", 0.0))).get("speaker")
    following = min(after, key=lambda line: float(line.get("start", 0.0))).get("speaker")
    return previous if previous is not None and previous == following else None


def split_line_at(line: Mapping[str, Any], start: float, end: float) -> list[dict[str, Any]]:
    """The line's words before a gap and after it as two lines with the same speaker, every
    word as it was; the line itself, unchanged, when it has no words on one side."""
    before: list[dict[str, Any]] = []
    after: list[dict[str, Any]] = []
    for word in line.get("words", []):
        (after if float(word.get("s", 0.0)) >= end - _EPSILON else before).append(dict(word))
    if not before or not after:
        return [dict(line)]

    def piece(chunk: list[dict[str, Any]]) -> dict[str, Any]:
        out = {key: value for key, value in line.items() if key not in ("start", "end", "text", "words")}
        out["start"] = float(chunk[0].get("s", 0.0))
        out["end"] = float(chunk[-1].get("e", chunk[-1].get("s", 0.0)))
        out["text"] = " ".join(piece_text for piece_text in (str(w.get("w", "")).strip() for w in chunk) if piece_text)
        out["words"] = chunk
        return out

    return [piece(before), piece(after)]


def strip_listener(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of the document without listener lines, resolutions or the applied record; the
    engine lines split around listener words are joined again."""
    out = json.loads(json.dumps(dict(doc)))
    out["lines"] = _joined(line for line in out.get("lines", []) if not is_listener_line(line))
    for mark in out.get("marks", []):
        mark.pop("resolution", None)
    out.pop("review_applied", None)
    return out


def _joined(lines: Any) -> list[dict[str, Any]]:
    """Consecutive pieces of one split line, marked as such, made one line again."""
    out: list[dict[str, Any]] = []
    for line in lines:
        if out and line.get("split_from") is not None and line.get("split_from") == out[-1].get("split_from"):
            previous = out[-1]
            words = list(previous.get("words", [])) + list(line.get("words", []))
            previous["words"] = words
            previous["end"] = float(line.get("end", previous.get("end", 0.0)))
            previous["text"] = " ".join(t for t in (str(w.get("w", "")).strip() for w in words) if t)
            continue
        out.append(line)
    for line in out:
        line.pop("split_from", None)
    return out


def apply_resolutions(doc: Mapping[str, Any], resolutions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A new document with the resolutions applied; `resolutions` has one entry per mark.

    A resolution with words gains a listener line at the gap; when the gap falls inside an
    engine line, that line is split around it so the words read in order, and the listener
    line takes its speaker unless the resolution names one (`speaker`: a label, or an empty
    string for none).
    """
    from twinscribe.outputs.transcript_doc import recount_speakers

    marks = list(doc.get("marks", []))
    if len(resolutions) != len(marks):
        raise ValueError(f"{len(resolutions)} resolutions for {len(marks)} marks")
    out = strip_listener(doc)
    lines: list[dict[str, Any]] = list(out.get("lines", []))
    n_text = n_nothing = n_open = n_words = 0
    for number, (mark, resolution) in enumerate(zip(out["marks"], resolutions)):
        status = str(resolution.get("status", STATUS_OPEN))
        note = str(resolution.get("note", "") or "").strip()
        if status == STATUS_TEXT and note:
            span_start = float(mark.get("span_start", mark.get("start", 0.0)))
            span_end = float(mark.get("span_end", mark.get("end", span_start)))
            if span_end <= span_start:
                span_end = span_start + 0.5
            inside = containing_line(lines, span_start, span_end)
            if inside is not None:
                position = next(i for i, line in enumerate(lines) if line is inside)
                pieces = split_line_at(inside, span_start, span_end)
                if len(pieces) == 2:
                    for piece in pieces:
                        piece["split_from"] = number
                lines[position : position + 1] = pieces
            words = listener_words(note, span_start, span_end)
            chosen = resolution.get("speaker")
            speaker = infer_speaker(lines, span_start, span_end) if chosen is None else (str(chosen) or None)
            lines.append(
                {
                    "start": span_start,
                    "end": span_end,
                    "speaker": speaker,
                    "text": " ".join(w["w"] for w in words),
                    "words": words,
                    "src": SOURCE_LISTENER,
                }
            )
            mark["resolution"] = {"status": STATUS_TEXT, "note": note, "speaker": speaker if speaker is not None else ""}
            n_text += 1
            n_words += len(words)
        elif status == STATUS_NOTHING:
            mark["resolution"] = {"status": STATUS_NOTHING, "note": ""}
            n_nothing += 1
        else:
            n_open += 1
    lines.sort(key=lambda line: (float(line.get("start", 0.0)), float(line.get("end", 0.0))))
    out["lines"] = lines
    recount_speakers(out)
    out["review_applied"] = {
        "applied_utc": utc_now(),
        "text": n_text,
        "nothing": n_nothing,
        "open": n_open,
        "listener_words": n_words,
    }
    return out


def listener_line_for(doc: Mapping[str, Any], mark_index: int) -> Mapping[str, Any] | None:
    """The listener line written for a mark, found by the mark's span start; None when the
    mark carries no words."""
    marks = doc.get("marks", [])
    if not (0 <= mark_index < len(marks)):
        return None
    mark = marks[mark_index]
    start = float(mark.get("span_start", mark.get("start", 0.0)))
    for line in doc.get("lines", []):
        if is_listener_line(line) and abs(float(line.get("start", 0.0)) - start) < 1e-3:
            return line
    return None


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


def all_checked(doc: Mapping[str, Any]) -> bool:
    """True when a listener has decided every mark of the review list."""
    applied = doc.get("review_applied")
    if not isinstance(applied, Mapping):
        return False
    checked = int(applied.get("text", 0)) + int(applied.get("nothing", 0))
    return checked > 0 and int(applied.get("open", 0)) == 0


def apply_session(
    transcript_path: str | os.PathLike[str],
    session_path: str | os.PathLike[str],
    author: str = "",
    render: bool = True,
) -> dict[str, Any]:
    """Apply a review session to the transcript document beside it and render the outputs
    again; returns the revised document. ValueError when the session does not fit the marks;
    OutputsNotWritten (an OSError) when the document was written but an output could not be."""
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
