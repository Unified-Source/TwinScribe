"""Export of one recording's outputs to a chosen folder in chosen formats: the plain text, the
Word document and the subtitles rendered again from the transcript document (so a renamed
speaker or a listener's edit is carried), and the transcript document, the review list and
the run record copied as they are.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from twinscribe.outputs.plain_text import render_text, transcript_lines
from twinscribe.outputs.subtitles import build_cues, render_srt, render_vtt
from twinscribe.outputs.transcript_doc import write_document
from twinscribe.outputs.word_docx import write_docx

FORMAT_TEXT = "text"
FORMAT_DOCX = "docx"
FORMAT_SRT = "srt"
FORMAT_VTT = "vtt"
FORMAT_TRANSCRIPT = "transcript"
FORMAT_REVIEW = "review"
FORMAT_RUN = "run"

# key, title shown to a person, file suffix
FORMATS: tuple[tuple[str, str, str], ...] = (
    (FORMAT_TEXT, "Plain text", ".txt"),
    (FORMAT_DOCX, "Word document", ".docx"),
    (FORMAT_SRT, "Subtitles, SRT", ".srt"),
    (FORMAT_VTT, "Subtitles, WebVTT", ".vtt"),
    (FORMAT_TRANSCRIPT, "Transcript document (JSON)", ".transcript.json"),
    (FORMAT_REVIEW, "Review list (JSON)", ".review.json"),
    (FORMAT_RUN, "Run record (JSON)", ".run.json"),
)
DEFAULT_FORMATS: tuple[str, ...] = (FORMAT_TEXT, FORMAT_DOCX, FORMAT_SRT)
_SUFFIX = {key: suffix for key, _, suffix in FORMATS}
_TITLE = {key: title for key, title, _ in FORMATS}


def format_title(key: str) -> str:
    return _TITLE[key]


def format_suffix(key: str) -> str:
    return _SUFFIX[key]


def transcript_text(doc: Mapping[str, Any]) -> str:
    """The transcript lines alone, for the clipboard."""
    return "\n".join(transcript_lines(doc)).rstrip("\n") + "\n"


def export_outputs(
    doc: Mapping[str, Any],
    destination: str | os.PathLike[str],
    stem: str,
    formats: Iterable[str],
    author: str = "",
    sources: Mapping[str, str | os.PathLike[str]] | None = None,
) -> list[Path]:
    """Write the chosen formats for `doc` into `destination` as `<stem><suffix>`.

    `sources` maps the copied formats (transcript, review, run) to the files they are copied
    from; a copied format whose source is absent is skipped, except the transcript document,
    which is written from `doc` instead. Returns the paths written, in the order of FORMATS.
    """
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    wanted = set(formats)
    unknown = wanted - set(_SUFFIX)
    if unknown:
        raise KeyError(f"unknown export format(s): {', '.join(sorted(unknown))}")
    if not stem.strip():
        raise ValueError("the file name stem must not be empty")
    given = {k: Path(v) for k, v in (sources or {}).items()}
    written: list[Path] = []
    cues = None
    for key, _, suffix in FORMATS:
        if key not in wanted:
            continue
        target = folder / f"{stem}{suffix}"
        if key == FORMAT_TEXT:
            target.write_text(render_text(doc), encoding="utf-8", newline="\n")
        elif key == FORMAT_DOCX:
            write_docx(doc, target, author=author)
        elif key in (FORMAT_SRT, FORMAT_VTT):
            if cues is None:
                cues = build_cues(doc)
            text = render_srt(cues) if key == FORMAT_SRT else render_vtt(cues)
            target.write_text(text, encoding="utf-8", newline="\n")
        elif key == FORMAT_TRANSCRIPT:
            source = given.get(key)
            if source is not None and source.is_file() and source.resolve() != target.resolve():
                shutil.copyfile(source, target)
            elif source is None or not source.is_file():
                write_document(doc, target)
        else:
            source = given.get(key)
            if source is None or not source.is_file():
                continue
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
        written.append(target)
    return written
