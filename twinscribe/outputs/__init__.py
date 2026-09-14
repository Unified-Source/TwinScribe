"""Renderers for the per-file outputs.

Every renderer is a pure function of the transcript document, so the plain text, the Word
document and the subtitles can be produced again later, for example after a speaker has been
renamed, without the engines or the audio. The three are written each on its own: a file
held by another program (Word holds its document open) must not stop the others.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from twinscribe.outputs.plain_text import render_text
from twinscribe.outputs.subtitles import build_cues, render_srt
from twinscribe.outputs.word_docx import write_docx

OUTPUT_TITLES: dict[str, str] = {"text": "plain text", "docx": "Word document", "subtitles": "subtitle file"}


def describe_failures(failures: Mapping[str, str]) -> str:
    """One clause naming the outputs that could not be written and why."""
    parts = [f"the {OUTPUT_TITLES.get(key, key)} ({reason})" for key, reason in failures.items()]
    return "could not write " + "; ".join(parts)


class OutputsNotWritten(OSError):
    """Some outputs could not be written: `failures` names each with its reason and `written`
    lists those that were."""

    def __init__(self, failures: Mapping[str, str], written: Sequence[str]) -> None:
        self.failures = dict(failures)
        self.written = list(written)
        super().__init__(describe_failures(failures))


def _reason(path: str | os.PathLike[str], exc: OSError) -> str:
    text = f"{Path(path).name}: {exc}"
    if isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32):
        text += "; is it open in another program?"
    return text


def render_outputs(
    doc: dict,
    text_path: str | os.PathLike[str],
    docx_path: str | os.PathLike[str],
    subtitles_path: str | os.PathLike[str],
    author: str = "",
) -> dict[str, str]:
    """Write the plain text, the Word document and the subtitle file, each on its own, and
    return the failures by output name with their reasons; empty when all three were written."""
    failures: dict[str, str] = {}
    writers = (
        ("text", text_path, lambda: Path(text_path).write_text(render_text(doc), encoding="utf-8", newline="\n")),
        ("docx", docx_path, lambda: write_docx(doc, docx_path, author=author)),
        (
            "subtitles",
            subtitles_path,
            lambda: Path(subtitles_path).write_text(render_srt(build_cues(doc)), encoding="utf-8", newline="\n"),
        ),
    )
    for key, path, write in writers:
        try:
            write()
        except OSError as exc:
            failures[key] = _reason(path, exc)
    return failures


def render_all(
    doc: dict,
    text_path: str | os.PathLike[str],
    docx_path: str | os.PathLike[str],
    subtitles_path: str | os.PathLike[str],
    author: str = "",
) -> None:
    """Write the three outputs for one document; every one that can be written is, and
    OutputsNotWritten names the rest."""
    failures = render_outputs(doc, text_path, docx_path, subtitles_path, author=author)
    if failures:
        raise OutputsNotWritten(failures, [key for key in OUTPUT_TITLES if key not in failures])


__all__ = ["OUTPUT_TITLES", "OutputsNotWritten", "describe_failures", "render_all", "render_outputs"]
