"""Renderers for the per-file outputs.

Every renderer is a pure function of the transcript document, so the plain text, the Word
document and the subtitles can be produced again later, for example after a speaker has been
renamed, without the engines or the audio.
"""

from __future__ import annotations

import os
from pathlib import Path

from twinscribe.outputs.plain_text import render_text
from twinscribe.outputs.subtitles import build_cues, render_srt
from twinscribe.outputs.word_docx import write_docx


def render_all(
    doc: dict,
    text_path: str | os.PathLike[str],
    docx_path: str | os.PathLike[str],
    subtitles_path: str | os.PathLike[str],
    author: str = "",
) -> None:
    """Write the plain text, the Word document and the subtitle file for one document."""
    Path(text_path).write_text(render_text(doc), encoding="utf-8", newline="\n")
    write_docx(doc, docx_path, author=author)
    Path(subtitles_path).write_text(render_srt(build_cues(doc)), encoding="utf-8", newline="\n")


__all__ = ["render_all"]
