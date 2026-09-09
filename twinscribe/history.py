"""A record of every recording transcribed on this machine, one JSON file under the
application home, so the window can show what was done before, where the outputs went and
how to reach them again after the library was cleared. The batch runner appends to it, so the
command line and the window share one history; nothing else writes it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from twinscribe.paths import app_home
from twinscribe.runrecord import utc_now, write_json_atomic

HISTORY_SCHEMA = "twinscribe.history.v1"
HISTORY_FILE = "history.json"
MAX_ENTRIES = 2000


@dataclass(frozen=True)
class HistoryEntry:
    """One transcribed recording: where it was, what was produced and when."""

    path: str
    name: str
    sha256: str
    duration_s: float
    profile: str
    marks: int
    speakers: int
    produced_utc: str
    outputs: dict[str, str] = field(default_factory=dict)
    publisher: str = ""
    detector: str = ""

    @property
    def transcript_path(self) -> Path | None:
        value = self.outputs.get("transcript")
        return Path(value) if value else None

    @property
    def outputs_present(self) -> bool:
        """True when the transcript document is still where it was written."""
        path = self.transcript_path
        return path is not None and path.is_file()

    @property
    def folder(self) -> Path:
        path = self.transcript_path
        return path.parent if path is not None else Path(self.path).parent

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "sha256": self.sha256,
            "duration_s": float(self.duration_s),
            "profile": self.profile,
            "marks": int(self.marks),
            "speakers": int(self.speakers),
            "produced_utc": self.produced_utc,
            "outputs": dict(self.outputs),
            "publisher": self.publisher,
            "detector": self.detector,
        }

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "HistoryEntry":
        outputs = doc.get("outputs")
        return cls(
            path=str(doc.get("path", "")),
            name=str(doc.get("name", "")) or Path(str(doc.get("path", ""))).name,
            sha256=str(doc.get("sha256", "")),
            duration_s=float(doc.get("duration_s", 0.0) or 0.0),
            profile=str(doc.get("profile", "")),
            marks=int(doc.get("marks", 0) or 0),
            speakers=int(doc.get("speakers", 0) or 0),
            produced_utc=str(doc.get("produced_utc", "")),
            outputs={str(k): str(v) for k, v in outputs.items()} if isinstance(outputs, Mapping) else {},
            publisher=str(doc.get("publisher", "")),
            detector=str(doc.get("detector", "")),
        )


def history_path(root: str | os.PathLike[str] | None = None) -> Path:
    """Where the history lives: under the given root, else under the application home."""
    base = Path(root) if root is not None else app_home()
    return base / HISTORY_FILE


def read_history(path: str | os.PathLike[str] | None = None) -> list[HistoryEntry]:
    """Every entry, newest first; empty when the file is absent, unreadable or of another schema."""
    target = Path(path) if path is not None else history_path()
    if not target.is_file():
        return []
    try:
        with open(target, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError):
        return []
    if not isinstance(doc, dict) or doc.get("schema") != HISTORY_SCHEMA:
        return []
    entries = doc.get("entries")
    if not isinstance(entries, list):
        return []
    out: list[HistoryEntry] = []
    for item in entries:
        if isinstance(item, Mapping) and item.get("path"):
            out.append(HistoryEntry.from_dict(item))
    return out


def write_history(entries: Iterable[HistoryEntry], path: str | os.PathLike[str] | None = None) -> Path:
    """Write the entries atomically, newest first as given, and return the path."""
    target = Path(path) if path is not None else history_path()
    doc = {"schema": HISTORY_SCHEMA, "updated_utc": utc_now(), "entries": [e.to_dict() for e in entries]}
    write_json_atomic(doc, target)
    return target


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def entry_from_result(result: Any) -> HistoryEntry:
    """The entry for a pipeline FileResult (source, outputs, document, run record)."""
    doc = result.document
    engines = doc.get("engines") or {}
    speakers = [s for s in doc.get("speakers", []) if s.get("label") is not None]
    outputs = result.outputs
    return HistoryEntry(
        path=str(result.source),
        name=str(doc.get("source", {}).get("name") or Path(str(result.source)).name),
        sha256=str(doc.get("source", {}).get("sha256", "")),
        duration_s=float(doc.get("duration_s", 0.0)),
        profile=str(doc.get("profile", "")),
        marks=int((doc.get("review") or {}).get("marks", 0)),
        speakers=len(speakers),
        produced_utc=str(doc.get("produced_utc", "")),
        outputs={
            "transcript": str(outputs.transcript),
            "text": str(outputs.text),
            "docx": str(outputs.docx),
            "subtitles": str(outputs.subtitles),
            "review": str(outputs.review),
            "run": str(outputs.run),
        },
        publisher=str((engines.get("publisher") or {}).get("model", "")),
        detector=str((engines.get("detector") or {}).get("model", "")),
    )


def append_entry(entry: HistoryEntry, path: str | os.PathLike[str] | None = None) -> list[HistoryEntry]:
    """Put the entry first, replacing any earlier entry for the same recording path; cap the
    length; write; return the new list."""
    entries = [e for e in read_history(path) if not _same_path(e.path, entry.path)]
    entries.insert(0, entry)
    del entries[MAX_ENTRIES:]
    write_history(entries, path)
    return entries


def remove_entries(paths: Iterable[str], path: str | os.PathLike[str] | None = None) -> list[HistoryEntry]:
    """Drop the entries for the given recording paths; write; return the new list."""
    wanted = list(paths)
    entries = [e for e in read_history(path) if not any(_same_path(e.path, p) for p in wanted)]
    write_history(entries, path)
    return entries


def clear_history(path: str | os.PathLike[str] | None = None) -> None:
    """Leave an empty history behind."""
    write_history([], path)
