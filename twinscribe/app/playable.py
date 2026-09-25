"""A playable copy of a recording for the window's player. The platform's media player reads
the common types; a recording it cannot read, a container or a codec it has no decoder for,
is decoded once through the pipeline's own decoder to a WAV under the application home and
played from there. The copy is made in a thread so that the window stays responsive.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from twinscribe import audio
from twinscribe.paths import app_home

PLAY_FOLDER = "play"


def play_dir() -> Path:
    """The folder of playable copies under the application home; created when absent."""
    folder = app_home() / PLAY_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def playable_copy_path(source: Path) -> Path:
    """Where the copy of `source` goes: the recording's stem and a digest of its resolved path,
    so that two recordings of one name in two folders keep two copies."""
    key = os.path.normcase(str(Path(source).resolve()))
    tag = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return play_dir() / f"{Path(source).stem}.{tag}.wav"


def partial_path(copy: Path) -> Path:
    """The name a copy is written under until it is complete: `<stem>.part.wav` beside it."""
    return copy.with_name(f"{copy.stem}.part.wav")


def copy_is_current(source: Path, copy: Path) -> bool:
    """A copy counts when it exists and is not older than the recording it was made from."""
    try:
        return copy.is_file() and copy.stat().st_mtime >= Path(source).stat().st_mtime
    except OSError:
        return False


class PlayableCopy(QThread):
    """Decodes one recording to its playable copy, in the engines' own format (16 kHz mono).
    `ready` carries the copy's path; `failed` the reason. A copy already current is not made
    again."""

    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, source: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.source = Path(source)
        self.target = playable_copy_path(self.source)

    def run(self) -> None:
        # Written under a name that keeps the .wav ending, which the decoder needs to choose
        # the container, and renamed only when complete; a failed decode leaves nothing behind.
        partial = partial_path(self.target)
        try:
            if not copy_is_current(self.source, self.target):
                audio.decode_to_wav(self.source, partial)
                os.replace(partial, self.target)
            self.ready.emit(self.target)
        except Exception as exc:  # noqa: BLE001 - the window shows the reason
            try:
                partial.unlink()
            except OSError:
                pass
            self.failed.emit(f"{type(exc).__name__}: {exc}")
