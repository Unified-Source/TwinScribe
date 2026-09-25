"""A playable copy of a recording for the window's player. The platform's media player reads
the common types; a recording it cannot read, a container or a codec it has no decoder for,
is decoded once through the pipeline's own decoder to a WAV under the application home and
played from there, and a recording in parts is joined the way the pipeline joins it. The copy
is made in a thread so that the window stays responsive.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from twinscribe import audio
from twinscribe.paths import PLAY_FOLDER, copy_is_current, play_dir, playable_copy_path
from twinscribe.pipeline import Parts

__all__ = ["PLAY_FOLDER", "PlayableCopy", "copy_is_current", "partial_path", "play_dir", "playable_copy_path"]


def partial_path(copy: Path) -> Path:
    """The name a copy is written under until it is complete: `<stem>.part.wav` beside it."""
    return copy.with_name(f"{copy.stem}.part.wav")


class PlayableCopy(QThread):
    """Decodes one recording to its playable copy, in the engines' own format (16 kHz mono);
    a recording in parts is decoded part by part and joined at the parts' offsets. `ready`
    carries the copy's path; `failed` the reason. A copy already current is not made again."""

    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, source: Path, parts: Parts | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.source = Path(source)
        self.parts = parts
        self.target = playable_copy_path(self.source)

    def run(self) -> None:
        # Written under a name that keeps the .wav ending, which the decoder needs to choose
        # the container, and renamed only when complete; a failed decode leaves nothing behind.
        partial = partial_path(self.target)
        pieces: list[Path] = []
        try:
            if not copy_is_current(self.source, self.target):
                if self.parts is None:
                    audio.decode_to_wav(self.source, partial)
                else:
                    placed: list[tuple[Path, float]] = []
                    for index, (part, offset) in enumerate(zip(self.parts.paths, self.parts.offsets_s)):
                        piece = self.target.with_name(f"{self.target.stem}.piece{index + 1}.wav")
                        audio.decode_to_wav(part, piece)
                        pieces.append(piece)
                        placed.append((piece, offset))
                    audio.join_wavs(placed, partial)
                os.replace(partial, self.target)
            self.ready.emit(self.target)
        except Exception as exc:  # noqa: BLE001 - the window shows the reason
            try:
                partial.unlink()
            except OSError:
                pass
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            for piece in pieces:
                try:
                    piece.unlink()
                except OSError:
                    pass
