"""The transcript pane: speaker-labelled lines that follow the audio.

The document is laid out once per recording as one text block per line, with the time in a
narrow gutter, the speaker in colour and the words in the reading face. Spans the review list
flagged appear between the lines as gap markers; a listener's resolution from a review session
is shown under its marker. Stretches without speech (silence, music, background noise, other
sound) appear as muted markers naming what is there. As the audio plays, the line under the
playhead is highlighted and kept in view. Clicking a time, a gap marker or a scene marker, or
double-clicking a line, seeks to it.
"""

from __future__ import annotations

import time
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from typing import Any

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QMouseEvent,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextFormat,
    QWheelEvent,
)
from PySide6.QtWidgets import QTextEdit, QWidget

from twinscribe.app.theme import Theme, theme_for, with_alpha
from twinscribe.labelling import UNLABELLED_NAME
from twinscribe.amend import LISTENER_SUFFIX, SOURCE_LISTENER
from twinscribe.outputs.transcript_doc import clock, scene_phrase, speaker_names

KIND_LINE = "line"
KIND_MARK = "mark"
KIND_SCENE = "scene"
FOLLOW_HOLD_S = 4.0
GUTTER_CHARS = 9


class TranscriptView(QTextEdit):
    """Read-only transcript that highlights the current line and seeks on click.

    `seek_requested` carries a time in seconds; `mark_requested` carries the index of a gap
    marker whose span the owner should play.
    """

    seek_requested = Signal(float)
    mark_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None, theme: Theme | None = None) -> None:
        super().__init__(parent)
        self._theme = theme if theme is not None else theme_for(False)
        self._doc: dict[str, Any] | None = None
        self._session: list[dict[str, Any]] | None = None
        self._line_starts: list[float] = []
        self._line_blocks: list[int] = []
        self._block_info: dict[int, tuple[str, int]] = {}
        self._gutter_end: dict[int, int] = {}
        self._current_line: int | None = None
        self._follow = True
        self._hold_until = 0.0
        self.setObjectName("transcript")
        self.setReadOnly(True)
        self.setCursorWidth(0)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setFrameShape(QTextEdit.Shape.NoFrame)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setPlaceholderText("Open a recording or drop one here.")
        self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        self.document().setDocumentMargin(18.0)

    # ----- appearance ----------------------------------------------------------------------

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        if self._doc is not None:
            self.set_document(self._doc, self._session)

    def set_follow(self, follow: bool) -> None:
        """Keep the current line in view while playing (a manual scroll pauses this briefly)."""
        self._follow = bool(follow)
        self._hold_until = 0.0
        if self._follow and self._current_line is not None:
            self._scroll_to_block(self._line_blocks[self._current_line])

    def follow(self) -> bool:
        return self._follow

    # ----- content -------------------------------------------------------------------------

    def set_document(self, doc: Mapping[str, Any] | None, session: Sequence[Mapping[str, Any]] | None = None) -> None:
        """Lay out a transcript document (or clear the pane when None)."""
        self._doc = dict(doc) if doc is not None else None
        self._session = [dict(s) for s in session] if session is not None else None
        self._line_starts = []
        self._line_blocks = []
        self._block_info = {}
        self._gutter_end = {}
        self._current_line = None
        self.setExtraSelections([])
        if self._doc is None:
            self.clear()
            return
        self._build()

    def document_loaded(self) -> bool:
        return self._doc is not None

    def line_count(self) -> int:
        return len(self._line_blocks)

    def current_line(self) -> int | None:
        return self._current_line

    def _formats(self) -> tuple[QTextCharFormat, QTextCharFormat, QTextCharFormat, QTextCharFormat, QTextCharFormat]:
        base_font = QFont(self.font())
        base_font.setPointSizeF(11.0)
        time_font = QFont(base_font)
        time_font.setFamilies(["Consolas", "Cascadia Mono", "SF Mono", "Menlo", "DejaVu Sans Mono", "Liberation Mono", "monospace"])
        time_font.setStyleHint(QFont.StyleHint.Monospace)
        time_font.setPointSizeF(9.0)
        name_font = QFont(base_font)
        name_font.setBold(True)
        name_font.setPointSizeF(10.0)
        mark_font = QFont(base_font)
        mark_font.setItalic(True)
        mark_font.setPointSizeF(9.5)

        time_format = QTextCharFormat()
        time_format.setFont(time_font)
        time_format.setForeground(self._theme.muted)
        name_format = QTextCharFormat()
        name_format.setFont(name_font)
        text_format = QTextCharFormat()
        text_format.setFont(base_font)
        text_format.setForeground(self.palette().color(self.palette().ColorRole.Text))
        mark_format = QTextCharFormat()
        mark_format.setFont(mark_font)
        mark_format.setForeground(self._theme.warning)
        resolved_format = QTextCharFormat()
        resolved_format.setFont(mark_font)
        resolved_format.setForeground(self._theme.success)
        return time_format, name_format, text_format, mark_format, resolved_format

    def _build(self) -> None:
        assert self._doc is not None
        doc = self._doc
        names = speaker_names(doc)
        labelled = [s["label"] for s in doc.get("speakers", []) if s.get("label") is not None]
        colours = {label: self._theme.speaker(i) for i, label in enumerate(labelled)}
        time_format, name_format, text_format, mark_format, resolved_format = self._formats()

        events: list[tuple[float, int, str, int]] = []
        for index, line in enumerate(doc.get("lines", [])):
            events.append((float(line.get("start", 0.0)), 2, KIND_LINE, index))
        for index, mark in enumerate(doc.get("marks", [])):
            events.append((float(mark.get("span_start", mark.get("start", 0.0))), 1, KIND_MARK, index))
        for index, scene in enumerate(doc.get("scenes", [])):
            events.append((float(scene.get("start", 0.0)), 0, KIND_SCENE, index))
        events.sort(key=lambda e: (e[0], e[1]))

        line_format = QTextBlockFormat()
        line_format.setTopMargin(3.0)
        line_format.setBottomMargin(3.0)
        line_format.setLeftMargin(0.0)
        mark_block = QTextBlockFormat()
        mark_block.setTopMargin(6.0)
        mark_block.setBottomMargin(6.0)
        mark_block.setLeftMargin(0.0)
        mark_block.setBackground(with_alpha(self._theme.warning, 22))
        scene_block = QTextBlockFormat()
        scene_block.setTopMargin(6.0)
        scene_block.setBottomMargin(6.0)
        scene_block.setLeftMargin(0.0)
        scene_format = QTextCharFormat()
        scene_font = QFont(self.font())
        scene_font.setPointSizeF(9.5)
        scene_font.setItalic(True)
        scene_format.setFont(scene_font)
        scene_format.setForeground(self._theme.muted)

        text_document = QTextDocument(self)
        text_document.setDocumentMargin(18.0)
        cursor = QTextCursor(text_document)
        first = True
        for start, _, kind, index in events:
            if kind == KIND_LINE:
                if first:
                    cursor.setBlockFormat(line_format)
                else:
                    cursor.insertBlock(line_format)
                line = doc["lines"][index]
                label = line.get("speaker")
                name = names.get(label, label) if label is not None else UNLABELLED_NAME
                colour = colours.get(label, self._theme.muted) if label is not None else self._theme.muted
                block_number = cursor.blockNumber()
                cursor.insertText(f"{clock(start):>{GUTTER_CHARS}}  ", time_format)
                self._gutter_end[block_number] = cursor.positionInBlock()
                coloured = QTextCharFormat(name_format)
                coloured.setForeground(QColor(colour))
                cursor.insertText(name, coloured)
                if line.get("src") == SOURCE_LISTENER:
                    cursor.insertText(f" ({LISTENER_SUFFIX})", scene_format)
                    cursor.insertText("  ", text_format)
                    cursor.insertText(str(line.get("text", "")), resolved_format)
                else:
                    cursor.insertText("  ", text_format)
                    cursor.insertText(str(line.get("text", "")), text_format)
                self._block_info[block_number] = (KIND_LINE, index)
                self._line_starts.append(start)
                self._line_blocks.append(block_number)
            elif kind == KIND_SCENE:
                if first:
                    cursor.setBlockFormat(scene_block)
                else:
                    cursor.insertBlock(scene_block)
                scene = doc["scenes"][index]
                block_number = cursor.blockNumber()
                cursor.insertText(f"{clock(start):>{GUTTER_CHARS}}  ", time_format)
                self._gutter_end[block_number] = cursor.positionInBlock()
                cursor.insertText(
                    f"{scene_phrase(scene)}, to {clock(float(scene.get('end', start)))}. Click to listen.",
                    scene_format,
                )
                self._block_info[block_number] = (KIND_SCENE, index)
            else:
                if first:
                    cursor.setBlockFormat(mark_block)
                else:
                    cursor.insertBlock(mark_block)
                mark = doc["marks"][index]
                block_number = cursor.blockNumber()
                words = int(mark.get("detector_words", 0))
                noun = "word" if words == 1 else "words"
                span_start = float(mark.get("span_start", mark.get("start", 0.0)))
                span_end = float(mark.get("span_end", mark.get("end", 0.0)))
                cursor.insertText(f"{'':>{GUTTER_CHARS}}  ", time_format)
                self._gutter_end[block_number] = 0
                cursor.insertText(
                    f"Possible missed speech, {clock(span_start)} to {clock(span_end)}: the second engine heard "
                    f"{words} {noun} here. Click to listen.",
                    mark_format,
                )
                resolution = self._resolution_for(index)
                if resolution is not None:
                    status = str(resolution.get("status", "open"))
                    if status == "nothing":
                        cursor.insertText("   Listener: nothing was said.", resolved_format)
                    elif status == "text":
                        cursor.insertText(f"   Listener heard: {resolution.get('note', '')}", resolved_format)
                self._block_info[block_number] = (KIND_MARK, index)
            first = False
        self.setDocument(text_document)
        self.verticalScrollBar().setValue(0)

    def _resolution_for(self, mark_index: int) -> dict[str, Any] | None:
        if self._session is None or not (0 <= mark_index < len(self._session)):
            return None
        entry = self._session[mark_index]
        return entry if entry.get("status", "open") != "open" else None

    # ----- tracking ------------------------------------------------------------------------

    def line_at(self, seconds: float) -> int | None:
        """Index of the last line that starts at or before `seconds`, or None before the first."""
        if not self._line_starts:
            return None
        position = bisect_right(self._line_starts, seconds) - 1
        return position if position >= 0 else None

    def set_position(self, seconds: float) -> None:
        """Highlight the line under the playhead and keep it in view when following."""
        line = self.line_at(seconds)
        if line == self._current_line:
            return
        self._current_line = line
        if line is None:
            self.setExtraSelections([])
            return
        block = self.document().findBlockByNumber(self._line_blocks[line])
        selection = QTextEdit.ExtraSelection()
        selection.cursor = QTextCursor(block)
        selection.format.setBackground(with_alpha(self._theme.accent, 34 if not self._theme.dark else 60))
        selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
        self.setExtraSelections([selection])
        if self._follow and time.monotonic() >= self._hold_until:
            self._scroll_to_block(block.blockNumber())

    def _scroll_to_block(self, block_number: int) -> None:
        block = self.document().findBlockByNumber(block_number)
        if not block.isValid():
            return
        rect = self.document().documentLayout().blockBoundingRect(block)
        bar = self.verticalScrollBar()
        viewport_height = self.viewport().height()
        target = int(rect.top() - viewport_height * 0.38)
        bar.setValue(max(bar.minimum(), min(bar.maximum(), target)))

    # ----- mouse -------------------------------------------------------------------------

    def _info_at(self, event: QMouseEvent) -> tuple[str, int, int] | None:
        cursor = self.cursorForPosition(event.position().toPoint())
        block = cursor.block()
        info = self._block_info.get(block.blockNumber())
        if info is None:
            return None
        return info[0], info[1], cursor.positionInBlock()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt virtual)
        if event.button() == Qt.MouseButton.LeftButton:
            found = self._info_at(event)
            if found is not None:
                kind, index, position = found
                if kind == KIND_MARK:
                    self.mark_requested.emit(index)
                    event.accept()
                    return
                if kind == KIND_SCENE:
                    scene = (self._doc or {}).get("scenes", [])[index]
                    self.seek_requested.emit(float(scene.get("start", 0.0)))
                    event.accept()
                    return
                block_number = self.cursorForPosition(event.position().toPoint()).block().blockNumber()
                if position <= self._gutter_end.get(block_number, 0):
                    self.seek_requested.emit(self._line_starts[index])
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt virtual)
        found = self._info_at(event)
        if found is not None and found[0] == KIND_LINE:
            self.seek_requested.emit(self._line_starts[found[1]])
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 (Qt virtual)
        self._hold_until = time.monotonic() + FOLLOW_HOLD_S
        super().wheelEvent(event)

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            # Keys belong to the window (play, pause, nudge); the pane only scrolls with the mouse.
            return False
        return super().event(event)
