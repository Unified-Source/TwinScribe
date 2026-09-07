"""The library: the recordings the window knows about, with the state of each.

A recording is new, queued, running (with progress), done (a transcript document sits beside
it) or failed. The model holds the items, the delegate paints one row per recording with its
name, its folder or its state, and a slim progress bar while it runs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QListView, QStyle, QStyledItemDelegate, QStyleOptionViewItem, QWidget

from twinscribe.app.icons import paint_icon
from twinscribe.app.theme import Theme, theme_for, with_alpha
from twinscribe.outputs.transcript_doc import TRANSCRIPT_SCHEMA, clock, speaker_count
from twinscribe.pipeline import discover_media, is_video, output_paths

STATUS_NEW = "new"
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

ROW_HEIGHT_PX = 54
ITEM_ROLE = int(Qt.ItemDataRole.UserRole) + 1


@dataclass
class MediaItem:
    """One recording and what is known about it."""

    path: Path
    status: str = STATUS_NEW
    progress: float = 0.0
    message: str = ""
    duration_s: float | None = None
    speakers: int | None = None
    marks: int | None = None
    transcript_path: Path | None = None
    error: str = ""

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def video(self) -> bool:
        return is_video(self.path)

    @property
    def done(self) -> bool:
        return self.status == STATUS_DONE

    def summary(self) -> str:
        """Second-line text: the state, or the folder when there is nothing to say."""
        if self.status == STATUS_RUNNING:
            return self.message or "Working"
        if self.status == STATUS_FAILED:
            return self.error or "Failed"
        if self.status == STATUS_QUEUED:
            return "Queued"
        if self.status == STATUS_DONE:
            parts: list[str] = []
            if self.speakers is not None:
                parts.append(f"{self.speakers} speaker" + ("" if self.speakers == 1 else "s"))
            if self.marks is not None:
                parts.append(f"{self.marks} to review" if self.marks else "nothing to review")
            return "  |  ".join(parts) if parts else "Transcribed"
        return self.path.parent.name or str(self.path.parent)


def read_document_facts(path: Path) -> dict[str, Any] | None:
    """Duration, speaker count and mark count from a transcript document beside a recording.

    Returns None when the file is absent, unreadable, of another schema or about another
    recording, so a stale document can never attach itself to the wrong file.
    """
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("schema") != TRANSCRIPT_SCHEMA:
        return None
    return {
        "name": str(doc.get("source", {}).get("name", "")),
        "duration_s": float(doc.get("duration_s", 0.0)),
        "speakers": speaker_count(doc),
        "marks": int(doc.get("review", {}).get("marks", 0)),
    }


class LibraryModel(QAbstractListModel):
    """Recordings in the order they were added."""

    def __init__(self, parent: QObject | None = None, out_dir: Path | None = None) -> None:
        super().__init__(parent)
        self._items: list[MediaItem] = []
        self._out_dir = out_dir

    def set_output_dir(self, out_dir: Path | None) -> None:
        """Where outputs are looked for (None means beside each recording)."""
        self._out_dir = out_dir

    # ----- Qt model ----------------------------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 (Qt virtual)
        return 0 if parent.isValid() else len(self._items)

    def data(self, index: QModelIndex, role: int = int(Qt.ItemDataRole.DisplayRole)) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._items)):
            return None
        item = self._items[index.row()]
        if role == int(Qt.ItemDataRole.DisplayRole):
            return item.name
        if role == int(Qt.ItemDataRole.ToolTipRole):
            return str(item.path)
        if role == ITEM_ROLE:
            return item
        return None

    # ----- items -------------------------------------------------------------------------

    def items(self) -> list[MediaItem]:
        return list(self._items)

    def item(self, row: int) -> MediaItem:
        return self._items[row]

    def row_for_path(self, path: Path) -> int | None:
        wanted = str(Path(path).resolve()).lower()
        for row, item in enumerate(self._items):
            if str(item.path.resolve()).lower() == wanted:
                return row
        return None

    def refresh_item(self, row: int) -> None:
        """Look again for a transcript document beside the recording and update the state."""
        item = self._items[row]
        transcript = output_paths(item.path, self._out_dir).transcript
        facts = read_document_facts(transcript)
        if facts is not None and facts["name"] == item.path.name:
            item.status = STATUS_DONE
            item.transcript_path = transcript
            item.duration_s = facts["duration_s"]
            item.speakers = facts["speakers"]
            item.marks = facts["marks"]
            item.error = ""
        elif item.status == STATUS_DONE:
            item.status = STATUS_NEW
            item.transcript_path = None
            item.speakers = None
            item.marks = None
        self._changed(row)

    def add_paths(self, paths: Iterable[Path], recursive: bool = True) -> list[int]:
        """Add the recordings found under `paths`; returns the rows added (duplicates skipped)."""
        found = discover_media(paths, recursive=recursive)
        added: list[int] = []
        for path in found:
            if self.row_for_path(path) is not None:
                continue
            row = len(self._items)
            self.beginInsertRows(QModelIndex(), row, row)
            self._items.append(MediaItem(path=path))
            self.endInsertRows()
            self.refresh_item(row)
            added.append(row)
        return added

    def remove_rows(self, rows: Sequence[int]) -> None:
        for row in sorted(set(rows), reverse=True):
            if 0 <= row < len(self._items):
                self.beginRemoveRows(QModelIndex(), row, row)
                del self._items[row]
                self.endRemoveRows()

    def clear(self) -> None:
        self.beginResetModel()
        self._items.clear()
        self.endResetModel()

    def pending_rows(self) -> list[int]:
        """Rows that are not done and not running."""
        return [row for row, item in enumerate(self._items) if item.status in (STATUS_NEW, STATUS_QUEUED, STATUS_FAILED)]

    def set_queued(self, rows: Iterable[int]) -> None:
        for row in rows:
            item = self._items[row]
            item.status = STATUS_QUEUED
            item.progress = 0.0
            item.message = ""
            item.error = ""
            self._changed(row)

    def set_progress(self, row: int, fraction: float, message: str) -> None:
        if not (0 <= row < len(self._items)):
            return
        item = self._items[row]
        item.status = STATUS_RUNNING
        item.progress = min(1.0, max(0.0, fraction))
        item.message = message
        self._changed(row)

    def set_done(self, row: int, document: dict[str, Any], transcript_path: Path) -> None:
        if not (0 <= row < len(self._items)):
            return
        item = self._items[row]
        item.status = STATUS_DONE
        item.progress = 1.0
        item.message = ""
        item.error = ""
        item.transcript_path = transcript_path
        item.duration_s = float(document.get("duration_s", 0.0))
        item.speakers = speaker_count(document)
        item.marks = int(document.get("review", {}).get("marks", 0))
        self._changed(row)

    def set_failed(self, row: int, error: str) -> None:
        if not (0 <= row < len(self._items)):
            return
        item = self._items[row]
        item.status = STATUS_FAILED
        item.progress = 0.0
        item.error = error
        self._changed(row)

    def set_duration(self, row: int, duration_s: float) -> None:
        if 0 <= row < len(self._items) and duration_s > 0.0:
            self._items[row].duration_s = float(duration_s)
            self._changed(row)

    def reset_pending(self) -> None:
        """Queued rows go back to new when a batch stops before reaching them."""
        for row, item in enumerate(self._items):
            if item.status in (STATUS_QUEUED, STATUS_RUNNING):
                item.status = STATUS_NEW
                item.progress = 0.0
                item.message = ""
                self._changed(row)

    def _changed(self, row: int) -> None:
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [])


class LibraryDelegate(QStyledItemDelegate):
    """Paints one recording per row: state glyph, name, folder or state, duration or progress."""

    def __init__(self, parent: QObject | None = None, theme: Theme | None = None) -> None:
        super().__init__(parent)
        self._theme = theme if theme is not None else theme_for(False)

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # noqa: N802
        return QSize(option.rect.width(), ROW_HEIGHT_PX)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        item = index.data(ITEM_ROLE)
        if not isinstance(item, MediaItem):
            super().paint(painter, option, index)
            return
        theme = self._theme
        palette = option.palette
        rect = QRectF(option.rect).adjusted(6.0, 2.0, -6.0, -2.0)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if selected:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(with_alpha(theme.accent, 46 if not theme.dark else 70))
            painter.drawRoundedRect(rect, 7.0, 7.0)
        elif hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(palette.color(palette.ColorRole.Midlight))
            painter.drawRoundedRect(rect, 7.0, 7.0)

        text_colour = palette.color(palette.ColorRole.Text)
        muted = theme.muted
        glyph_size = 14.0
        glyph_x = rect.left() + 10.0
        glyph_y = rect.center().y() - glyph_size / 2.0

        # State glyph.
        painter.save()
        painter.translate(glyph_x, glyph_y)
        if item.status == STATUS_DONE:
            paint_icon(painter, "check", theme.success, glyph_size)
        elif item.status == STATUS_FAILED:
            paint_icon(painter, "warning", theme.warning, glyph_size)
        elif item.status == STATUS_RUNNING:
            pen = QPen(theme.accent)
            pen.setWidthF(2.0)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(2.0, 2.0, glyph_size - 4.0, glyph_size - 4.0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(theme.accent)
            span = int(-360.0 * 16 * item.progress)
            painter.drawPie(QRectF(4.0, 4.0, glyph_size - 8.0, glyph_size - 8.0), 90 * 16, span)
        else:
            pen = QPen(theme.accent if item.status == STATUS_QUEUED else muted)
            pen.setWidthF(1.6)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(2.5, 2.5, glyph_size - 5.0, glyph_size - 5.0))
        painter.restore()

        # Right column: duration, or percentage while running.
        base_font = QFont(option.font)
        small_font = QFont(base_font)
        small_font.setPointSizeF(max(7.5, base_font.pointSizeF() - 1.5))
        right_text = ""
        if item.status == STATUS_RUNNING:
            right_text = f"{int(round(100.0 * item.progress))}%"
        elif item.duration_s:
            right_text = clock(item.duration_s, tenths=False)
        right_metrics = QFontMetrics(small_font)
        right_width = right_metrics.horizontalAdvance(right_text) if right_text else 0

        text_left = glyph_x + glyph_size + 12.0
        text_right = rect.right() - 12.0 - (right_width + 10.0 if right_text else 0.0)
        name_rect = QRectF(text_left, rect.top() + 7.0, max(10.0, text_right - text_left), 20.0)
        summary_rect = QRectF(text_left, rect.top() + 28.0, max(10.0, rect.right() - 12.0 - text_left), 18.0)

        painter.setFont(base_font)
        painter.setPen(text_colour)
        name = QFontMetrics(base_font).elidedText(item.name, Qt.TextElideMode.ElideMiddle, int(name_rect.width()))
        painter.drawText(name_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), name)

        if right_text:
            painter.setFont(small_font)
            painter.setPen(muted)
            right_rect = QRectF(rect.right() - 12.0 - right_width, rect.top() + 8.0, right_width, 18.0)
            painter.drawText(right_rect, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), right_text)

        painter.setFont(small_font)
        if item.status == STATUS_RUNNING:
            bar = QRectF(text_left, rect.top() + 36.0, rect.right() - 12.0 - text_left, 3.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(palette.color(palette.ColorRole.Mid))
            painter.drawRoundedRect(bar, 1.5, 1.5)
            painter.setBrush(theme.accent)
            painter.drawRoundedRect(QRectF(bar.left(), bar.top(), bar.width() * item.progress, bar.height()), 1.5, 1.5)
            painter.setPen(muted)
            painter.drawText(QRectF(text_left, rect.top() + 24.0, bar.width(), 12.0),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                             right_metrics.elidedText(item.summary(), Qt.TextElideMode.ElideRight, int(bar.width())))
        else:
            colour: QColor = muted
            if item.status == STATUS_FAILED:
                colour = theme.warning
            painter.setPen(colour)
            summary = right_metrics.elidedText(item.summary(), Qt.TextElideMode.ElideRight, int(summary_rect.width()))
            painter.drawText(summary_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), summary)
        painter.restore()


class LibraryView(QListView):
    """The list of recordings."""

    def __init__(self, parent: QWidget | None = None, theme: Theme | None = None) -> None:
        super().__init__(parent)
        self._delegate = LibraryDelegate(self, theme)
        self.setItemDelegate(self._delegate)
        self.setUniformItemSizes(True)
        self.setMouseTracking(True)
        self.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSpacing(1)
        self.setFrameShape(QListView.Shape.NoFrame)

    def set_theme(self, theme: Theme) -> None:
        self._delegate.set_theme(theme)
        self.viewport().update()

    def selected_rows(self) -> list[int]:
        return sorted({index.row() for index in self.selectionModel().selectedRows()})
