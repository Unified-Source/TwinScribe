"""The application icon, painted at call time like every other icon so that no image file
ships: a rounded square in the accent colour carrying two traces, the published engine's solid
and the checking engine's fainter beneath it, which is the shape of the two-engine design.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap

from twinscribe.app.theme import theme_for

ICON_SIZES: tuple[int, ...] = (16, 24, 32, 48, 64, 128, 256)
FOREGROUND = QColor("#ffffff")


def _trace(size: float, amplitude: float, phase: float, cycles: float = 2.5, points: int = 64) -> QPainterPath:
    """A damped sine across the icon, from 16 to 84 per cent of the width."""
    x0, x1 = 0.16 * size, 0.84 * size
    path = QPainterPath()
    for i in range(points + 1):
        t = i / points
        x = x0 + (x1 - x0) * t
        envelope = math.sin(math.pi * t) ** 0.6
        y = 0.5 * size + amplitude * size * envelope * math.sin(2.0 * math.pi * cycles * t + phase)
        if i == 0:
            path.moveTo(QPointF(x, y))
        else:
            path.lineTo(QPointF(x, y))
    return path


def paint_app_icon(painter: QPainter, size: float, background: QColor, foreground: QColor = FOREGROUND) -> None:
    """Paint the icon at `size` pixels with its origin at the painter's origin."""
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(background)
    painter.drawRoundedRect(QRectF(0.0, 0.0, size, size), 0.22 * size, 0.22 * size)

    faint = QColor(foreground)
    faint.setAlpha(140)
    thin = QPen(faint)
    thin.setWidthF(max(1.0, 0.055 * size))
    thin.setCapStyle(Qt.PenCapStyle.RoundCap)
    thin.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(thin)
    painter.drawPath(_trace(size, 0.13, math.pi / 2.2))

    thick = QPen(foreground)
    thick.setWidthF(max(1.5, 0.085 * size))
    thick.setCapStyle(Qt.PenCapStyle.RoundCap)
    thick.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(thick)
    painter.drawPath(_trace(size, 0.2, 0.0))


def app_image(size: int, dark: bool = False) -> QImage:
    """The icon as an image with transparent corners."""
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        paint_app_icon(painter, float(size), theme_for(dark).accent)
    finally:
        painter.end()
    return image


def app_pixmap(size: int, dark: bool = False) -> QPixmap:
    return QPixmap.fromImage(app_image(size, dark))


def app_icon(dark: bool = False) -> QIcon:
    """The icon at every standard size, for windows and the task bar."""
    icon = QIcon()
    for size in ICON_SIZES:
        icon.addPixmap(app_pixmap(size, dark))
    return icon


def pack_ico(images: Sequence[tuple[int, bytes]]) -> bytes:
    """A Windows icon file holding PNG-encoded images, one entry per (size, png bytes).

    The header names the entry count; each 16-byte directory entry gives the dimensions (0
    standing for 256), one colour plane, 32 bits per pixel, the entry's length and its offset
    from the start of the file; the image data follows in order.
    """
    if not images:
        raise ValueError("an icon file needs at least one image")
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    entries: list[bytes] = []
    payload = b""
    for size, data in images:
        if not (1 <= size <= 256):
            raise ValueError(f"icon sizes run from 1 to 256 pixels, got {size}")
        dimension = 0 if size == 256 else size
        entries.append(struct.pack("<BBBBHHII", dimension, dimension, 0, 0, 1, 32, len(data), offset))
        payload += data
        offset += len(data)
    return header + b"".join(entries) + payload


def ico_entries(data: bytes) -> list[tuple[int, int, int]]:
    """(size, length, offset) per entry of an icon file, for checking what was written."""
    if len(data) < 6:
        raise ValueError("not an icon file")
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    if reserved != 0 or kind != 1:
        raise ValueError("not an icon file")
    out = []
    for index in range(count):
        width, _height, _colours, _reserved, _planes, _bits, length, offset = struct.unpack_from("<BBBBHHII", data, 6 + 16 * index)
        out.append((256 if width == 0 else width, length, offset))
    return out
