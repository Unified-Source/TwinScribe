"""Icons painted with QPainter at call time, so that the screens depend on no icon font and
no image file. Every icon is a simple filled shape in one colour on a transparent square.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

KINDS: tuple[str, ...] = (
    "play",
    "pause",
    "stop",
    "back",
    "forward",
    "previous",
    "next",
    "folder",
    "file",
    "gear",
    "check",
    "warning",
    "flag",
    "plus",
    "close",
    "trash",
    "follow",
)


def _triangle(p: QPainter, x0: float, x1: float, y0: float, y1: float, right: bool = True) -> None:
    if right:
        points = [QPointF(x0, y0), QPointF(x1, (y0 + y1) / 2.0), QPointF(x0, y1)]
    else:
        points = [QPointF(x1, y0), QPointF(x0, (y0 + y1) / 2.0), QPointF(x1, y1)]
    p.drawPolygon(QPolygonF(points))


def paint_icon(p: QPainter, kind: str, colour: QColor, s: float) -> None:
    """Paint one icon of size `s` at the origin of `p`."""
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(colour)
    pen = QPen(colour)
    pen.setWidthF(max(1.5, s * 0.11))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

    if kind == "play":
        _triangle(p, 0.28 * s, 0.84 * s, 0.16 * s, 0.84 * s)
    elif kind == "pause":
        p.drawRoundedRect(QRectF(0.22 * s, 0.18 * s, 0.2 * s, 0.64 * s), 1.5, 1.5)
        p.drawRoundedRect(QRectF(0.58 * s, 0.18 * s, 0.2 * s, 0.64 * s), 1.5, 1.5)
    elif kind == "stop":
        p.drawRoundedRect(QRectF(0.22 * s, 0.22 * s, 0.56 * s, 0.56 * s), 2.0, 2.0)
    elif kind == "back":
        _triangle(p, 0.1 * s, 0.5 * s, 0.22 * s, 0.78 * s, right=False)
        _triangle(p, 0.5 * s, 0.9 * s, 0.22 * s, 0.78 * s, right=False)
    elif kind == "forward":
        _triangle(p, 0.1 * s, 0.5 * s, 0.22 * s, 0.78 * s)
        _triangle(p, 0.5 * s, 0.9 * s, 0.22 * s, 0.78 * s)
    elif kind == "previous":
        p.drawRect(QRectF(0.16 * s, 0.2 * s, 0.12 * s, 0.6 * s))
        _triangle(p, 0.32 * s, 0.86 * s, 0.2 * s, 0.8 * s, right=False)
    elif kind == "next":
        _triangle(p, 0.14 * s, 0.68 * s, 0.2 * s, 0.8 * s)
        p.drawRect(QRectF(0.72 * s, 0.2 * s, 0.12 * s, 0.6 * s))
    elif kind == "folder":
        path = QPainterPath()
        path.addRoundedRect(QRectF(0.1 * s, 0.3 * s, 0.8 * s, 0.5 * s), 2.0, 2.0)
        path.addRoundedRect(QRectF(0.1 * s, 0.2 * s, 0.36 * s, 0.2 * s), 2.0, 2.0)
        p.drawPath(path.simplified())
    elif kind == "file":
        path = QPainterPath()
        path.moveTo(0.24 * s, 0.12 * s)
        path.lineTo(0.62 * s, 0.12 * s)
        path.lineTo(0.78 * s, 0.28 * s)
        path.lineTo(0.78 * s, 0.88 * s)
        path.lineTo(0.24 * s, 0.88 * s)
        path.closeSubpath()
        p.drawPath(path)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        p.drawRect(QRectF(0.36 * s, 0.5 * s, 0.3 * s, 0.06 * s))
        p.drawRect(QRectF(0.36 * s, 0.64 * s, 0.3 * s, 0.06 * s))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    elif kind == "gear":
        centre = QPointF(0.5 * s, 0.5 * s)
        for tooth in range(8):
            p.save()
            p.translate(centre)
            p.rotate(tooth * 45.0)
            p.drawRoundedRect(QRectF(-0.09 * s, -0.46 * s, 0.18 * s, 0.24 * s), 1.5, 1.5)
            p.restore()
        p.drawEllipse(centre, 0.3 * s, 0.3 * s)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        p.drawEllipse(centre, 0.13 * s, 0.13 * s)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    elif kind == "check":
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        path.moveTo(0.2 * s, 0.52 * s)
        path.lineTo(0.42 * s, 0.74 * s)
        path.lineTo(0.82 * s, 0.3 * s)
        p.drawPath(path)
    elif kind == "warning":
        p.drawPolygon(QPolygonF([QPointF(0.5 * s, 0.12 * s), QPointF(0.92 * s, 0.86 * s), QPointF(0.08 * s, 0.86 * s)]))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        p.drawRoundedRect(QRectF(0.45 * s, 0.36 * s, 0.1 * s, 0.28 * s), 1.0, 1.0)
        p.drawEllipse(QPointF(0.5 * s, 0.74 * s), 0.06 * s, 0.06 * s)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    elif kind == "flag":
        p.drawRect(QRectF(0.2 * s, 0.12 * s, 0.08 * s, 0.78 * s))
        p.drawPolygon(
            QPolygonF([QPointF(0.28 * s, 0.14 * s), QPointF(0.84 * s, 0.32 * s), QPointF(0.28 * s, 0.52 * s)])
        )
    elif kind == "plus":
        p.drawRoundedRect(QRectF(0.44 * s, 0.16 * s, 0.12 * s, 0.68 * s), 1.5, 1.5)
        p.drawRoundedRect(QRectF(0.16 * s, 0.44 * s, 0.68 * s, 0.12 * s), 1.5, 1.5)
    elif kind == "close":
        p.setPen(pen)
        p.drawLine(QPointF(0.24 * s, 0.24 * s), QPointF(0.76 * s, 0.76 * s))
        p.drawLine(QPointF(0.76 * s, 0.24 * s), QPointF(0.24 * s, 0.76 * s))
    elif kind == "trash":
        p.drawRoundedRect(QRectF(0.24 * s, 0.28 * s, 0.52 * s, 0.6 * s), 2.0, 2.0)
        p.drawRect(QRectF(0.16 * s, 0.18 * s, 0.68 * s, 0.08 * s))
        p.drawRect(QRectF(0.4 * s, 0.1 * s, 0.2 * s, 0.08 * s))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        p.drawRect(QRectF(0.38 * s, 0.4 * s, 0.06 * s, 0.36 * s))
        p.drawRect(QRectF(0.56 * s, 0.4 * s, 0.06 * s, 0.36 * s))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    elif kind == "follow":
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(0.5 * s, 0.5 * s), 0.3 * s, 0.3 * s)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(colour)
        p.drawEllipse(QPointF(0.5 * s, 0.5 * s), 0.11 * s, 0.11 * s)
    else:
        raise KeyError(f"unknown icon {kind!r}; known: {', '.join(KINDS)}")


def make_icon(kind: str, colour: QColor, size: int = 18, ratio: float = 2.0) -> QIcon:
    """A QIcon of one painted shape, rendered at `ratio` times the logical size for sharpness."""
    pixels = int(round(size * ratio))
    pixmap = QPixmap(pixels, pixels)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        paint_icon(painter, kind, colour, float(size))
    finally:
        painter.end()
    return QIcon(pixmap)
