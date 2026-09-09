"""The application marks, painted at call time like every other icon so that no image file
ships, and the wordmark.

Three marks are drawn. "panels": two facing panels each carrying a waveform, the published
engine's dark and the checking engine's light, which is the shape of the two-engine design.
"monogram": the initials with a waveform beside them. "document": a page with text lines and
a waveform running through it, grey where it enters and blue where it leaves. The windows, the
executables and the documentation use the mark named by DEFAULT_CONCEPT; the other two stay
available to `tools/make_icon.py --concept`, so that the choice is made on rendered files. The
wordmark sets "Twin" in the text colour and "Scribe" in the accent.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QIcon,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)

from twinscribe.app.theme import Theme, theme_for

ICON_SIZES: tuple[int, ...] = (16, 24, 32, 48, 64, 128, 256)

CONCEPT_PANELS = "panels"
CONCEPT_MONOGRAM = "monogram"
CONCEPT_DOCUMENT = "document"
DEFAULT_CONCEPT = CONCEPT_PANELS

TILE_DARK = QColor("#1b2237")
TILE_LIGHT = QColor("#f2f4f9")
NAVY = QColor("#1e2f6b")
NAVY_ON_DARK = QColor("#2d47a6")
BLUE = QColor("#2b63c9")
BLUE_SOFT = QColor("#6f8fe8")
GREY = QColor("#8d96a8")
PAGE_FOLD = QColor("#d5dae6")
WHITE = QColor("#ffffff")
TEXT_LIGHT = QColor("#e8e8ea")

# Every coordinate below is a fraction of the icon's size.
TILE_RADIUS = 0.22
PANEL_RADIUS = 0.045
# The panels as the polygons the pen is drawn round; the visible edge lies PANEL_RADIUS beyond.
LEFT_PANEL: tuple[tuple[float, float], ...] = ((0.175, 0.325), (0.425, 0.245), (0.425, 0.755), (0.175, 0.675))
RIGHT_PANEL: tuple[tuple[float, float], ...] = ((0.575, 0.275), (0.825, 0.345), (0.825, 0.655), (0.575, 0.725))
LEFT_BARS: tuple[float, ...] = (0.10, 0.20, 0.32, 0.20, 0.10)
RIGHT_BARS: tuple[float, ...] = (0.08, 0.16, 0.26, 0.16, 0.08)
MONOGRAM_BARS: tuple[float, ...] = (0.10, 0.22, 0.34, 0.16)
DOCUMENT_BARS: tuple[float, ...] = (0.06, 0.10, 0.16, 0.10, 0.22, 0.14, 0.30, 0.16, 0.24, 0.12, 0.18, 0.10, 0.06)
DOCUMENT_BLUE_FROM = 6
BAR_WIDTH = 0.035
SMALL = 24  # below this size every other bar is drawn, at twice the spacing

PainterFn = Callable[[QPainter, float, bool], None]


def _tile(painter: QPainter, size: float, colour: QColor) -> None:
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(colour)
    painter.drawRoundedRect(QRectF(0.0, 0.0, size, size), TILE_RADIUS * size, TILE_RADIUS * size)


def _rounded_polygon(painter: QPainter, points: Sequence[tuple[float, float]], size: float, radius: float, colour: QColor) -> None:
    """A polygon filled and stroked in one colour with a round-joined pen twice the radius
    wide, which gives it rounded corners a radius beyond its points."""
    polygon = QPolygonF([QPointF(x * size, y * size) for x, y in points])
    pen = QPen(colour)
    pen.setWidthF(2.0 * radius * size)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(colour)
    painter.drawPolygon(polygon)


def _bars(
    painter: QPainter,
    size: float,
    centre_x: float,
    centre_y: float,
    heights: Sequence[float],
    spacing: float,
    colours: Sequence[QColor] | QColor,
) -> None:
    """Vertical round-capped bars centred on (centre_x, centre_y), `spacing` apart."""
    if size < SMALL:
        heights = tuple(heights[::2])
        colours = tuple(colours[::2]) if not isinstance(colours, QColor) else colours
        spacing *= 2.0
    count = len(heights)
    for index, height in enumerate(heights):
        colour = colours if isinstance(colours, QColor) else colours[index]
        pen = QPen(colour)
        pen.setWidthF(max(1.0, BAR_WIDTH * size))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        x = (centre_x + (index - (count - 1) / 2.0) * spacing) * size
        half = max(0.5, 0.5 * height * size)
        painter.drawLine(QPointF(x, centre_y * size - half), QPointF(x, centre_y * size + half))


def paint_panels(painter: QPainter, size: float, dark: bool) -> None:
    """Two facing panels, the published engine's dark and the checking engine's light, each
    with a waveform; on a light tile, or a dark one for the dark palette."""
    _tile(painter, size, TILE_DARK if dark else TILE_LIGHT)
    _rounded_polygon(painter, LEFT_PANEL, size, PANEL_RADIUS, NAVY_ON_DARK if dark else NAVY)
    _rounded_polygon(painter, RIGHT_PANEL, size, PANEL_RADIUS, BLUE_SOFT)
    _bars(painter, size, 0.30, 0.50, LEFT_BARS, 0.065, WHITE)
    _bars(painter, size, 0.70, 0.50, RIGHT_BARS, 0.065, WHITE)


def paint_monogram(painter: QPainter, size: float, dark: bool) -> None:
    """The initials, a white T and a blue S, with a waveform beside them on a dark tile."""
    _tile(painter, size, TILE_DARK)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(WHITE)
    painter.drawRect(QRectF(0.14 * size, 0.24 * size, 0.32 * size, 0.10 * size))
    painter.drawRect(QRectF(0.27 * size, 0.24 * size, 0.10 * size, 0.46 * size))
    gradient = QLinearGradient(QPointF(0.5 * size, 0.24 * size), QPointF(0.5 * size, 0.80 * size))
    gradient.setColorAt(0.0, BLUE_SOFT)
    gradient.setColorAt(1.0, BLUE)
    pen = QPen(QBrush(gradient), max(1.5, 0.09 * size))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath(QPointF(0.68 * size, 0.30 * size))
    path.cubicTo(QPointF(0.44 * size, 0.24 * size), QPointF(0.44 * size, 0.52 * size), QPointF(0.56 * size, 0.52 * size))
    path.cubicTo(QPointF(0.68 * size, 0.52 * size), QPointF(0.68 * size, 0.80 * size), QPointF(0.44 * size, 0.74 * size))
    painter.drawPath(path)
    _bars(painter, size, 0.83, 0.50, MONOGRAM_BARS, 0.07, BLUE_SOFT)


def paint_document(painter: QPainter, size: float, dark: bool) -> None:
    """A page with text lines and a waveform through it, grey on the way in and blue on the
    way out, on a dark tile."""
    _tile(painter, size, TILE_DARK)
    page = QPolygonF([
        QPointF(0.24 * size, 0.15 * size), QPointF(0.66 * size, 0.15 * size), QPointF(0.76 * size, 0.25 * size),
        QPointF(0.76 * size, 0.85 * size), QPointF(0.24 * size, 0.85 * size),
    ])
    edge = QPen(WHITE)
    edge.setWidthF(max(1.0, 0.03 * size))
    edge.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(edge)
    painter.setBrush(WHITE)
    painter.drawPolygon(page)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(PAGE_FOLD)
    painter.drawPolygon(QPolygonF([QPointF(0.66 * size, 0.15 * size), QPointF(0.66 * size, 0.25 * size), QPointF(0.76 * size, 0.25 * size)]))
    line = QPen(GREY)
    line.setWidthF(max(1.0, BAR_WIDTH * size))
    line.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(line)
    for y, x1 in ((0.27, 0.58), (0.33, 0.62), (0.39, 0.52), (0.69, 0.60), (0.75, 0.48)):
        if size < SMALL and y in (0.33, 0.75):
            continue
        painter.drawLine(QPointF(0.31 * size, y * size), QPointF(x1 * size, y * size))
    colours = tuple(GREY if index < DOCUMENT_BLUE_FROM else BLUE for index in range(len(DOCUMENT_BARS)))
    _bars(painter, size, 0.50, 0.53, DOCUMENT_BARS, 0.06, colours)


CONCEPTS: dict[str, PainterFn] = {
    CONCEPT_PANELS: paint_panels,
    CONCEPT_MONOGRAM: paint_monogram,
    CONCEPT_DOCUMENT: paint_document,
}


def paint_app_icon(painter: QPainter, size: float, dark: bool = False, concept: str | None = None) -> None:
    """Paint the mark at `size` pixels with its origin at the painter's origin."""
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    CONCEPTS[concept or DEFAULT_CONCEPT](painter, size, dark)


def app_image(size: int, dark: bool = False, concept: str | None = None) -> QImage:
    """The mark as an image with transparent corners."""
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        paint_app_icon(painter, float(size), dark, concept)
    finally:
        painter.end()
    return image


def app_pixmap(size: int, dark: bool = False, concept: str | None = None) -> QPixmap:
    return QPixmap.fromImage(app_image(size, dark, concept))


def app_icon(dark: bool = False, concept: str | None = None) -> QIcon:
    """The mark at every standard size, for windows and the task bar."""
    icon = QIcon()
    for size in ICON_SIZES:
        icon.addPixmap(app_pixmap(size, dark, concept))
    return icon


def brand_markup(theme: Theme) -> str:
    """The name for a rich-text label: "Twin" in the text colour, "Scribe" in the accent."""
    twin = TEXT_LIGHT.name() if theme.dark else NAVY.name()
    return f'<span style="color:{twin};">Twin</span><span style="color:{theme.accent.name()};">Scribe</span>'


def _brand_font(pixel_size: int) -> QFont:
    font = QFont("Segoe UI")
    font.setStyleHint(QFont.StyleHint.SansSerif)
    font.setPixelSize(pixel_size)
    font.setWeight(QFont.Weight.Bold)
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 98.0)
    return font


def logo_image(height: int = 160, dark: bool = False, concept: str | None = None, tagline: str = "") -> QImage:
    """The mark beside the wordmark on a transparent ground, for the documentation; a tagline,
    when given, sits under the name in spaced capitals."""
    theme = theme_for(dark)
    mark = height
    gap = int(0.22 * height)
    name_font = _brand_font(int(0.58 * height))
    name_metrics = QFontMetricsF(name_font)
    twin_width = name_metrics.horizontalAdvance("Twin")
    name_width = twin_width + name_metrics.horizontalAdvance("Scribe")
    tag_font = QFont(name_font)
    tag_font.setPixelSize(max(8, int(0.15 * height)))
    tag_font.setWeight(QFont.Weight.Medium)
    tag_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 135.0)
    tag_text = tagline.upper()
    tag_metrics = QFontMetricsF(tag_font)
    tag_width = tag_metrics.horizontalAdvance(tag_text) if tag_text else 0.0
    width = int(mark + gap + max(name_width, tag_width) + 0.1 * height)
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        paint_app_icon(painter, float(mark), dark, concept)
        block = name_metrics.height() + (tag_metrics.height() + 0.04 * height if tag_text else 0.0)
        top = (height - block) / 2.0
        x = mark + gap
        baseline = top + name_metrics.ascent()
        painter.setFont(name_font)
        painter.setPen(TEXT_LIGHT if dark else NAVY)
        painter.drawText(QPointF(x, baseline), "Twin")
        painter.setPen(theme.accent)
        painter.drawText(QPointF(x + twin_width, baseline), "Scribe")
        if tag_text:
            painter.setFont(tag_font)
            painter.setPen(theme.muted)
            painter.drawText(QPointF(x + 0.01 * height, baseline + 0.04 * height + tag_metrics.ascent()), tag_text)
    finally:
        painter.end()
    return image


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
