"""The application marks, painted at call time like every other icon so that no image file
ships, and the wordmark.

Three marks are drawn. "panels": two facing panels each carrying a waveform on a dark tile
with a thin blue edge, the published engine's light with navy bars and the checking engine's
blue with white bars, a dot between them, which is the shape of the two-engine design.
"monogram": the initials with a waveform beside them. "document": a page with text lines and
a waveform running through it, grey where it enters and blue where it leaves. The windows, the
executables and the documentation use the mark named by DEFAULT_CONCEPT; the other two stay
available to `tools/make_icon.py --concept`, so that the choice is made on rendered files. The
wordmark in the window sets "Twin" in the text colour and "Scribe" in the accent; the rendered
banner and lockup set "Twin" in white and "Scribe" in the mark's blue on the mark's own dark
ground, with the tagline in spaced capitals under the name.
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
# The panels mark and the rendered files: a dark navy ground, a thin edge and the checking
# engine's panel in one bright blue, the published engine's panel near white with navy bars.
TILE = QColor("#0e1629")
TILE_EDGE = QColor("#2f7cf6")
GROUND = QColor("#0a1020")
PANEL_LIGHT = QColor("#eef1f7")
PANEL_BLUE = QColor("#2f7cf6")
BARS_NAVY = QColor("#141c33")
DOT = QColor("#b9c4dd")
NAME_LIGHT = QColor("#f4f6fb")
TAGLINE_COLOUR = QColor("#aab4cc")
TAGLINE = "Two engines. A clearer record."
# The rendered wordmark's family; a machine without it falls back to its bold sans-serif.
BRAND_FONT = "Segoe UI Variable Display"
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
PANEL_BARS: tuple[float, ...] = (0.10, 0.20, 0.32, 0.20, 0.10)
DOT_HEIGHT = 0.06
EDGE_WIDTH = 0.025
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


def _edge(painter: QPainter, size: float, colour: QColor) -> None:
    """A thin line along the tile's rounded edge, inside it."""
    width = max(1.0, EDGE_WIDTH * size)
    pen = QPen(colour)
    pen.setWidthF(width)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    inset = width / 2.0
    radius = TILE_RADIUS * size - inset
    painter.drawRoundedRect(QRectF(inset, inset, size - width, size - width), radius, radius)


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
    """Two facing panels on a dark tile, whatever the palette: the published engine's light
    with navy bars, the checking engine's blue with white bars, a dot between them, and a
    thin blue edge along the tile from 24 pixels up."""
    _tile(painter, size, TILE)
    if size >= SMALL:
        _edge(painter, size, TILE_EDGE)
    _rounded_polygon(painter, LEFT_PANEL, size, PANEL_RADIUS, PANEL_LIGHT)
    _rounded_polygon(painter, RIGHT_PANEL, size, PANEL_RADIUS, PANEL_BLUE)
    _bars(painter, size, 0.30, 0.50, PANEL_BARS, 0.065, BARS_NAVY)
    _bars(painter, size, 0.70, 0.50, PANEL_BARS, 0.065, WHITE)
    if size >= SMALL:
        _bars(painter, size, 0.50, 0.50, (DOT_HEIGHT,), 0.065, DOT)


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


def _brand_font(pixel_size: int, family: str | None = None) -> QFont:
    font = QFont(family or BRAND_FONT)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    font.setPixelSize(pixel_size)
    font.setWeight(QFont.Weight.Bold)
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 98.0)
    return font


def _tagline_font(pixel_size: int, family: str | None = None) -> QFont:
    font = _brand_font(max(8, pixel_size), family)
    font.setWeight(QFont.Weight.Medium)
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 150.0)
    return font


def _paint_name(painter: QPainter, x: float, baseline: float, font: QFont) -> float:
    """"TwinScribe" from x on the baseline, "Twin" in white and "Scribe" in the mark's blue;
    returns the width painted."""
    metrics = QFontMetricsF(font)
    twin_width = metrics.horizontalAdvance("Twin")
    painter.setFont(font)
    painter.setPen(NAME_LIGHT)
    painter.drawText(QPointF(x, baseline), "Twin")
    painter.setPen(PANEL_BLUE)
    painter.drawText(QPointF(x + twin_width, baseline), "Scribe")
    return twin_width + metrics.horizontalAdvance("Scribe")


def banner_image(width: int = 1600, height: int = 534, tagline: str = TAGLINE, concept: str | None = None, family: str | None = None) -> QImage:
    """The mark beside the wordmark, with the tagline under the name in spaced capitals, centred
    on the dark ground: the banner of the repository's front page."""
    mark = int(0.44 * height)
    gap = int(0.16 * height)
    name_font = _brand_font(int(0.30 * height), family)
    name_metrics = QFontMetricsF(name_font)
    name_width = name_metrics.horizontalAdvance("TwinScribe")
    tag_font = _tagline_font(int(0.075 * height), family)
    tag_metrics = QFontMetricsF(tag_font)
    tag_text = tagline.upper()
    tag_width = tag_metrics.horizontalAdvance(tag_text) if tag_text else 0.0
    block_width = mark + gap + max(name_width, tag_width)
    left = (width - block_width) / 2.0
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(GROUND)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.translate(left, (height - mark) / 2.0)
        paint_app_icon(painter, float(mark), False, concept)
        painter.resetTransform()
        x = left + mark + gap
        text_height = name_metrics.capHeight() + (0.14 * height + tag_metrics.capHeight() if tag_text else 0.0)
        baseline = (height - text_height) / 2.0 + name_metrics.capHeight()
        _paint_name(painter, x, baseline, name_font)
        if tag_text:
            painter.setFont(tag_font)
            painter.setPen(TAGLINE_COLOUR)
            painter.drawText(QPointF(x + 0.01 * height, baseline + 0.14 * height + tag_metrics.capHeight()), tag_text)
    finally:
        painter.end()
    return image


def logo_image(size: int = 1024, tagline: str = TAGLINE, concept: str | None = None, family: str | None = None) -> QImage:
    """The mark above the wordmark and the tagline on the dark ground, a square: the
    repository's logo."""
    mark = int(0.42 * size)
    name_font = _brand_font(int(0.15 * size), family)
    name_metrics = QFontMetricsF(name_font)
    name_width = name_metrics.horizontalAdvance("TwinScribe")
    tag_font = _tagline_font(int(0.04 * size), family)
    tag_metrics = QFontMetricsF(tag_font)
    tag_text = tagline.upper()
    tag_width = tag_metrics.horizontalAdvance(tag_text) if tag_text else 0.0
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(GROUND)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.translate((size - mark) / 2.0, 0.17 * size)
        paint_app_icon(painter, float(mark), False, concept)
        painter.resetTransform()
        baseline = 0.17 * size + mark + 0.08 * size + name_metrics.capHeight()
        _paint_name(painter, (size - name_width) / 2.0, baseline, name_font)
        if tag_text:
            painter.setFont(tag_font)
            painter.setPen(TAGLINE_COLOUR)
            painter.drawText(QPointF((size - tag_width) / 2.0, baseline + 0.075 * size + tag_metrics.capHeight()), tag_text)
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
