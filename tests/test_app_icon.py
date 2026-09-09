"""Tests for the painted marks, the wordmark and the icon-file packer, run under the offscreen
platform. Sample pixels are hand-derived from the geometry at 64 pixels."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from twinscribe.app.app_icon import (  # noqa: E402
    CONCEPT_DOCUMENT,
    CONCEPT_MONOGRAM,
    CONCEPT_PANELS,
    CONCEPTS,
    DEFAULT_CONCEPT,
    ICON_SIZES,
    app_icon,
    app_image,
    brand_markup,
    ico_entries,
    logo_image,
    pack_ico,
)
from twinscribe.app.theme import theme_for  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    existing = QApplication.instance()
    return existing if existing is not None else QApplication([])


def bluish(colour: QColor) -> bool:
    return colour.blue() > colour.red() + 40


def greyish(colour: QColor) -> bool:
    channels = (colour.red(), colour.green(), colour.blue())
    return max(channels) - min(channels) < 40


def test_ico_packer_arithmetic() -> None:
    images = [(16, b"a" * 10), (256, b"b" * 20)]
    data = pack_ico(images)
    # 6-byte header, two 16-byte entries, then the payloads in order.
    assert len(data) == 6 + 32 + 30
    assert ico_entries(data) == [(16, 10, 38), (256, 20, 48)]
    assert data[38:48] == b"a" * 10 and data[48:68] == b"b" * 20
    with pytest.raises(ValueError):
        pack_ico([])
    with pytest.raises(ValueError):
        pack_ico([(512, b"x")])
    with pytest.raises(ValueError):
        ico_entries(b"\x01\x00\x01\x00\x00\x00")


@pytest.mark.parametrize("concept", sorted(CONCEPTS))
def test_every_mark_has_every_size_and_transparent_corners(app: QApplication, concept: str) -> None:
    icon = app_icon(concept=concept)
    sizes = {(s.width(), s.height()) for s in icon.availableSizes()}
    assert sizes == {(s, s) for s in ICON_SIZES}
    for size in ICON_SIZES:
        for dark in (False, True):
            image = app_image(size, dark, concept)
            assert image.format() == QImage.Format.Format_ARGB32_Premultiplied
            assert image.pixelColor(0, 0).alpha() == 0 and image.pixelColor(size - 1, size - 1).alpha() == 0
            assert image.pixelColor(size // 2, size // 2).alpha() == 255


def test_panels_mark_geometry(app: QApplication) -> None:
    light = app_image(64, False, CONCEPT_PANELS)
    # The tile at (3, 32): full width at mid-height, light. The left panel spans x 8 to 30 at
    # mid-height, so (12, 32) is navy; its middle bar is centred on x 19.2 and 2.2 wide, so
    # column 19 is white. The right panel spans x 34 to 56, so (51, 32) is the soft blue and
    # its middle bar, centred on x 44.8, makes column 44 white.
    assert light.pixelColor(3, 32).lightness() > 200
    assert bluish(light.pixelColor(12, 32)) and light.pixelColor(12, 32).lightness() < 120
    assert light.pixelColor(19, 32).lightness() > 200
    assert bluish(light.pixelColor(51, 32)) and 100 < light.pixelColor(51, 32).lightness() < 200
    assert light.pixelColor(44, 32).lightness() > 200
    dark = app_image(64, True, CONCEPT_PANELS)
    assert dark.pixelColor(3, 32).lightness() < 80
    assert bluish(dark.pixelColor(12, 32)) and dark.pixelColor(19, 32).lightness() > 200
    small = app_image(16, False, CONCEPT_PANELS)
    assert small.pixelColor(8, 8).alpha() == 255


def test_monogram_mark_geometry(app: QApplication) -> None:
    image = app_image(64, False, CONCEPT_MONOGRAM)
    # The T's stem spans x 17.3 to 23.7 and y 15.4 to 44.8, so (20, 30) is white; the third
    # bar of the waveform is centred on x 55.4, 21.8 tall about y 32, so (55, 32) is the soft blue.
    assert image.pixelColor(20, 30).lightness() > 240
    assert bluish(image.pixelColor(55, 32)) and 100 < image.pixelColor(55, 32).lightness() < 220
    assert image.pixelColor(3, 32).lightness() < 80
    # The S passes through the middle band between the T and the bars in a blue.
    assert any(bluish(image.pixelColor(x, 44)) for x in range(36, 46))


def test_document_mark_geometry(app: QApplication) -> None:
    image = app_image(64, False, CONCEPT_DOCUMENT)
    # The page spans x 15.4 to 48.6 and y 9.6 to 54.4; (19, 51) lies below the last text line
    # (centred on y 48, 2.2 thick) and the waveform, so it is white. Bar 4 is grey, centred on
    # x 24.3 and 14 tall about y 33.9; bar 8 is blue, centred on x 39.7 and 15.4 tall.
    assert image.pixelColor(19, 51).lightness() > 240
    assert greyish(image.pixelColor(24, 32)) and 120 < image.pixelColor(24, 32).lightness() < 180
    assert bluish(image.pixelColor(39, 32)) and image.pixelColor(39, 32).lightness() < 160
    assert image.pixelColor(3, 32).lightness() < 80


def test_default_mark_and_wordmark(app: QApplication) -> None:
    assert DEFAULT_CONCEPT in CONCEPTS
    assert app_image(32).constBits() == app_image(32, concept=DEFAULT_CONCEPT).constBits()
    light = brand_markup(theme_for(False))
    assert "Twin" in light and "Scribe" in light and theme_for(False).accent.name() in light
    dark = brand_markup(theme_for(True))
    assert "#e8e8ea" in dark and "#1e2f6b" not in dark


def test_logo_image_has_the_mark_and_text(app: QApplication) -> None:
    image = logo_image(80, dark=False, tagline="Two engines. A clearer record.")
    assert image.height() == 80 and image.width() > 240
    assert image.pixelColor(0, 0).alpha() == 0 and image.pixelColor(40, 40).alpha() == 255
    # Text pixels to the right of the mark: some in the name's colours, none outside the image.
    painted = [(x, y) for x in range(100, image.width()) for y in range(0, 80) if image.pixelColor(x, y).alpha() > 0]
    assert painted
    dark = logo_image(80, dark=True)
    assert dark.width() < image.width() or dark.width() == image.width()
    assert any(dark.pixelColor(x, y).lightness() > 200 for x, y in painted[:2000] if x < dark.width())
