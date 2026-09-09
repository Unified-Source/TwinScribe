"""Tests for the painted application icon and the icon-file packer, run under the offscreen
platform."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from twinscribe.app.app_icon import ICON_SIZES, app_icon, app_image, ico_entries, pack_ico  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    existing = QApplication.instance()
    return existing if existing is not None else QApplication([])


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


def test_app_icon_has_every_size_and_paints(app: QApplication) -> None:
    icon = app_icon()
    sizes = {(s.width(), s.height()) for s in icon.availableSizes()}
    assert sizes == {(s, s) for s in ICON_SIZES}
    image = app_image(64)
    assert image.format() == QImage.Format.Format_ARGB32_Premultiplied
    # Transparent corner, coloured centre, and a bright trace pixel in the middle band.
    assert image.pixelColor(0, 0).alpha() == 0
    assert image.pixelColor(32, 32).alpha() == 255
    bright = any(image.pixelColor(x, y).lightness() > 200 for x in range(10, 54) for y in range(20, 44))
    assert bright
    dark = app_image(32, dark=True)
    assert dark.pixelColor(16, 16).alpha() == 255
