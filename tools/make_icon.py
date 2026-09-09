"""Render the painted application icon to files for the parts of the system that need one:
a multi-size Windows icon for the executables and a PNG for the documentation.

Usage:
    python tools/make_icon.py [--ico assets/twinscribe.ico] [--png docs/images/icon.png] [--size 256]

Runs on the offscreen platform; the icon is shapes only, so no font is needed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QBuffer, QIODevice  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402

from twinscribe.app.app_icon import ICON_SIZES, app_image, pack_ico  # noqa: E402


def png_bytes(size: int) -> bytes:
    image = app_image(size)
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "PNG"):
        raise RuntimeError(f"could not encode the {size} px icon as PNG")
    return bytes(buffer.data())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the application icon to an .ico and a .png.")
    parser.add_argument("--ico", type=Path, default=REPO_ROOT / "assets" / "twinscribe.ico")
    parser.add_argument("--png", type=Path, default=REPO_ROOT / "docs" / "images" / "icon.png")
    parser.add_argument("--size", type=int, default=256, help="the PNG's size in pixels")
    args = parser.parse_args(argv)

    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    images = [(size, png_bytes(size)) for size in ICON_SIZES]
    args.ico.parent.mkdir(parents=True, exist_ok=True)
    args.ico.write_bytes(pack_ico(images))
    args.png.parent.mkdir(parents=True, exist_ok=True)
    if not app_image(args.size).save(str(args.png), "PNG"):
        raise RuntimeError(f"could not write {args.png}")
    print(f"{args.ico} ({args.ico.stat().st_size} bytes, sizes {', '.join(str(s) for s in ICON_SIZES)})")
    print(f"{args.png} ({args.size} px)")
    del app
    return 0


if __name__ == "__main__":
    sys.exit(main())
