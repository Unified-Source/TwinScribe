"""Render the painted mark to files for the parts of the system that need one: a multi-size
Windows icon for the executables, a PNG of the mark, and the mark beside the wordmark on a
transparent ground for the documentation, in the light and the dark colours.

Usage:
    python tools/make_icon.py [--concept panels|monogram|document] [--tagline TEXT]
        [--ico assets/twinscribe.ico] [--png docs/images/icon.png] [--size 256]
        [--logo docs/images/logo.png] [--logo-dark docs/images/logo-dark.png] [--logo-height 160]

The mark is shapes only and renders anywhere; the wordmark needs the platform's fonts, which
the offscreen platform lacks, so the tool runs on the display platform and the logo files come
out in whichever bold sans-serif the machine offers. --no-logo renders the icon files only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PySide6.QtCore import QBuffer, QIODevice  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402

from twinscribe.app.app_icon import CONCEPTS, DEFAULT_CONCEPT, ICON_SIZES, app_image, logo_image, pack_ico  # noqa: E402


def png_bytes(size: int, concept: str) -> bytes:
    image = app_image(size, concept=concept)
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "PNG"):
        raise RuntimeError(f"could not encode the {size} px icon as PNG")
    return bytes(buffer.data())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the mark to an .ico, a .png and the logo files.")
    parser.add_argument("--concept", default=DEFAULT_CONCEPT, choices=sorted(CONCEPTS), help="which mark to render")
    parser.add_argument("--ico", type=Path, default=REPO_ROOT / "assets" / "twinscribe.ico")
    parser.add_argument("--png", type=Path, default=REPO_ROOT / "docs" / "images" / "icon.png")
    parser.add_argument("--size", type=int, default=256, help="the PNG's size in pixels")
    parser.add_argument("--logo", type=Path, default=REPO_ROOT / "docs" / "images" / "logo.png", help="mark and wordmark, light colours")
    parser.add_argument("--logo-dark", type=Path, default=REPO_ROOT / "docs" / "images" / "logo-dark.png", help="the same in the dark colours")
    parser.add_argument("--logo-height", type=int, default=160, help="the logo files' height in pixels")
    parser.add_argument("--tagline", default="", help="a line under the name in the logo files")
    parser.add_argument("--no-logo", action="store_true", help="render the icon files only")
    args = parser.parse_args(argv)

    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    images = [(size, png_bytes(size, args.concept)) for size in ICON_SIZES]
    args.ico.parent.mkdir(parents=True, exist_ok=True)
    args.ico.write_bytes(pack_ico(images))
    args.png.parent.mkdir(parents=True, exist_ok=True)
    if not app_image(args.size, concept=args.concept).save(str(args.png), "PNG"):
        raise RuntimeError(f"could not write {args.png}")
    print(f"{args.ico} ({args.ico.stat().st_size} bytes, sizes {', '.join(str(s) for s in ICON_SIZES)}, {args.concept})")
    print(f"{args.png} ({args.size} px)")
    if not args.no_logo:
        for path, dark in ((args.logo, False), (args.logo_dark, True)):
            path.parent.mkdir(parents=True, exist_ok=True)
            image = logo_image(args.logo_height, dark=dark, concept=args.concept, tagline=args.tagline)
            if not image.save(str(path), "PNG"):
                raise RuntimeError(f"could not write {path}")
            print(f"{path} ({image.width()} x {image.height()} px)")
    del app
    return 0


if __name__ == "__main__":
    sys.exit(main())
