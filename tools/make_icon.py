"""Render the painted mark to files for the parts of the system that need one: a multi-size
Windows icon for the executables, a PNG of the mark, the square lockup of the mark above the
wordmark and the tagline, and the wide banner of the mark beside them, on the mark's own dark
ground.

Usage:
    python tools/make_icon.py [--concept panels|monogram|document] [--tagline TEXT]
        [--ico assets/twinscribe.ico] [--png docs/images/icon.png] [--size 256]
        [--logo docs/images/logo.png] [--logo-size 1024]
        [--banner docs/images/banner.png] [--banner-width 1600] [--font FAMILY]

The mark is shapes only and renders anywhere; the wordmark needs the platform's fonts, which
the offscreen platform lacks, so the tool runs on the display platform and the lockup and the
banner come out in the bold sans-serif the machine offers under the family asked for. --no-logo
renders the icon files only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PySide6.QtCore import QBuffer, QIODevice  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402

from twinscribe.app.app_icon import (  # noqa: E402
    CONCEPTS,
    DEFAULT_CONCEPT,
    ICON_SIZES,
    TAGLINE,
    app_image,
    banner_image,
    logo_image,
    pack_ico,
)


def png_bytes(size: int, concept: str) -> bytes:
    image = app_image(size, concept=concept)
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "PNG"):
        raise RuntimeError(f"could not encode the {size} px icon as PNG")
    return bytes(buffer.data())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the mark to an .ico, a .png, the lockup and the banner.")
    parser.add_argument("--concept", default=DEFAULT_CONCEPT, choices=sorted(CONCEPTS), help="which mark to render")
    parser.add_argument("--ico", type=Path, default=REPO_ROOT / "assets" / "twinscribe.ico")
    parser.add_argument("--png", type=Path, default=REPO_ROOT / "docs" / "images" / "icon.png")
    parser.add_argument("--size", type=int, default=256, help="the PNG's size in pixels")
    parser.add_argument("--logo", type=Path, default=REPO_ROOT / "docs" / "images" / "logo.png", help="the square lockup")
    parser.add_argument("--logo-size", type=int, default=1024, help="the lockup's size in pixels")
    parser.add_argument("--banner", type=Path, default=REPO_ROOT / "docs" / "images" / "banner.png", help="the wide banner")
    parser.add_argument("--banner-width", type=int, default=1600, help="the banner's width in pixels; the height is a third of it")
    parser.add_argument("--tagline", default=TAGLINE, help="the line under the name in the lockup and the banner")
    parser.add_argument("--font", default=None, help="the wordmark's font family (default: the module's)")
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
        lockup = logo_image(args.logo_size, tagline=args.tagline, concept=args.concept, family=args.font)
        args.logo.parent.mkdir(parents=True, exist_ok=True)
        if not lockup.save(str(args.logo), "PNG"):
            raise RuntimeError(f"could not write {args.logo}")
        print(f"{args.logo} ({lockup.width()} x {lockup.height()} px)")
        banner = banner_image(args.banner_width, args.banner_width // 3, tagline=args.tagline, concept=args.concept, family=args.font)
        args.banner.parent.mkdir(parents=True, exist_ok=True)
        if not banner.save(str(args.banner), "PNG"):
            raise RuntimeError(f"could not write {args.banner}")
        print(f"{args.banner} ({banner.width()} x {banner.height()} px)")
    del app
    return 0


if __name__ == "__main__":
    sys.exit(main())
