"""Assemble the site the project's domain serves: the page, its stylesheet and its script from
site/, with the pictures, the recording and the icon the page shows copied from where the
documentation keeps them, so that no picture is stored twice in the repository. Every local
reference in the page is then checked against the assembled folder, and the site's text files
are checked to hold ASCII only, as the conventions ask; either failing fails the build.

Usage:
    python tools/build_site.py --out _site
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = REPO_ROOT / "site"

# Published path -> repository path, for everything the page shows that lives elsewhere.
ASSETS: dict[str, str] = {
    "favicon.ico": "assets/twinscribe.ico",
    "images/icon.png": "docs/images/icon.png",
    "images/banner.png": "docs/images/banner.png",
    "images/window-transcribing.png": "docs/images/window-transcribing.png",
    "images/window-playing.png": "docs/images/window-playing.png",
    "images/verify.png": "docs/images/verify.png",
    "media/live-transcribe.mp4": "docs/media/live-transcribe.mp4",
}

TEXT_SUFFIXES = {".html", ".css", ".js", ".txt", ".svg"}

_REFERENCE = re.compile(r'\b(?:src|href|poster)="([^"]+)"')


def local_references(html: str) -> list[str]:
    """The relative paths a page refers to, in order of appearance: every src, href or poster
    that is not an address elsewhere, a fragment or inline data. A query or fragment on the
    path is dropped."""
    found: list[str] = []
    for value in _REFERENCE.findall(html):
        if value.startswith(("http://", "https://", "mailto:", "data:", "#", "//")):
            continue
        path = value.split("#", 1)[0].split("?", 1)[0]
        if path:
            found.append(path)
    return found


def assemble(out: Path, repo_root: Path = REPO_ROOT) -> list[Path]:
    """Copy site/ and the named assets into `out`, replacing what was there. Returns the files
    written, in the order written."""
    site = repo_root / "site"
    if out.exists():
        shutil.rmtree(out)
    written: list[Path] = []
    for source in sorted(site.rglob("*")):
        if source.is_file():
            target = out / source.relative_to(site)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            written.append(target)
    for published, stored in ASSETS.items():
        target = out / published
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / stored, target)
        written.append(target)
    return written


def missing_references(out: Path) -> list[str]:
    """Every local reference in the assembled pages that resolves to no file, as
    `page: reference` lines."""
    problems: list[str] = []
    for page in sorted(out.rglob("*.html")):
        html = page.read_text(encoding="utf-8")
        for reference in local_references(html):
            if not (page.parent / reference).is_file():
                problems.append(f"{page.relative_to(out)}: {reference}")
    return problems


def non_ascii(folder: Path) -> list[str]:
    """Every text file under `folder` holding a byte outside ASCII, with the first offending
    line, as `file:line` entries."""
    problems: list[str] = []
    for file in sorted(folder.rglob("*")):
        if file.suffix not in TEXT_SUFFIXES or not file.is_file():
            continue
        for number, line in enumerate(file.read_bytes().splitlines(), start=1):
            if any(byte > 127 for byte in line):
                problems.append(f"{file.relative_to(folder)}:{number}")
                break
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "_site", help="folder to assemble into")
    args = parser.parse_args(argv)

    written = assemble(args.out)
    print(f"assembled {len(written)} files into {args.out}")
    problems = [f"unresolved reference {p}" for p in missing_references(args.out)]
    problems += [f"non-ASCII text in {p}" for p in non_ascii(SITE_DIR)]
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
