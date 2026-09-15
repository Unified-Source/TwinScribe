"""Tests for the site's assembly: the reference parser on a hand-written snippet, the
assembled folder resolving every reference, the site's text files holding ASCII only, and
the page carrying the name and the tagline."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import build_site  # noqa: E402

TAGLINE = "Two engines. A clearer record."


def test_local_references_keep_the_relative_paths_only() -> None:
    html = (
        '<link rel="stylesheet" href="site.css">'
        '<a href="https://example.invalid/x">x</a>'
        '<a href="#design">d</a>'
        '<img src="images/icon.png?v=2" alt="">'
        '<video src="media/run.mp4" poster="images/still.png#top"></video>'
        '<img src="data:image/png;base64,AAAA" alt="">'
    )
    assert build_site.local_references(html) == [
        "site.css",
        "images/icon.png",
        "media/run.mp4",
        "images/still.png",
    ]


def test_assembled_site_resolves_every_reference(tmp_path: Path) -> None:
    out = tmp_path / "site"
    written = build_site.assemble(out)
    assert (out / "index.html").is_file()
    assert (out / "favicon.ico").is_file()
    assert all(path.is_file() for path in written)
    assert build_site.missing_references(out) == []


def test_assembly_replaces_a_stale_folder(tmp_path: Path) -> None:
    out = tmp_path / "site"
    out.mkdir()
    (out / "stale.txt").write_text("old", encoding="ascii")
    build_site.assemble(out)
    assert not (out / "stale.txt").exists()


def test_missing_reference_is_reported(tmp_path: Path) -> None:
    out = tmp_path / "site"
    out.mkdir()
    (out / "index.html").write_text('<img src="images/absent.png" alt="">', encoding="ascii")
    assert build_site.missing_references(out) == ["index.html: images/absent.png"]


def test_site_text_is_ascii_with_no_dash_but_the_hyphen() -> None:
    assert build_site.non_ascii(build_site.SITE_DIR) == []
    for file in build_site.SITE_DIR.rglob("*"):
        if file.suffix in build_site.TEXT_SUFFIXES:
            text = file.read_text(encoding="ascii")
            assert "–" not in text and "—" not in text


def test_non_ascii_is_found_with_its_line(tmp_path: Path) -> None:
    (tmp_path / "a.css").write_text("a { }\nb { content: '—'; }\n", encoding="utf-8")
    (tmp_path / "b.png").write_bytes(b"\x89PNG")
    assert build_site.non_ascii(tmp_path) == ["a.css:2"]


def test_page_carries_the_name_and_the_tagline() -> None:
    page = (build_site.SITE_DIR / "index.html").read_text(encoding="ascii")
    assert "<title>TwinScribe</title>" in page
    assert TAGLINE in page
    assert '<span class="twin">Twin</span><span class="scribe">Scribe</span>' in page
    assert "generator" not in page.lower()
