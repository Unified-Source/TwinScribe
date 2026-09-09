# Specification: the mark and the wordmark

Read `CONVENTIONS.md` first. Deliver `twinscribe/app/app_icon.py` with `tests/test_app_icon.py`,
`tools/make_icon.py`, the rendered files under `assets/` and `docs/images/`, the wordmark in
the window's top bar, and the README's header.

## 1. Three marks, one chosen

Three marks are painted at call time, like every other icon in the program, so that no image
file is needed by the running program:

- **panels**: two facing panels, each carrying a waveform; the published engine's dark, the
  checking engine's light. This is the shape of the two-engine design and it reads at every
  size down to 16 pixels as two blocks and a gap. Light tile in the light palette, dark tile
  in the dark one.
- **monogram**: the initials, a white T and a blue S, with a waveform beside them on a dark
  tile.
- **document**: a page with text lines and a waveform through it, grey where it enters and
  blue where it leaves, on a dark tile.

`DEFAULT_CONCEPT` names the one the windows, the executables and the documentation use; the
other two stay available to `tools/make_icon.py --concept`, so that a choice between them is
made on rendered files at real sizes rather than on descriptions. Every mark uses the same
colours: the accent blue of the palette, a navy, a soft blue, a grey, white, and two tile
colours. Below 24 pixels every other bar of a waveform is drawn at twice the spacing, so the
bars stay separate.

## 2. The wordmark

"TwinScribe" set in a bold sans-serif with "Twin" in the text colour (navy in the light
palette, near-white in the dark) and "Scribe" in the accent. `brand_markup(theme)` gives it as
rich text for the window's top-bar label, which follows the palette when it changes.
`logo_image(height, dark, concept, tagline)` paints the mark beside the wordmark on a
transparent ground, with an optional tagline under the name in spaced capitals, for the
documentation.

## 3. Rendered files

`tools/make_icon.py` writes `assets/twinscribe.ico` (every standard size, PNG-compressed
entries) for the executables, `docs/images/icon.png`, and the logo in the light and the dark
colours to `docs/images/logo.png` and `docs/images/logo-dark.png`; `--concept` renders another
mark, `--tagline` adds a line under the name. The README's header shows the logo through a
`picture` element that picks the dark file under a dark colour scheme. The wordmark's font is
whichever bold sans-serif the rendering machine offers; the files are re-rendered where the
choice of mark changes, together with the window screenshots that show the mark in the top bar.

## 4. Tests

Every mark at every size in both palettes: transparent corners, a painted centre. Per mark,
sample pixels hand-derived from the geometry at 64 pixels: the tile, a panel, a bar, the T's
stem, the page, a grey bar and a blue bar. The default mark is one of the three; the wordmark
markup carries the two colours; the logo image holds the mark and painted text to its right.
The icon-file packer's header and directory arithmetic on small images.
