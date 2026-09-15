# Specification: the mark and the wordmark

Read `CONVENTIONS.md` first. Deliver `twinscribe/app/app_icon.py` with `tests/test_app_icon.py`,
`tools/make_icon.py`, the rendered files under `assets/` and `docs/images/`, the wordmark in
the window's top bar, and the README's header.

## 1. Three marks, one chosen

Three marks are painted at call time, like every other icon in the program, so that no image
file is needed by the running program:

- **panels**: two facing panels, each carrying a waveform, on a dark navy tile with a thin
  blue edge; the published engine's panel near white with navy bars, the checking engine's
  in one bright blue with white bars, and a dot between them. This is the shape of the
  two-engine design and it reads at every size down to 16 pixels as two blocks and a gap.
  The tile is dark in both palettes, so the mark carries its own ground wherever it is
  shown; below 24 pixels the edge and the dot are left out.
- **monogram**: the initials, a white T and a blue S, with a waveform beside them on a dark
  tile.
- **document**: a page with text lines and a waveform through it, grey where it enters and
  blue where it leaves, on a dark tile.

`DEFAULT_CONCEPT` names the one the windows, the executables and the documentation use; the
other two stay available to `tools/make_icon.py --concept`, so that a choice between them is
made on rendered files at real sizes rather than on descriptions. The panels mark keeps its
own few colours (the dark tile, the bright blue, the near-white panel, the navy bars); the
other two use the palette's accent, a navy, a soft blue, a grey, white and two tile colours.
Below 24 pixels every other bar of a waveform is drawn at twice the spacing, so the bars stay
separate.

## 2. The wordmark

"TwinScribe" set in a bold sans-serif with "Twin" in the text colour (navy in the light
palette, near-white in the dark) and "Scribe" in the accent. `brand_markup(theme)` gives it as
rich text for the window's top-bar label, which follows the palette when it changes. The
rendered files set the name on the mark's own dark ground, "Twin" in white and "Scribe" in the
mark's blue, with the tagline under it in spaced capitals: `banner_image(width, height,
tagline)` paints the mark beside the name, centred, for the wide banner; `logo_image(size,
tagline)` paints the mark above the name for the square lockup.

## 3. Rendered files

`tools/make_icon.py` writes `assets/twinscribe.ico` (every standard size, PNG-compressed
entries) for the executables, `docs/images/icon.png`, the square lockup to
`docs/images/logo.png` and the banner to `docs/images/banner.png`; `--concept` renders another
mark, `--tagline` changes the line under the name, `--font` the family. The README's header
shows the banner, whose dark ground serves both colour schemes. The wordmark's family is
named in the module and a machine without it renders its bold sans-serif; the files are
re-rendered where the mark changes, together with the window screenshots that show the mark in
the top bar.

## 4. Tests

Every mark at every size in both palettes: transparent corners, a painted centre. Per mark,
sample pixels hand-derived from the geometry at 64 pixels: the tile and its edge, a panel, a
bar, the dot, the T's stem, the page, a grey bar and a blue bar. The default mark is one of
the three; the wordmark markup carries the two colours; the banner and the lockup are opaque
on the dark ground, hold the mark where the layout puts it and painted text in the name's
colours. The icon-file packer's header and directory arithmetic on small images.
