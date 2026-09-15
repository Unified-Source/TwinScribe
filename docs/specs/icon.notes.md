# Notes: the mark and the wordmark

Companion to `icon.md`. Records what was built, what was checked, and what was left.

## Built and checked

- Three marks painted from three concept sketches, each at every standard size in both
  palettes, with the tile's corners transparent and the waveforms thinned below 24 pixels so
  the bars stay apart. A comparison sheet rendered outside the repository shows each mark at
  200 pixels on a light and a dark ground, the 48, 32, 24 and 16 pixel sizes, the lockup with
  the concept's tagline, and a title-bar strip with the 22 pixel mark beside the name; the
  panels mark was chosen on it.
- The wordmark in the window's top bar as rich text, "Twin" in navy and "Scribe" in the
  accent, near-white for "Twin" in the dark palette; it follows a palette change with the
  other themed widgets. Checked on the display with a window in each palette.
- `tools/make_icon.py` rendered the panels mark to `assets/twinscribe.ico`,
  `docs/images/icon.png`, and the lockup to `docs/images/logo.png` and `logo-dark.png`
  (672 by 160 pixels); the README's header shows the pair through a `picture` element.
- The README's window stills and the recording of a live transcription were taken again with
  the mark and the wordmark in the top bar: the two stills with the showcase chapters playing
  in each palette, and the recording as a fresh run of the first chapter at the Standard
  level, 23 seconds captured at the rate the window could be grabbed while the pipeline ran,
  encoded at that rate so the clip lasts as long as the run it shows.
- The suite: 438 tests pass, 10 skip; the icon tests sample hand-derived pixels of each mark.

## Choices

- The panels mark: it is the two-engine design drawn, the published engine's panel dark and
  the checking engine's light, and it reads at 16 pixels as two blocks and a gap. The
  monogram merges its S and bars below 32 pixels and says the name rather than the work; the
  document mark is clear from 48 pixels up but looks like many transcription tools at
  task-bar size. The other two remain in the module so that a later choice is made on
  rendered files.
- The mark's blue is the palette's accent, so that the icon, the primary button and the
  selection agree; the concept sketches used a brighter blue.
- The wordmark is not shipped as an image inside the program: the top bar sets it as text in
  the interface font, which follows the platform; only the documentation carries a rendered
  lockup, and that in the font of the machine that rendered it.
- The README's header shows the lockup with the tagline "Two engines. A clearer record." under
  the name; the lockup is the one place the tagline appears, since the window's top bar states
  what the program does instead.

## Not done, and why

- The offscreen platform used by the tests has no fonts on the development machine, so text
  renders as boxes there; `tools/make_icon.py` runs on the display platform for that reason,
  and the logo test checks for painted pixels rather than glyph shapes.
- The verification screen's still was not retaken: it carries no mark.

## The second drawing

- The owner asked for a more minimalistic banner, logo and icon and gave two pictures of the
  look wanted: the same two facing panels on a dark navy tile with a thin blue edge, the left
  panel near white with navy bars, the right one a bright blue with white bars, a dot between
  them, the name in white and blue on a dark ground, and the tagline in spaced capitals under
  it; one as a wide banner, one as a square lockup. The pictures carried gradients, a glow and
  soft shadows. They were not put into the repository: the program paints its marks, the icon
  file needs clean drawings at 16 and 24 pixels, and a flat drawing is the minimal version
  asked for; so the panels mark was redrawn from them in the same geometry with the new
  colours, and the lockup and the banner are painted from it.
- The tile is now dark in both palettes. The earlier light tile in the light palette made two
  marks of one program; the dark tile carries its own ground on the light window, the task
  bar and the page, and one banner file serves both colour schemes of the front page, which
  drops the light and dark pair of logo files.
- The mark's blue is no longer the palette's accent but the pictures' brighter blue; the
  primary button and the selection keep the accent. The window's top-bar wordmark keeps its
  text colours, since it sits on the window's own ground.
- The wordmark's family for the rendered files is named in the module and was chosen on a
  sheet of the banner in six families the rendering machine offers; a machine without the
  family renders its bold sans-serif in its place. The sheet showed the mark at 256, 64, 48,
  32, 24 and 16 pixels on a light and a dark ground, where the edge and the dot are left out
  below 24 pixels so the small sizes stay two blocks and a gap.
- Rendered again: the icon file, the icon image, the lockup (1024 pixels square) and the
  banner (1600 by 533) in place of the two logo files; the README's header shows the banner.
  The window stills and the live recording were taken again with the mark in the top bar.
