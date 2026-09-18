# Notes: the site

Companion to `site.md`. Records what was built, what was checked, and what was left.

## Built and checked

- One page under `site/`: `index.html`, `site.css` and `release.js`, with nothing loaded from
  elsewhere. The copy is the README's and the release notes', shortened; every hyphen is the
  ASCII one and the files are ASCII throughout, which the test holds.
- `tools/build_site.py` assembles ten files: the three of the page, the icon file as the
  favicon, five pictures and the recording. Every local reference in the page resolved in the
  assembled folder; the assembled folder served locally rendered the page in both palettes.
- Checked at a phone width of 375 pixels and at 1200: no horizontal overflow at either; the
  top bar's links wrap under the wordmark, the claim and the recording stack, the two panels
  stack flat, the figures fall into one column, and the install ways keep their numbers. At
  1200 the claim and the recording sit side by side, the panels face each other across the
  dot, and the figures sit in one row.
- The script against the live releases: the button took the zip's own address, the line
  under it read the name, `v0.0.2`, `1.56 GB` and "a pre-release", and the digest line showed
  the asset's SHA-256 as the release record carries it. The wordmark resolved to the family
  `app_icon.py` names on a machine that has it.
- `tests/test_site.py`: seven tests, all passing beside the suite.
- The tour, added 2026-09-18: 94 seconds at 1920 by 1080, 10.9 MB with its score, stored as
  `docs/media/tour.mp4` with its poster `docs/images/tour-poster.jpg` and copied at assembly,
  which now makes twelve files. The hero holds one column when it carries the tour, so the
  tour takes the measure under the claim; the real-run clip moves to the window's section,
  no longer set to play on its own. The script's reduced-motion rule covers whichever video is
  set to play on its own, which is now the tour. Checked again at 375 and 1200 pixels: no
  horizontal overflow, the tour under the claim at both, the poster shown until it plays.

## Choices

- The interface font, not a loaded one. The window's top bar sets the wordmark as text in
  the platform's interface font rather than shipping it as an image, and the page does the
  same, naming the rendered files' family first; a page that fetched a typeface from another
  service would sit oddly under a tool whose point is that nothing reaches the network.
- Dark by default, light on request. The mark carries its own dark ground and the banner
  is painted on it; the window has two palettes and so does the page, with the light one
  taken from the window's colours. The checking panel's fill is a step darker than the mark's
  blue so that white text on it reads at body size; the mark's blue is kept for the edge, the
  button and "Scribe".
- The recording, not the animated picture. The README shows the live run as a GIF because
  the front page there plays nothing else; the page uses the MP4 of the same clip, which is
  about half the size, and pauses it under a reduced-motion preference while leaving the
  controls in view. The MP4 first published was an older run whose top bar carried the mark
  before its redrawing; it was encoded again from the frames of the retaken run, 26 seconds
  at the animated picture's timing, so the clip, its poster and the page show one mark.
- The install section, 2026-09-18: the portable folder first, as the specification orders the
  ways, now that its parts are on the Releases page; the paragraph gives the two downloads,
  the third for an NVIDIA device, and the three steps, and says nothing is fetched.
- The tour leads and the run follows. A reader who has not opened the repository meets the
  idea before the window: the tour says what the second engine is for before the run shows it
  working, and the run keeps its place beside the description of the window, where a reader
  who wants the real thing looks. The tour is silent by default, since a page that starts
  playing sound is a page that gets closed; the score is one press away on the player's own
  control. Everything in it is the page's own material or synthesised, so the tour brings
  nothing to clear and nothing to credit.
- The download button asks for the newest release rather than naming a version in the
  markup, so the page does not go stale between releases; the markup's own address is the
  Releases page, which is right whatever the script does. `releases/latest` was not used: it
  resolves only to a release that is not marked as a pre-release, and both releases so far are.
- The pictures are copied at assembly rather than committed twice; the mapping of published
  path to stored file is one table in `build_site.py`, and the test that every reference
  resolves catches a picture renamed in the documentation.
- The two panels are drawn with a small perspective rotation on a layer under the text, so
  they face each other as the mark's do while the text stays flat and the corners stay
  rounded; below 720 pixels they stack flat.

## Not done, and why

- Publishing needs two settings outside the repository: the repository's Pages source set to
  the workflow, and the custom domain set on the repository with a matching record at the
  registrar. The workflow's first run fails until the first is made.
- The window's dark-palette still is not on the page; the two stills shown are the light
  ones, which read as windows on either ground.
- No analytics and no cookie of any kind, by design; nothing on the page reports the reader.
- The tour's source is not in the repository; a change to the page's copy or to the window's
  stills is not reflected in the tour until it is built again.
