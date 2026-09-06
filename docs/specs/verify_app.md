# Specification: the verification screen

Read `CONVENTIONS.md` first, then section 9 of `../Design_and_Findings_2026-09-06.md` and
section 6 of `engines_and_review.md` for the review-set JSON it reads.

Deliver `twinscribe/app/__init__.py`, `twinscribe/app/verify.py`, `twinscribe/app/theme.py`,
`twinscribe/app/timeline.py`, and `tests/test_app_verify.py`. PySide6 6.8 or later with
QtMultimedia. The entry point is `python -m twinscribe.app.verify <review_set.json> [--dark] [--shot out.png]`.

## 1. Purpose

A person works through the review list with the audio playing at each mark, and records for
each one either that nothing was said or what was said. The screen exists so that the
published engine's silent failures become a short, ordered list of places to listen to.

## 2. Layout, top to bottom

1. Header line: the audio file name, its duration in minutes, "published by <engine model>",
   "checked against <engine model>, which is never published".
2. Summary line: the number of marks and the share of the recording they cover; when the
   review set carries an evaluation, also how many marks were on real speech and what share
   of the dropped words the marks cover.
3. Timeline (`timeline.py`): one bar for the whole recording; every mark drawn on it in the
   accent colour, resolved marks in the success colour, the current mark outlined; a playhead
   line; clicking the bar seeks. Minimum height 70 px. Time labels at both ends.
4. A horizontal splitter:
   - left: the list of marks, one row each: index, start time as m:ss.t, span length, number
     of detector words; a tick prefix once resolved;
   - right: the current mark. A bold "start to end (n seconds)" line; a read-only panel
     "what the published transcript has here" showing up to fourteen published words either
     side of the gap with a visible `[ GAP ]` between; a read-only panel "what the second
     engine heard, as a hint of what to listen for"; and, only when the review set carries
     reference information, a clearly labelled test-only panel stating how many reference
     words fall in the span and which speakers, or that the mark is a false alarm. Then three
     buttons: "Play this span (Enter)", "Nothing was said (N)", "Type what was said (T)", and
     a status line.
5. Footer line listing the keys.

## 3. Behaviour

- Keys: Space play or pause; J and K next and previous mark; Enter play the current span
  (seek to `start`, play, pause automatically at `end`); N resolve as nothing said and advance;
  T open a text prompt pre-filled with any earlier note, resolve with the typed text and
  advance; Left and Right nudge five seconds.
- Selecting a mark seeks to its `start`.
- Playback through `QMediaPlayer` and `QAudioOutput` from the review set's `audio` path;
  when the file is missing, playback controls do nothing and the status line says so, and the
  screen otherwise works.
- Resolutions are kept in memory and written on close to `<review_set stem>.session.json`
  beside the review set: `{"schema": "twinscribe.review-session.v1", "marks": [{"start", "end", "status", "note"}]}`.
  Also written whenever a mark is resolved, so a crash loses nothing.
- Window title shows "<n> of <total> done".

## 4. Theme (`theme.py`)

Fusion style with a hand-set palette; one accent colour used only for state, a success
colour for resolved marks, a warning colour for the test-only panel when words were really
spoken. Light and dark palettes; `--dark` selects dark. No web fonts, no icon fonts, no
external assets of any kind.

## 5. Screenshots

`--shot out.png` renders the window, saves it after a short delay, and exits. Document in
the module docstring that screenshots must be taken on the platform's real backend; the
offscreen platform has no font database and renders every glyph as a box.

## 6. Tests

Under `QT_QPA_PLATFORM=offscreen`, with a synthetic review set written to `tmp_path` and no
audio file: the window constructs; the list has one row per mark; selecting a row updates the
current-mark panels with the expected context and gap; N resolves and advances; T with a
value resolves with that text; J and K move; the session file is written with the resolutions;
the test-only panel is absent when the review set has no reference and present when it does.
Use `QtTest.QTest` for key events. Skip the whole module if PySide6 is not importable.

## 7. Notes file

`docs/specs/verify_app.notes.md`: deviations and anything QtMultimedia did on this platform
that the specification did not anticipate.
