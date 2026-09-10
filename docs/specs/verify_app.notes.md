# Notes: the verification screen

Delivered: `twinscribe/app/__init__.py`, `twinscribe/app/verify.py`, `twinscribe/app/theme.py`,
`twinscribe/app/timeline.py`, `tests/test_app_verify.py`. Built against PySide6 6.11 with
QtMultimedia; the tests run under `QT_QPA_PLATFORM=offscreen` and use `QtTest.QTest` for keys.

## Deviations from the specification, and why

1. The specification points at `engines_and_review.md` section 6 for the review-set document.
   That file does not exist; the document is defined in `review.md` section 1 and the screen
   reads that shape. The screen parses the JSON itself and does not import `twinscribe.review`,
   so the app package stands on its own and the review module can change independently.
2. An existing `<stem>.session.json` beside the review set is loaded when the screen opens, so
   a session interrupted by a crash or a close resumes where it stopped. The specification only
   says the file is written; writing it "so a crash loses nothing" is of little use unless it is
   read back. A session is accepted only when it has one entry per mark and every entry's start
   and end agree with the mark's within a millisecond; anything else is ignored, so a stale file
   from a different review set can never attach itself to this one.
3. On close the session is written only when at least one mark is resolved or the file already
   exists. Unconditional writing left an all-open session file behind every screenshot run and
   every open-and-close. Every resolution still writes the file at once, as specified.
4. The session file lists every mark, resolved or not. The specification gives the entry shape
   (`start`, `end`, `status`, `note`) without naming the status values; the values used are
   `open`, `nothing` and `text`. An unresolved mark has status `open` and an empty note.
5. After N or T the screen advances to the next mark in list order; on the last mark the
   selection stays where it is. The specification says "advance" without saying where to.
6. A relative `audio` path in the review set is resolved against the review set's folder; an
   absolute path is used as given. The specification does not say which folder a relative path
   is relative to, and beside the review set is the only sensible reading.
7. The header separates its four parts with a vertical bar, since dashes are not allowed.
8. The share of the recording the marks cover is computed from the union of the marks' padded
   play spans, not read from the evaluation, so the summary line reads the same with and
   without an evaluation. The evaluation supplies only the on-speech count and the dropped-word
   coverage.
9. The window title is `<n> of <total> done | twinscribe verify | <audio name>`; the required
   text is the prefix.
10. T opens no prompt: the words are typed in a box that is always in view (the section at
    the end of these notes) and Ctrl+Enter keeps them, so no dialog ever blocks and the tests
    type into the box. The first form of the screen used a prompt; it is gone.
11. Keys are handled twice over: in the window's `keyPressEvent` and in an event filter on the
    focusable children. A `QListWidget` turns unhandled letters into a keyboard search and
    consumes Space, and a read-only text panel scrolls on Space and the arrows, so without the
    filter the keys would work only while no child had focus. Ctrl and Alt combinations are left
    alone.
12. The tick prefix on a resolved row is U+2713, written as an escape so the source stays ASCII.
13. When the review set has no marks the screen still opens, the four buttons and the words
    box are disabled and the keys do nothing. Not specified; avoids a crash on an empty list.

## QtMultimedia on this platform

Observed with PySide6 6.11.2 on Windows 11 (64-bit ARM), whose QtMultimedia uses the FFmpeg
backend.

- Under `QT_QPA_PLATFORM=offscreen`, `QMediaPlayer` and `QAudioOutput` construct silently and
  emit no warning or error. The tests never set a source (there is no audio file), and nothing
  is logged. The specification anticipated that the missing file would need handling; it did
  not anticipate that the multimedia objects themselves would be entirely quiet offscreen.
- On the real backend, `setSource` prints to stderr, outside Qt's logging categories: an
  `Input #0, wav, from '...'` block describing the stream, and two `MFT name:` lines from the
  h264 and hevc Media Foundation encoder probes, even though only audio is played. This is
  harmless noise from the backend's own logging and is not produced by this code.
- `play()` after a seek moves `mediaStatus` LoadedMedia -> BufferingMedia -> BufferedMedia, and
  about one second passes before `positionChanged` starts advancing (position 3.16 s at 1.1 s of
  wall time after starting at 3.1 s). The span is heard in full, just late by that much.
- `positionChanged` fires at roughly 30 ms steps. The automatic pause at a span end lands within
  one step past the end (6.43 s for an end of 6.4 s).
- Seeking while paused returns `mediaStatus` to LoadedMedia; playing again returns it to
  BufferedMedia. `setSource(QUrl())` on close returns it to NoMedia with no error.
- `QWidget.grab()` at 800 ms after `show()` produced a complete rendering for `--shot` in both
  palettes on the real backend. Under the offscreen platform every glyph is a box, as the design
  record warns; the module docstring records this.

## The words box, and the transcript written on every resolution

Observed on a copy of a real transcript (an audiobook chapter with forty marks): with the
first form of the screen, typing words for a mark through the prompt and closing the screen
changed only the session file; the transcript document and the text, Word and subtitle
files were unchanged until the A key was pressed, and nothing on the screen said so. The
revised screen wrote the document and the three outputs on the first kept words (status
line: one span with the listener's words, thirty-nine still open) and took them out again
on reopening.

- The words box is a plain text editor under the same event filter as the other children,
  but it gives up only Ctrl+Enter and Esc; every other key types. The filter returns False
  for those keys so the editor still receives them. The single-letter keys of the screen
  (N, T, O, J, K) therefore work from the list and the buttons, not from inside the box; T
  moves the cursor into the box with its text selected, Esc moves it back.
- A span is played when its mark is selected, except the first selection when the screen
  opens, since nothing has been asked for yet; a resolution that advances plays the next
  span without touching the status line, so the sentence about what was written stays
  readable. Offscreen there is no audio: the test sets the availability flag and checks the
  pending stop time instead.
- Reopening does not advance, and refills the box with the second engine's hint.
- The tick on a resolved row is followed by "nothing said" or the kept words cut to twenty
  four characters; the full words are in the box when the row is selected.
- With no transcript document beside the review set (a review set opened on its own), the
  session file alone keeps the resolutions, and both the opening status and each
  resolution's status say so.
- The A key and the Apply button are gone; `apply_to_transcript()` remains for a caller that
  wants the document written again from the resolutions held.
