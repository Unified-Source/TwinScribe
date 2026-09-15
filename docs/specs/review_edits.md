# Specification: review edits applied to the transcript

Read `CONVENTIONS.md` first, then `verify_app.md`.

Deliver `twinscribe/amend.py`, the additions to `twinscribe/app/verify.py`,
`twinscribe/outputs/plain_text.py`, `twinscribe/outputs/word_docx.py` and
`twinscribe/app/transcript_view.py`, and `tests/test_amend.py` with the added cases in
`tests/test_app_verify.py`.

## 1. Purpose

The verification screen records what a listener found at every mark. Until now that record
stayed beside the transcript as a session file and a note between the lines. A listener who
has heard what was said in a gap must be able to put it into the transcript itself, and every
output must show those words as the listener's, never as the engine's.

## 2. Applying a session (`twinscribe/amend.py`)

`apply_resolutions(doc, resolutions)` returns a new document, one resolution per mark:

- a mark resolved with text puts the typed words into the transcript where the gap is, spread
  evenly across the publisher's silent span, each word carrying `"src": "listener"` and the
  number of its mark, so that the reader meets them in the flow of what was said: inside the
  engine line the gap falls in, at the gap; else at the end of the line before the gap when
  that line's speaker is the words' speaker, else at the start of the line after it when
  that line's is; only when neither line is that speaker's do the words form a line of their
  own, carrying `"src": "listener"` itself. The words' speaker is the one the resolution names
  (`speaker`: a label, or an empty string for none), else the speaker of the line the gap
  falls inside, else the one speaking both immediately before and immediately after the gap,
  else none; a named speaker other than the one whose line the gap falls inside splits that
  line around the gap, every word as it was, and the words stand between the pieces. The
  speaker given is recorded in the mark's resolution;
- a mark resolved as nothing said gains `"resolution": {"status": "nothing"}`; a mark with text
  gains the text as its resolution; open marks gain nothing;
- lines are kept in time order and the speaker summary is counted again from the lines, so
  the listener's words show in the counts;
- `review_applied` records when, how many spans carried text, how many were silent, how many
  are still open, and how many words the listener added.

The engine's words are never altered or removed, and the review list keeps the engine's own
account of each gap. Applying again first takes the earlier listener words and lines out and
joins the split lines, so a changed resolution replaces rather than accumulates.
`apply_session(transcript, session, author)` reads the session beside the review set, applies
it, writes the document and renders the text, Word and subtitle outputs again.
`text_runs(line)` gives a line's text as runs of consecutive words with whether each run is
the listener's, for the outputs; `listener_line_for(doc, mark)` finds the line holding a
mark's words by the number they carry.

## 3. The verification screen

- The words box is pre-filled with the listener's earlier words, else with what the second
  engine heard in the span, so a listener edits rather than types; it accepts several lines.
- Keeping the words (Ctrl+Enter, or the button), resolving a span as silent (N) and reopening
  a mark (O) each write the session and at once apply it to the transcript document beside
  the review set, rendering the text, Word and subtitle outputs again; the status line says
  what was written. The main window follows every write while the screen is open and lays the
  recording out again when it closes, showing the listener's lines.

## 4. Rendering the listener's words

- Plain text: the listener's words stand in braces inside the line, `[m:ss] Name: the engine's
  words {the listener's} more of the engine's`; the header carries one sentence from
  `reviewed_note` when a review was applied, which says the words are marked in braces.
- Word: the listener's words in italics inside the line; the same header sentence, saying
  italics, among the metadata lines.
- Subtitles: the words as they are; a cue of a line of the listener's own carries the mark in
  its prefix, a cue of an engine line holding listener words carries nothing.
- The transcript pane: the listener's words in the success colour and italics inside the line.

## 5. Tests

Spread words and their source and mark; speaker inference on both sides agreeing or not; the
words joining the line before the gap, the line after it, the line around it, or standing on
their own, and a named speaker splitting a line; the recount; idempotence and replacement,
with the lines as they were once the words are taken out; the header sentence in each state;
the session reader's tolerance; `apply_session` writing the document and the three outputs;
the screen's action writing a revised document with the listener's words (offscreen, with the
text prompt replaced); the runs of a line and their rendering in braces and italics.
