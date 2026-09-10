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

- a mark resolved with text gains a line of its own whose words are the typed words spread
  evenly across the publisher's silent span, each word carrying `"src": "listener"`, the line
  too; the line's speaker is the one speaking both immediately before and immediately after
  the gap, else none;
- a mark resolved as nothing said gains `"resolution": {"status": "nothing"}`; a mark with text
  gains the text as its resolution; open marks gain nothing;
- lines are kept in time order and the speaker summary is counted again from the lines, so
  the listener's words show in the counts;
- `review_applied` records when, how many spans carried text, how many were silent, how many
  are still open, and how many words the listener added.

The engine's words are never altered or removed, and the review list keeps the engine's own
account of each gap. Applying again first removes the earlier listener lines, so a changed
resolution replaces rather than accumulates. `apply_session(transcript, session, author)`
reads the session beside the review set, applies it, writes the document and renders the text,
Word and subtitle outputs again.

## 3. The verification screen

- The words box is pre-filled with the listener's earlier words, else with what the second
  engine heard in the span, so a listener edits rather than types; it accepts several lines.
- Keeping the words (Ctrl+Enter, or the button), resolving a span as silent (N) and reopening
  a mark (O) each write the session and at once apply it to the transcript document beside
  the review set, rendering the text, Word and subtitle outputs again; the status line says
  what was written. The main window follows every write while the screen is open and lays the
  recording out again when it closes, showing the listener's lines.

## 4. Rendering the listener's words

- Plain text: a listener line reads `[m:ss] Name (heard on review): words`; the header carries
  one sentence from `reviewed_note` when a review was applied.
- Word: the listener's words in italics with a muted "(heard on review)" after the name; the
  same header sentence among the metadata lines.
- Subtitles: the words as they are; a cue carries no provenance.
- The transcript pane: listener lines in the success colour and italics, with the same suffix.

## 5. Tests

Spread words and their source; speaker inference on both sides agreeing or not; the recount;
idempotence and replacement; the header sentence in each state; the session reader's
tolerance; `apply_session` writing the document and the three outputs; the screen's A action
writing a revised document with the listener's line (offscreen, with the text prompt replaced).
