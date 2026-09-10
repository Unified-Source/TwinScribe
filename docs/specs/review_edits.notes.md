# Notes: review edits applied to the transcript

Companion to `review_edits.md`. Records the choices the specification left open and what was
observed.

## Choices

- The listener's words go into the transcript as a line of their own rather than into the
  neighbouring engine line, so that the engine's lines stay exactly as the engine produced
  them and every renderer can mark the line at a glance. The words are spread evenly across
  the publisher's silent span, which is the only timing there is; the subtitle cues therefore
  carry approximate times for those words, like the review marks themselves.
- The line's speaker is inferred only when the same speaker spoke immediately before and
  after the gap; otherwise the line is unlabelled and counts under the unlabelled entry of
  the speaker summary. A guess across a speaker change would be exactly the kind of silent
  error the summary exists to expose.
- The T prompt starts from the second engine's hint when there is no earlier note. This is
  deliberate: the hint is what a listener would otherwise have to retype, and what is
  accepted is recorded as the listener's, marked as such in every output, so the hint never
  reaches a transcript under the engine's name. The prompt says so, and takes several lines.
- Applying was at first an explicit action (A, or a button), so that a listener could resolve
  a few marks over several sittings and apply once. In use that step was missed: the typed
  words stayed in the session file and the transcript did not change. Applying now happens
  on every resolution and every reopening, which is safe because applying replaces the
  earlier listener lines rather than accumulating them; the session file remains the record
  a later sitting resumes from, open marks are left alone, and the status line counts what
  was written and what is still open.
- A mark selected again shows the listener's earlier words in the words box, so they can be
  changed or the mark reopened.
- The Word document shows the listener's words in italics after a muted "(heard on review)";
  the plain text puts the suffix after the name; the pane shows the suffix muted and the
  words in the success colour. The subtitles carry the words as they are.

## Observed

- On a showcase chapter with forty marks: the first mark's prompt started from the hint, the
  accepted text became a listener line placed at the span's start with the speaker inferred
  from both sides, the second mark resolved as silent, and after Apply the transcript carried
  `review_applied` with one text, one silent and thirty-eight open, the text output carried
  the header sentence and the marked line, and the status line reported the counts.
- On a copy of the same chapter with the first form of the screen: words typed for a mark
  and the screen closed without A left the transcript document and the text, Word and
  subtitle files unchanged; only the session file was written. With the revised screen the
  first kept words wrote the document and the three outputs at once (one span with the
  listener's words, thirty-nine still open) and reopening the mark took them out again.
- The offscreen test covers the same path on the fixture document, the pre-fill, the replace
  on a second keep, the reopening, and the case of a review set with no transcript document
  beside it.

## Not done, and why

- Editing the engine's own words in place is out of scope: the design's rule is that the
  published engine's transcript is what the engine produced, verified against the recording;
  a person's corrections belong beside it, marked, which is what the listener lines are.
