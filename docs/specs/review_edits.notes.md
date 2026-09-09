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
- Applying is an explicit action (A, or the button), not a side effect of closing the screen:
  a listener may resolve a few marks over several sittings and apply once. Applying again
  replaces the earlier listener lines, so a changed resolution replaces rather than
  accumulates; open marks are left alone and counted in the status line.
- The Word document shows the listener's words in italics after a muted "(heard on review)";
  the plain text puts the suffix after the name; the pane shows the suffix muted and the
  words in the success colour. The subtitles carry the words as they are.

## Observed

- On a showcase chapter with forty marks: the first mark's prompt started from the hint, the
  accepted text became a listener line placed at the span's start with the speaker inferred
  from both sides, the second mark resolved as silent, and after Apply the transcript carried
  `review_applied` with one text, one silent and thirty-eight open, the text output carried
  the header sentence and the marked line, and the status line reported the counts.
- The offscreen test covers the same path on the fixture document, the pre-fill, the replace
  on a second apply, and the case of a review set with no transcript document beside it.

## Not done, and why

- Editing the engine's own words in place is out of scope: the design's rule is that the
  published engine's transcript is what the engine produced, verified against the recording;
  a person's corrections belong beside it, marked, which is what the listener lines are.
- The verification screen does not yet show the listener's earlier line for a mark that was
  applied before; the session note in the panel carries the same text.
