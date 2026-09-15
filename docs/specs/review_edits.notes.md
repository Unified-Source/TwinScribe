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

## Words inside a line, and the speaker

- The first rule, a speaker only when the same one spoke immediately before and after the
  gap, left the commonest gap unlabelled: on the showcase chapter thirty-eight of forty marks
  fall inside an engine line (a pause between 0.8 s and the 1.5 s line break), and a line
  that contains the gap is neither before nor after it. The rule now looks for the containing
  line first; the same voice is on both sides of such a pause by construction.
- Words kept for a gap inside a line were placed after the whole line, up to sixty words
  late, because the line is one entry that starts before the gap. The containing line is now
  split around the gap into two lines of the same speaker, every word untouched, and the
  pieces carry a `split_from` mark so that taking the listener's line out joins them again
  exactly. The speaker's seconds are therefore no longer counted twice over the gap.
- A resolution may name the speaker (`speaker`); the verification screen's "Spoken by" box
  sets it, and the recorded resolution carries the speaker the line was given, so a pass
  seeded from the document keeps it. `resolutions_from_document` and `listener_line_for` serve
  that seeding and the status line.
- The header of the text and Word outputs says when every mark has been checked, rather than
  that speech may still be missing; a subtitle cue of the listener's words carries the mark in
  its prefix, as every other output does.

## The words in the line

- Found by the owner on a one-reader chapter, reading the transcript after a review: the
  words kept for a gap stood as a line of their own between two lines of the same reader,
  with its own time and a "(heard on review)" tag, and the sentence around the gap read in
  three pieces. With the words already in the lines on either side (an echo of the bordering
  words, `review.notes.md`), the same words stood twice. The report: the review should
  correct the lines the engine wrote rather than insert orphan lines that break the reader's
  comprehension of what was said.
- The kept words now go into the transcript where the gap is. Inside an engine line, at the
  gap, without splitting the line; between lines, at the end of the line before the gap when
  that line's speaker is the words' speaker, else at the start of the line after it when that
  line's is. Only when neither neighbour is the speaker's do the words form a line of their
  own, as when the speakers differ on either side and none is chosen, or the listener names a
  third; and a named speaker other than the one whose line the gap falls inside still splits
  that line, since one voice's words cannot sit inside another's line. Every listener word
  carries its source and the number of its mark, so a changed or reopened resolution takes
  exactly those words out and the lines return to what the engine wrote.
- The provenance moved from the line to the words: the text output puts the listener's words
  in braces, the Word document sets them in italics, the pane shows them in the success colour
  and italics, and the header sentence of the text and Word outputs names the marking. The
  "(heard on review)" tag after the speaker's name is gone from lines, since a line can now be
  the engine's with the listener's words inside; a subtitle cue of a line of the listener's
  own keeps its prefix.
- The screen's status line still names the speaker the words were given, from the line that
  holds them; the captions say the words go into the line where the gap is.
