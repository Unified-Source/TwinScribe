# Notes: history and export

Companion to `history_export.md`. Records the choices the specification left open and what
was observed.

## Choices

- The history is written by the batch runner, once per recording that completes, and by
  nothing else; the window and the command line therefore share one file without either
  knowing about the other. The file sits beside the batch records: under the parent of the
  folder given for them, else under the application home, so a test that redirects the
  records redirects the history with them and never writes to the real home.
- A recording transcribed again replaces its earlier entry; paths are compared case-insensitively
  and normalised, so a drive letter written two ways is one recording. The cap of two thousand
  entries is generous for a laptop and keeps the file small.
- Failures to write the history are swallowed inside the runner: the history is a convenience
  and a run must never fail for it. A foreign or unreadable file reads as empty rather than
  raising, for the same reason.
- The export dialog renders the text, Word and subtitle formats again from the transcript
  document rather than copying the files beside the recording, so a renamed speaker or an
  applied review is carried without a separate step; the JSON documents are copied as they are.
  A copy onto itself is skipped so that exporting into the outputs folder is harmless.
- The command line's `export` keeps its default (text, Word, SRT beside the document) and its
  message; `--out` and `--formats` add the dialog's choices. WebVTT is offered because the
  subtitle renderer already produced it and a browser wants it.
- The history dialog offers the entry's outputs folder and the export dialog for one selected
  entry; several selected entries can be added to the library or removed at once. Entries
  whose transcript document has gone are shown muted with a note, and cannot be exported.

## Observed

- A run from the command line into a chosen output folder appended one entry with the
  recording's name, duration, level, mark and speaker counts and the six output paths; the
  dialog opened on that file showed the row, its folder, and had add, show outputs and export
  enabled for it. The offscreen tests cover the window's own batch appending an entry and the
  dialog handing a recording back to the library.

## Not done, and why

- The history does not record failed runs; the batch records do, and a failure is not a
  transcription. It does not record the outputs' digests either; the run record beside the
  outputs carries the input's digest, which is the one that matters for provenance.
