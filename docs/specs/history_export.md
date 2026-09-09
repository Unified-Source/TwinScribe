# Specification: history and export

Read `CONVENTIONS.md` first, then `batch_app.md` sections 5 and 6.

Deliver `twinscribe/history.py`, `twinscribe/outputs/export.py`, `twinscribe/app/history_dialog.py`,
`twinscribe/app/export_dialog.py`, the hooks in `twinscribe/pipeline.py`, `twinscribe/cli.py` and
`twinscribe/app/main.py`, and `tests/test_history.py`, `tests/test_export.py`,
`tests/test_app_dialogs.py`.

## 1. History (`twinscribe/history.py`)

One JSON file, `history.json` under the application home (schema `twinscribe.history.v1`),
listing every recording transcribed on this machine, newest first: the recording's path and
name, its digest, duration, quality level, mark count and speaker count, when it was produced,
the six output paths, and the models that published and checked it. The batch runner appends
an entry after every recording it completes, so the command line and the window share one
history and nothing else writes it. A recording transcribed again replaces its earlier entry.
The file is capped at two thousand entries. An unreadable or foreign file reads as empty.
Removal of chosen entries and clearing leave the recordings and their outputs untouched.

## 2. Export (`twinscribe/outputs/export.py`)

`export_outputs(doc, destination, stem, formats, author, sources)` writes the chosen formats
for one transcript document into a folder as `<stem><suffix>`: the plain text, the Word
document and the subtitles (SRT and WebVTT) rendered again from the document, so a renamed
speaker or a listener's edit is carried; the transcript document, the review list and the run
record copied from where they are, the transcript document written from `doc` when it has no
source, the other two skipped when absent. A copy onto itself is not made. Unknown formats and
an empty name are errors. `transcript_text(doc)` is the transcript lines alone, for the
clipboard.

The command line's `export` command gains `--out FOLDER` and `--formats` so the same choices
are available without the window.

## 3. The window

- A History button in the top bar opens the history dialog: a table (when, recording,
  duration, level, marks, speakers, outputs folder), newest first, entries whose outputs are
  gone shown muted and marked; actions on the selection: add to the library (recordings that
  still exist), show the outputs folder, export (one entry, through the export dialog), remove
  from history, clear history after a confirmation. A double-click adds to the library.
- An Export button beside Show outputs opens the export dialog for the current recording:
  check boxes per format with the text, Word and SRT on by default and the review list and run
  record offered only when present, the folder (default: where the outputs are), the file
  name (default: the outputs' stem), Copy transcript text, Export; the dialog stays open and
  reports the files written.

## 4. Tests

The history round trip, replacement of an entry for the same path under another spelling, the
cap, removal and clearing, tolerance of foreign files; every export format with its content
checked (the Word author, the WebVTT header, a copied review list, the self-copy guard); the
dialogs offscreen: default selections, disabled formats, the export result and its status, the
clipboard, the history table's columns and muted rows, the add signal, removal.
