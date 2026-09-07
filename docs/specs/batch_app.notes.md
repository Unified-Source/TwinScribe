# Notes: the batch application

Delivered: `twinscribe/paths.py`, `twinscribe/models.py`, `twinscribe/profiles.py`,
`twinscribe/labelling.py`, `twinscribe/outputs/` (transcript document, plain text, subtitles,
Word), `twinscribe/pipeline.py`, `twinscribe/cli.py`, `twinscribe/__main__.py`,
`twinscribe/app/main.py`, `library.py`, `transcript_view.py`, `player.py`, `worker.py`,
`settings.py`, `icons.py`, `app/__main__.py`; `twinscribe/app/theme.py` and `timeline.py`
extended; tests `tests/test_models.py`, `test_profiles.py`, `test_labelling.py`,
`test_outputs.py`, `test_pipeline.py`, `test_cli.py`, `test_app_main.py` with shared fixtures in
`tests/_fixtures.py`. Two tools outside the package: `tools/fetch_models.py` and
`tools/build_portable.py`, described in `docs/Portable_Layout.md`.

## Changes to earlier modules

1. `engines/presets.py` gained the Whisper preset `quick`: production with greedy decoding
   and word times on. The benchmark preset has word times off and so cannot feed the review
   list; a fast level needed a preset that can.
2. `engines/whisper_ct2.transcribe()` and `engines/parakeet.transcribe()` accept an optional
   `progress` callback (fraction of the audio reached; the transducer reports at most a few
   hundred times per file and once with 1.0 after the flush). An exception raised inside the
   callback propagates, which is how the application cancels between engine steps. Default
   None; nothing else about the wrappers changed.
3. `app/theme.py`: the stylesheet is applied by `apply_styles()`, separate from
   `apply_theme()`. Setting a stylesheet wraps the style in a proxy whose object name is no
   longer "fusion", which the verification-screen test checks; the verification screen run on
   its own therefore has the palette but not the stylesheet, and inherits it when opened from
   the application window. `Theme` gained `muted` and the eight `speakers` colours.
4. `app/timeline.py` draws a waveform overview when one is set (`set_peaks`), draws marks
   translucent over it so the waveform shows through, and can hide its time labels. Without an
   overview it renders as before.
5. `pipeline.run_batch()` reports each outcome as it lands through `on_outcome`, so the
   worker thread can update a row before the batch ends.

## Deviations and interpretations

- The specification counts six outputs per recording. The speaker summary the design lists is
  the head of the text and Word documents rather than a seventh file; the sixth file is the
  transcript document, which the application reads back and every renderer works from.
- Outputs go beside the recording by default because that is where players look for a
  subtitle file, and because the review set can then name the audio by a relative path. Two
  recordings with the same stem in one folder would overwrite each other's outputs; not
  guarded, recorded here.
- The run record's `input_path` is the path as given, which may be absolute; the record is a
  local artifact of the operator. The transcript document carries only the file name.
- The transcript document never carries the detector's text; the review set does, as the hint
  the verification screen shows. The gap markers in the pane show only the word count.
- The Word document's author is the settings' author, and the application name when that is
  empty; the extended properties name only this application.
- Subtitle cues carry the speaker's name on every cue rather than only at a change of
  speaker, so a viewer who joins mid-way knows who is speaking. Cues under one second are
  lengthened up to the next cue.
- The decoded 16 kHz work file lives under the application home and is removed after the run
  unless `keep_audio` is set; a compliant 16 kHz mono WAV is fed as it is.
- The batch record goes under the application home (`runs/`) rather than beside the
  recordings, so a folder of recordings is not littered with one record per batch.
- `discover_media()` decides by extension; a file with a wrong extension is reported by ffmpeg
  as a failure of that file and the batch goes on.
- The application home is `TWINSCRIBE_HOME`, else the per-user application data folder, else
  a hidden folder under the home directory. Settings are one JSON file there; nothing is
  written to the registry.
- The library persists as the list of recording paths in the settings and is restored on
  start, existing files only.
- A double-click on a speaker chip renames the speaker; the rename rewrites the transcript
  document and the text, Word and subtitle files, not the review set or the run record, which
  describe the run as it happened.
- Clicking inside the words of a line selects text rather than seeking, so a passage can be
  copied; the time gutter and a double-click seek. Gap markers seek and play on a single click.
- The worker checks for the engine libraries before starting only when it will use the real
  engines; with injected engines (tests) the check is skipped.

## Not exercised here

Neither engine library nor any model was present on the development machine, so:

- the pipeline ran only with synthetic engines; the real engine calls run as far as their
  input checks (as recorded in `engines.notes.md`); the `progress` callbacks in the two
  wrappers are written but were not run inside a live decode;
- `tools/fetch_models.py` was run for the two smallest catalogue entries only (the voice
  detector, a single file, and the segmentation model, an archive member), into a scratch
  folder, to exercise both download paths and the lock; the large models were not fetched;
- `tools/build_portable.py` was run in `--dry-run` form only; no embeddable interpreter or
  wheel set was downloaded and no portable folder was assembled;
- the Word document was checked as well-formed OOXML and read back through a Word library in
  the tests; it was not opened in Word itself on this machine.

## The platform, observed

Windows 11, 64-bit ARM, PySide6 6.11.2 with the FFmpeg multimedia backend.

- Screenshots on the real backend rendered every glyph; the window was checked in both
  palettes, at rest and during playback with a synthetic recording of shaped noise and a
  transcript document built from the fixtures. During playback the playhead advanced, the
  line under it was highlighted and kept in view, and the transport showed the pause glyph.
- `QMediaPlayer.setSource()` on the real backend prints a stream description and two encoder
  probe lines to stderr, as `verify_app.notes.md` records; harmless.
- Under the offscreen platform a `QMediaPlayer` with a real WAV as source, a `QVideoWidget`
  as video output and an `QAudioOutput` all construct and load silently; the tests exercise
  seeking and position tracking without starting playback.
- A `QTextEdit` extra selection with the full-width property highlights the whole line
  including the margin; the block background of the gap markers is set on the block format,
  which paints the full width too. Both read correctly in both palettes.
- Setting a stylesheet changes `QApplication.style().objectName()`; see change 3 above.
