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
  subtitle file, and because the review set can then name the audio by a relative path. When
  another recording with the same stem sits in the same folder, the outputs take the full file
  name as their base (`call.mp3.txt` beside `call.wav.txt`); the transcript document records
  the base it was written under so the text and Word pointers to the review list stay right.
  A same-stem pair spread over two folders with one shared output folder is not guarded.
- Subtitle lines longer than a cue are halved at the word boundary nearest the middle, and
  the halves again until every chunk fits, rather than filled greedily, so that a long line
  yields chunks of similar size and never a one-word tail.
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

## Exercised on the development machine, after the first delivery

The first delivery ran only with synthetic engines. Afterwards the engine libraries were
installed: sherpa-onnx natively on the 64-bit ARM interpreter, and the whole engines extra
(faster-whisper, CTranslate2, sherpa-onnx) in a second, x64 environment that runs under the
platform's emulation, because CTranslate2 publishes no wheel for the native architecture. The
standard-profile models were fetched into `models/` beside the package with
`tools/fetch_models.py` and every file verified against the lock afterwards.

- The three live engine tests pass: the transducer and the diarizer natively and under
  emulation, Whisper under emulation.
- `twinscribe run` at the standard level processed a synthesised two-voice conversation (two
  of the operating system's speech-synthesiser voices reading a 28-line script, 143 seconds)
  in WAV, MP3 and MP4 form (the MP4 carrying a blank video track). Every output was written
  for all three, the two speakers were found and labelled correctly throughout, and the four
  or five review marks per file landed on phrases the published engine had dropped ("Is it"
  before "the same lane", "It does." before "The booking is for one hour"). The figures are
  in `engines.notes.md`. Synthesised speech says nothing about accuracy on recordings; it
  shows that the chain works end to end.
- The window opened the produced documents: the transcript followed the audio, the video file
  showed its video pane, the review screen opened on the review set.
- Two defects surfaced and are fixed here: the fixture's `.mp3` and `.wav` share a stem and
  overwrote each other's outputs (the naming rule above); and greedy cue splitting left a
  one-word trailing cue on lines just over the limit (the balanced splitting above).

## Review for other machines, and the acceleration plan

A pass over the delivered code for machines other than the one it was built on found and
fixed: child processes (ffmpeg) opening a console window when the window is started without
one on Windows; the application home ignoring the conventions of macOS and of the XDG data
folder; media discovery lowering case on file systems that distinguish it; no processor,
memory, load or power probes on macOS; no font fallbacks beyond the one platform; ffmpeg
looked for on the search path only. Each is in `hardware.py`, `audio.py`, `paths.py`,
`pipeline.py`, `runrecord.py`, `load.py` or the theme; the macOS probes are written from the
documented interfaces and not run.

The plan (`hardware.py`) and the second detector (`engines/whisper_onnx.py`) are described
in `batch_app.md` section 5a and in `docs/Platforms.md`. Measured on this machine, natively,
with the ONNX turbo export on the synthesised conversation:

- decoding on the voice detector's utterances, as the transducer does, made the two engines
  fail in the same places and the review list found nothing (zero marks);
- decoding in thirty-second windows cut at quiet moments gave a transcript as good as the
  CTranslate2 conversion's (normalised WER 1.8 against 1.5 per hundred words, no deletion),
  but spreading words over whole segment spans put words into pauses: 34 marks, 43 per cent
  of the audio, almost all false;
- confining the spread words to the voiced parts of each span gave 3 marks, all three on real
  dropped phrases and no false alarm, against the 5 of the CTranslate2 detector; the two
  missed were two-word gaps, the cost of approximate word times.

The published exports carry no cross-attention outputs; an export made with the library's
attention-keeping script would give token timestamps, which the detector uses when present.
Not run here: a CUDA device (none on this machine), the CUDA build of sherpa-onnx, the CUDA
runtime packages, the parallel-engine path with real engines (exercised with synthetic engines
in the tests), Linux and macOS.

## The job card

Before it, the detail pane showed the not-transcribed placeholder for a recording that was
being worked on, engine reports could be seconds apart, and loading a model or labelling
speakers reported nothing. Now:

- Every progress report carries the seconds elapsed and an estimate of the seconds left,
  computed per file from the pace so far (elapsed times the fraction remaining over the
  fraction done), smoothed by an exponential average, and withheld until five per cent is
  done. The card words it with hedges ("roughly 3 min left") because the pace of the second
  engine differs from the first's.
- The three engine wrappers call the progress callback once with 0.0 after their model has
  loaded, so the pipeline's "Loading the published engine" message gives way to
  "Transcribing" at the right moment; speaker labelling reports nothing inside the library
  call, so its message says so and the heartbeat carries it.
- The heartbeat is a dot that alternates colour on a timer while the card is running,
  beside the time since the last engine report once that exceeds a few seconds, so a long
  silence reads as "still working" rather than as a hang.
- The published engine's segments reach the card as they are decoded and are shown as
  provisional lines with their start times and no speaker labels. The detector's segments
  are not passed on at all; the worker forwards the publisher's only.
- The command line got the same information as one line updating in place on a terminal,
  and as a line per stage change or ten per cent step when the output is not a terminal.

## Not exercised here

- The quick and careful levels: their detectors are not fetched on this machine.
- Speed. Under emulation the pipeline took about as long as the audio for a 143 second file
  with the model loads included, and the run records flagged other load on the machine. The
  design record rules out quoting any throughput figure from this machine; the lab hardware
  measures speed.
- `tools/build_portable.py` was run in `--dry-run` form only; no portable folder was assembled.
- The Word document was checked as well-formed OOXML and read back through a Word library; it
  was not opened in Word itself on this machine.

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
- A window grab (`QWidget.grab()`) does not capture the frames the video pane paints; the
  pane shows as a blank area in a screenshot while the frames are visible on screen.
- Two interpreters share one checkout on this machine: the native one runs the window and
  the transducer, the emulated x64 one runs the whole pipeline. The window's Transcribe
  button therefore reports the Whisper library as not installed in the native environment;
  the pipeline was driven from the command line of the x64 environment instead. On the lab
  hardware, where every wheel is native, one environment serves both.

## Reading the provisional lines while they are written

The first form of the job card appended every line to a plain text pane and moved its cursor
to the end, so the pane was pulled to the newest line whatever the reader was looking at. The
lines are now items of a list widget, which gives each line a click target and a background of
its own.

- Following is a state, not a timer: the list follows the newest line only while the reader
  is at the bottom. Scrolling up (any amount) or clicking a line holds the view; scrolling
  back to the bottom, or the button, follows again. The finished transcript pane keeps its
  own rule (a manual scroll pauses following for a few seconds) because there the audio, not
  an engine, drives the view; the two rules were kept apart on purpose.
- The scroll bar's valueChanged signal is the only way the widget reports a scroll, and the
  card's own scrolls fire it too; a flag set around every programmatic scroll tells the two
  apart. Adding an item does not move the value, so lines arriving while the view is held
  leave it where it was; they are counted and the button back to the latest line says how
  many.
- `scrollToBottom()` runs any pending layout before it reads the maximum, so a scroll issued
  right after `addItem()` lands on the new item; the offscreen tests rely on this.
- The list takes no keyboard focus, so Space, the arrows and the other keys of the window keep
  their meaning while the reader clicks in the list. A click seeks, a double-click seeks and
  plays; the second click of a double-click also arrives as a click, which is harmless because
  both seek to the same time.
- The line under the playhead is highlighted with the same tint as the finished pane uses and,
  while the view is held and the follow toggle is on, is kept in view; while the list follows
  the newest line, the highlight moves without scrolling, since the reader asked to see the
  newest lines rather than the audio.
- Two defects surfaced when a line was clicked just after a recording was selected. The
  multimedia backend answers a seek issued while the media is still loading by keeping the
  position at zero, so the window now holds such a seek and applies it when the status turns
  to loaded (position reports from the loading source are ignored meanwhile). And the
  timeline clamped the playhead to a duration it did not know yet (zero until the player
  reports it), which flattened the same click to zero on the bar while the time readout showed
  the right value; the clamp now applies only once a duration is known.

## Non-speech scenes

A report that silence appeared to be filled in. Measured on a synthesised fixture (110 s, two
voices, three silences of 3 to 6 s, two music passages of 9 and 12 s, one noise passage of 8 s,
two lines spoken over faint music), with the native engines: the published engine wrote nothing
inside any passage, because the voice detector passed neither the synthetic music nor the
noise, so the words "inside" a passage in a naive count were the last word of the utterance
before it, whose end the transducer places at the utterance's end. The second engine
hallucinated one word inside the music and one inside the noise, and the review list carried
one false mark on the music passage, which is the reported failure: the review list, not the
transcript, was filling silence, and with the detector's approximate word times a real word
after the music was spread into it as well.

- With the scene pass all six passages are found with the right kind (silence by level, music
  and background noise by the tagger), the transcript carries a marker for each, the review
  list has no mark, one detector word is set aside, and both lines over faint music are kept
  with their speaker labels.
- The utterance rule (words set aside when the tagger is confident an utterance holds no
  speech) was not exercised by the fixture, since no utterance was non-speech; it is covered
  by the unit tests and guarded three ways: never on level alone, never under one and a half
  seconds, and only with speech below 0.15 and music or noise at 0.5 or more. The design's
  rule for silence stripping (log every removed region) is met by listing every utterance set
  aside in the run record with its text.
- The tagger is optional. Without its model the pass still marks silence by level (-50 dBFS)
  and calls a loud pause sound of an unknown kind; the detector's words inside those are left
  out of the review list too.
- The stage weights moved to make room for the pass (publisher 0.32, detector 0.38, scenes
  0.03); the job card's strip gained a "Non-speech" step.

## Speaker labels: what could be improved here, and what could not

A request to improve the accuracy of the speaker labels. What this machine can measure is
limited: the synthesised fixtures use two of the operating system's voices, which the
embeddings separate almost perfectly (DER 1.7 per cent on the scene fixture, all of it missed
speech at turn edges, no confusion), so no change to the clustering can be shown to help or
hurt here. The design record's finding stands: labelling is the weakest part of the system,
its worst case (quiet and overlapped speakers) is structural, and the honest measurement needs
the public corpora on the lab hardware. What was done is therefore what can be shown correct
without a corpus:

- The speaker count as an explicit opt-in. With the count known, the design measured a large
  gain, and also the failure it hides (a degenerate cluster satisfying the count while two real
  speakers are absorbed elsewhere); so the default stays the threshold, the control says why,
  and a fixed count is written into the run record, the document and the text header with the
  caution beside it.
- Flicker smoothing inside utterances. A turn boundary from the segmentation model jitters
  against the transducer's word boundaries by a fraction of a second; a single short word in
  the middle of a sentence then carries the other speaker's label. Such a word now takes the
  label of its neighbours when both agree; edges and multi-word runs are untouched, so a real
  interjection is never absorbed. The count relabelled is in the run record, so a recording
  where the rule fires often is visible.
- Contiguous labels: the library skips cluster numbers it discards (speaker_00 and speaker_03
  for two speakers on the fixture); labels are renumbered by first appearance.

Not done, and why: swapping the embedding model (the section 6 candidates) or tuning the
clustering threshold without a labelled corpus would be tuning on synthetic voices, which the
design forbids; both wait for the lab hardware and the corpus pipeline, where per-speaker
recall and the Jaccard error rate can be scored.
