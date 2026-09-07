# Specification: the batch application

Read `CONVENTIONS.md` first, then sections 1, 3, 4, 9 and 11 (item 4) of
`../Design_and_Findings_2026-09-06.md`, `engines.md` for the engine wrappers and their records,
`review.md` for the review list and the run record, and `verify_app.md` for the verification
screen this application opens.

Deliver the model store and quality levels (`twinscribe/models.py`, `twinscribe/profiles.py`),
speaker attribution (`twinscribe/labelling.py`), the output renderers (`twinscribe/outputs/`),
the per-file pipeline and batch driver (`twinscribe/pipeline.py`), the command line
(`twinscribe/cli.py`, the `twinscribe` script the packaging already declares), the application
window (`twinscribe/app/main.py` with `library.py`, `transcript_view.py`, `player.py`,
`worker.py`, `settings.py`, `icons.py`), the folders module (`twinscribe/paths.py`), and tests
under `tests/`. The engine libraries are not installed in the development environment; the
pipeline takes its three engine calls as an injectable record so that every path above the
engines runs with synthetic engines.

## 1. Purpose

One window that behaves like a media player: recordings and folders are dropped in, each plays
at once, and the ones that have been transcribed show their transcript following the audio.
Choosing a quality level and pressing one button runs the two-engine pipeline over everything
not yet done and writes, beside each recording, the outputs the design names. Nothing in the
package reaches the network; models load from a local folder.

## 2. Model store and quality levels

- A models root holds one folder per model. The catalogue names each folder, the files it must
  contain, the licence, the credit and the upstream source (a fetch tool outside the package
  uses the sources). Roles: publisher, detector, voice detector, segmentation, embedding.
- `find_models(root)` reports which folders are complete; `verify_store(root)` digests every
  required file against a lock file written when the files were fetched, so a store is
  checked against what was downloaded rather than trusted.
- The root is the environment variable `TWINSCRIBE_MODELS`, else a `models` folder beside the
  package, else one under the application home.
- Three quality levels: standard (the measured configuration: Parakeet TDT 0.6B v2 published,
  Whisper large-v3-turbo checking at the production preset), quick (Distil-Whisper large-v3
  checking with greedy decoding, word times kept on because the review list needs them), and
  careful (full Whisper large-v3 checking). A level is offered only when every model it needs
  is present. All three share the publisher, the voice detector, the speaker models and the
  review thresholds of the design (0.8 s, two words, 0.4 s run-in).

## 3. Speaker attribution and lines

Every published word takes the label of the diarization turn it overlaps most; a word no turn
overlaps takes the nearest turn within one second; anything still unlabelled inherits the
previous label, or the next at the start. Consecutive words of one speaker form a line; a line
breaks at a pause over 1.5 s or at sixty words. The speaker summary counts words and seconds per
label in order of first appearance, with unlabelled words last. Display names default to
Speaker 1, Speaker 2, ... and can be changed.

## 4. Outputs per recording

Beside the recording (or in one chosen folder), named by the recording's stem, or by its full
file name when another recording with the same stem sits beside it (`call.mp3` beside
`call.wav`), so that two recordings never overwrite each other's outputs:

1. `<stem>.transcript.json` (schema `twinscribe.transcript.v1`): source name, digest and size;
   duration; engines; speakers with names, word and second counts; lines with their words and
   times; the review marks without the detector's text; a waveform overview for the timeline;
   the review count and share. Every other renderer is a pure function of this document.
2. `<stem>.txt`: header (file, duration, engines, speaker summary, review pointer, the draft
   notice), then `[m:ss.t] Name: words` per line, a blank line between speakers.
3. `<stem>.docx`: the same content as a Word document written with the standard library only
   (zip of OOXML parts): title, engine facts, a speaker table, one paragraph per line with a
   hanging indent, the review note. The document properties carry the author from the
   settings and this application's name; no other tool is named.
4. `<stem>.srt`: cues cut at word boundaries (at most 84 characters or 7 seconds), every cue
   prefixed with the speaker's name, so that any player loads the transcript beside the media.
5. `<stem>.review.json`: the review set of `review.md`, with the audio path relative when the
   review set sits beside the recording and absolute otherwise.
6. `<stem>.run.json`: the run record of `review.md`, with both engines, the speaker models,
   versions, the effective presets and thresholds, the digest, the timings and the load
   verdict. A failed run leaves a run record naming the failure. A batch record under the
   application home lists every recording with its outputs or its error.

Speaker labelling that fails does not lose the transcript: the failure is recorded and the
words stay unlabelled.

## 5. Pipeline and batch

Per recording: digest; decode to 16 kHz mono through ffmpeg unless the file already is; run the
publisher; run the detector; label speakers; build the review list; write the outputs. Progress
is reported as an overall fraction with the stage named, using the engines' progress callbacks;
a cancel check stops within one engine progress step. A batch runs recordings one after
another and never stops for a failure. `discover_media(paths)` lists the audio and video files
under files and folders, recursively.

## 5a. The machine and the plan

`twinscribe/hardware.py` probes the machine (operating system, architecture, cores, memory,
NVIDIA devices, which engine libraries import) and chooses a plan: the detector backend
(CTranslate2 where it imports, else Whisper through sherpa-onnx), the device and compute type
or provider of each engine, the thread count, and whether the two transcription engines run
at the same time (only when they sit on different devices). A preference of auto, cpu or cuda
constrains it. The plan is chosen once per batch, recorded in every run record, printed by
`twinscribe check` and shown in the settings. Each level lists its detector candidates in
order of preference (the CTranslate2 conversion, then the ONNX export), and a level is offered
when its models are present and one candidate can run with the libraries installed.

`twinscribe/engines/whisper_onnx.py` is the second detector: windows of at most thirty
seconds cut at the quietest moment before each limit (never the voice detector's utterances,
which the transducer decodes), token timestamps when the export carries attention outputs,
otherwise words spread inside each segment's span and confined to its voiced parts, with the
transcript recording that its word times are approximate.

## 6. The window, top to bottom

1. Top bar: the name, Open files, Open folder, the quality level, Transcribe (primary), Stop
   while a batch runs, settings.
2. A horizontal splitter:
   - left, the library: one row per recording with a state glyph (not transcribed, queued,
     running with a progress ring and bar, done with a tick, failed with a warning), its name,
     its folder or its state, its duration; an empty-state hint when empty;
   - right, the recording: name and a meta line (duration, speakers, level, review count);
     speaker chips with word counts (double-click renames, which rewrites the text, Word and
     subtitle files); a video pane shown only when the file has video; the transcript pane,
     or, while the recording is queued or being worked on, the job card in its place: a stage
     strip with the current stage lit and the finished ones ticked, a progress bar and
     percentage, the message of the moment (loading an engine, transcribing, checking,
     labelling speakers, writing), the time elapsed, a hedged estimate of the time left from
     the pace so far, a heartbeat that keeps moving between engine reports with the time since
     the last report, the acceleration plan in use, its place in the batch, and the lines the
     published engine has produced so far (provisional, without speaker labels; the detector's
     text never appears); the player bar with the timeline (waveform overview, review marks,
     playhead), back and forward five seconds, play or pause, the time readout, follow, speed
     and volume.
3. Status bar: the last message on the left, the batch state on the right.

## 7. Behaviour

- Dropping files or folders, Open files and Open folder add recordings; a recording whose
  transcript document sits beside it is shown as done at once.
- Selecting a recording loads it into one QMediaPlayer; playback works whether or not it has
  been transcribed. With a transcript, the line under the playhead is highlighted and kept in
  view (a manual scroll pauses following for a few seconds; the follow button toggles it).
- Gap markers between the lines name the spans the review list flagged, with the count of
  words the second engine heard there and never their text; clicking one plays the span and
  pauses at its end. A resolution from a review session is shown under its marker.
- Clicking a line's time seeks to it; double-clicking a line seeks to it; clicking the
  timeline seeks.
- Keys: Space play or pause; Left and Right nudge five seconds; J and K play the next and
  previous review span; F toggles following; Delete removes the selected recordings; Ctrl+O
  and Ctrl+Shift+O open files and a folder.
- Transcribe queues every recording not yet done, runs the batch in a thread and updates each
  row as it goes; the current recording shows the job card while it waits and while it runs,
  the window title carries the percentage, and the transcript replaces the card when the
  outputs land; the taskbar entry is flashed when a batch finishes while the window is not
  active. Before starting, the engine libraries and a complete model set are checked and a
  plain message names what is missing. Stop cancels after the current step; recordings not
  reached go back to not transcribed.
- Engines report once when their model has loaded, so the card can tell loading from
  decoding, and hand each segment to the card as it is produced; the command line shows one
  status line updating in place on a terminal (stage, percentage, elapsed, time left), and
  prints on stage changes and every ten per cent elsewhere.
- Review opens the verification screen of `verify_app.md` on the recording's review set and
  reloads the pane when it closes. Show outputs opens the folder that holds the outputs.
- Settings: models folder (with a report of what it holds), outputs beside each recording or
  in one folder, author, threads, acceleration (automatic, processor only, CUDA device) with
  the plan the choice yields, light or dark. Settings, volume, speed, follow, the library and
  the window size persist in one JSON file under the application home, never the registry.

## 8. Theme

The palette and Fusion style of `verify_app.md`, plus one stylesheet built from the same table
(radii, borders, spacing), a set of eight speaker colours per palette, painted icons and no
external asset of any kind.

## 9. Command line

`twinscribe app [paths] [--models DIR] [--dark] [--shot PNG]`, `twinscribe run <paths> [--models]
[--out] [--quality] [--threads] [--author] [--keep-audio] [--no-recurse]`, `twinscribe check
[--verify]`, `twinscribe export <transcript.json>` (render text, Word and subtitles again), and
`twinscribe verify <review.json>`.

## 10. Tests

Under `tests/`, with synthetic engines and a one-byte stand-in for every model file: the
catalogue is consistent and the store reader finds, misses and verifies; levels follow the
store; attribution and lines by hand-derived cases; the document, text, subtitle and Word
renderers (Word parts well-formed, read back by a Word library when one is present); the
pipeline writes the six outputs, records a speaker failure without losing the transcript,
leaves a run record on failure, cancels, decodes non-compliant audio through ffmpeg (skipped
when absent), and the batch records every outcome; the command line; and, under the offscreen
platform, the window: library states, the transcript pane following a position, keys, gap
markers, renaming, a whole batch through the worker thread, drops, and settings persistence.

## 11. Notes file

`docs/specs/batch_app.notes.md`: deviations, what could not be exercised without the engine
libraries and models, and what QtMultimedia and the platform did that the specification did
not anticipate.
