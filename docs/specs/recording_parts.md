# Specification: a recording in parts

Read `CONVENTIONS.md` first, then `batch_app.md` (discovery, the pipeline, the window) and
`verify_app.md` (playback). Deliver the additions to `twinscribe/audio.py`,
`twinscribe/pipeline.py`, `twinscribe/paths.py`, `twinscribe/review.py`,
`twinscribe/outputs/transcript_doc.py`, `twinscribe/runrecord.py`, `twinscribe/cli.py`,
`twinscribe/app/library.py`, `twinscribe/app/worker.py`, `twinscribe/app/main.py`,
`twinscribe/app/verify.py` and `twinscribe/app/playable.py`, with `tests/test_parts.py` and
the additions to the existing tests, and `recording_parts.notes.md`.

## 1. The problem

Court and interview recording systems write a session as consecutive files, one every few
minutes, each file's header carrying the moment the recorder started it. Transcribed file by
file, a session of fifty minutes gives eleven transcripts whose times restart at zero, eleven
review lists, and no place where the whole is read or played. The recording is the session;
the files are its parts.

## 2. What a part is, and what joins them

A part is a file whose header says when it started and how long it runs. So far the only
header read is the AVI header: `audio.avi_facts(path)` returns the start (a UTC FILETIME in
the recorder's `TUTC` chunk) and the duration (the audio stream's length over its rate, else
the frame count times the frame time), or nothing for a file that is not an AVI or carries no
start time. `pipeline.recording_facts(path)` applies it to `.avi` and `.trm` files; every
other file has no facts and stands alone.

`pipeline.group_parts(files)` chains the timed files of one folder that share an extension:
sorted by start, a file follows the previous when it begins where the previous ended, with an
overlap of up to `JOIN_TOLERANCE_S` (two seconds, the rollover of a recorder) or a gap of up to
`MAX_JOIN_GAP_S` (fifteen minutes, a pause with the recorder stopped and started again). A
longer gap starts another recording. A chain of one is a recording on its own. Each chain
becomes a `Recording(source, parts)`: `source` is the first file, `parts` the `Parts` record
of every file in order with its offset in seconds from the first file's start, taken from
the header times, so a pause between two files stays a pause in the joined recording.
`discover_recordings(paths, recursive, join)` lists recordings this way under files and
folders; with `join` false every file is a recording of its own, which the command line
offers as `--no-join`. `discover_media` keeps listing the files.

## 3. The joined recording

The pipeline takes the parts on the job (`Job.parts`). Each part is decoded to the engines'
format as a single file is, and `audio.join_wavs(pieces, target)` writes them into one WAV,
each piece at its offset: silence fills a gap, and a piece that begins before the previous
one ends cuts the previous one short at its own start. From there the run is the run of one
recording: one transcript, one review list, times from the first file's start. The recording
is named by its first file, so the outputs sit beside it under that file's name, and
`output_paths` needs no change.

The joined audio is kept: no single file holds the whole recording, so the window and the
verification screen need it to play. The pipeline moves the joined WAV to the recording's
playable copy under the application home (`paths.playable_copy_path(source)`, the same place
the window's own copies go) instead of deleting it with the work files.

The document's `source` names the first file and carries `parts`, one record per file with
its name, digest, size, offset and duration; the document's `source.sha256` is the digest of
the parts' digests joined, and `source.bytes` their sizes summed. The review set carries
`parts` too, each with its `audio` reference (a name beside the review set, or a resolved
path when the outputs went elsewhere) and its offset, so that the verification screen can
rebuild the joined audio when the copy is gone. The run record carries `parts` with the same
facts and names the first file as its input.

## 4. The window and the command line

The library shows one entry for a recording in parts, named by its first file, its second
line saying how many parts; the transcript beside the first file is its document, matched by
the first file's name and size. The batch passes the parts to the pipeline with the source.
Selecting the recording plays its copy when the copy is current; otherwise the playable copy
is made from the parts, not from the first file alone, in the window and in the verification
screen alike (`PlayableCopy(source, parts)` joins them as the pipeline does).

The command line's `run` discovers recordings the same way, says which recordings are in
parts and how many, transcribes each as one, and with `--no-join` transcribes every file on
its own.

## 5. Tests

The AVI header reader on a header written in the test, with the start and the duration
recovered, and on files that are no AVI or carry no time; the chaining rule on made-up facts:
contiguous files chained with their offsets, an overlap within the tolerance chained, a gap
past the limit split, a file without facts alone, a different extension apart; discovery with
and without joining; the WAV join on synthetic tones with a gap filled and an overlap cut; the
pipeline on two parts with a gap, the document, the review set and the run record carrying the
parts, the duration spanning the gap, the joined copy kept where the window looks; the
library showing one entry with its parts and matching its document; the command line's flag;
and the window and the verification screen handing the parts to the copy.
