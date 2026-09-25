# Notes: a recording in parts

Companion to `recording_parts.md`. Records what was built, what was checked, and what was left.

## Built and checked

- `audio.avi_facts` walks the RIFF header up to the movie list and reads the recorder's
  `TUTC` start (a FILETIME, to the microsecond) and the duration from the audio stream
  header (length times scale over rate), falling back to the frame count and frame time.
  Checked on a header written in the test, on a header without an audio stream, on one
  without a start, on a WAV (RIFF, not AVI) and on a file that is no container. On the
  session that called for this, the audio header gave 300.05 s where the frame count gave
  299.6 s, and the probe of the bundled decoder agreed with the former.
- `pipeline.group_parts` chains timed files of one extension in one folder by their header
  times: an overlap of up to two seconds (the recorder's rollover measured at 0.0 to 0.3 s)
  and a pause of up to fifteen minutes join; a longer gap, another extension or a file
  without a start stands alone. `discover_recordings` reads headers only in a folder holding
  at least two files of one extension, so a folder of single recordings costs nothing more.
  Checked on made-up facts: three files with a pause of 145 s and a rollover of 0.2 s
  chained with their offsets, a fourth twenty minutes on split off, a file without a time and
  a file of another kind apart; an overlap of 1.5 s chained, one of 8.5 s split.
- `audio.join_wavs` streams the pieces into one WAV at their offsets, a second of silence at
  a time for a gap, and cuts a piece short where the next begins earlier. Checked on three
  tones with a gap and an overlap, sample for sample.
- The pipeline decodes the parts one by one, joins them, and runs as for one file; the
  document's `source` carries `parts` (name, digest, size, offset, duration), its digest is
  the digest of the parts' digests and its size their sum; the review set carries `parts`
  with each audio reference and offset; the run record carries `parts` and names the first
  file. The joined audio is moved to the recording's playable copy under the application
  home instead of being deleted with the work files, and the pieces are removed. Checked on
  two synthetic parts with a five-second gap: the records, the copy of eight seconds, no
  piece left behind.
- The library shows one entry per recording, named by its first file, its second line
  giving the number of parts; the document beside the first file is matched by the first
  part's size, not the summed size; the batch hands the parts to the pipeline; the window
  and the verification screen hand them to the playable copy, which joins them as the
  pipeline does. The command line says which recordings are in parts and offers `--no-join`.
  Checked in the tests with a stand-in for the copy's thread.
- The session that called for this, read locally: eleven `.trm` files of a court recording
  system, 51.2 minutes of audio, a pause of 145 s after the first file and rollovers of 0.0
  to 0.3 s between the rest. With the extension read (the previous change) and no joining,
  it gave eleven transcripts; with joining, one recording of 3,220 s, the parts at 0, 255.3,
  555.5, 855.6 s and so on from the headers, each part's decoded duration within 0.3 s of
  the header's, the pause a silence scene of 145 s in the transcript, 89 review marks, no
  failures, in 1,222 s on the development laptop (the ONNX detector on the processor). The
  speaker stage gave 21 labels for a room of a few voices, the known limit of the clustering
  on a long recording and the reason the speaker count can be given; nothing in the join
  changes that.

## Choices

- The header time, not the file name, decides what follows what. Names carry a time only by
  convention and a sequence number only sometimes; the recorder's own start time in each
  header is exact, and a pause between files is then a pause in the joined audio rather than
  a splice, so the transcript's times agree with the clock on the wall.
- Fifteen minutes as the longest pause joined. A recorder stopped and started again within
  a session, a break, is one recording; two matters heard in one room an hour apart are two.
  The number is a guess at the boundary and is a constant, not a setting, until a case shows
  it wrong; `--no-join` is the way round it on the command line.
- The joined audio is kept, in the engines' own format, rather than rebuilt for every
  playback: fifty minutes of 16 kHz mono is about a hundred megabytes and decodes in under a
  minute, and the copy is what the transcript was made from. The verification screen can
  rebuild it from the review set's `parts` when the copy is gone.
- The recording is named by its first file, so the outputs sit where a single file's would
  and every existing rule about names and collisions holds; the document says which files
  it covers.

## Not done, and why

- Only the AVI header is read for a start time. Containers that carry a creation time as
  metadata are not chained, because for some recorders that time is when the file was
  closed, not opened, and a chain built on it would insert silences that were never there;
  a second source of start times wants a recorder to check it against.
- The window has no switch for joining; a recording in parts is always one entry. The
  command line has `--no-join`.
- The playable copies accumulate under the application home and are never removed by the
  program.
- The `.trs` index beside a court recorder's files, which names the session's files and
  their times, is not read; the headers suffice for the join, and the index's labels are a
  possible source of the room's name for a later change.
