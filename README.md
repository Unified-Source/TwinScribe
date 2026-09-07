# twinscribe

Offline transcription with speaker labels for long recordings, built for material that
will be relied on.

**Status: early. The design is measured; the implementation is built from it and has run end
to end with the live engines on synthesised speech. The accuracy table on public corpora and
the first throughput figures are the next milestone, on lab hardware.**

## What it is for

Take audio and video recordings on an ordinary laptop, with no network connection and no
administrator rights, and produce a timestamped, speaker-labelled transcript in plain text
and in Word, together with a record of how it was produced. The two workloads that shape
every decision are batches of twenty-minute two-party telephone recordings at narrowband
quality, and older interview-room video with one far-field microphone and several speakers.

The output will be read by people who were not present, so the governing rule is that a
transcript is a draft until it has been verified against the recording. The tool's job is to
make that verification targeted and cheap rather than a read-through.

## The design in one paragraph

Open transcription engines come in two families and they fail in opposite directions. The
Whisper family predicts each word partly from the words it has already written, so it always
produces something, and on hard or silent audio it can write fluent text that was never said.
Transducer engines such as NVIDIA Parakeet can only emit a word while consuming audio, so
they cannot write into silence, and their failure is the opposite: on hard audio they produce
nothing. twinscribe runs both. It publishes only the transducer's transcript, and it uses the
Whisper engine for one purpose, never shown to anybody: to mark every span where it heard
speech and the published engine heard nothing. Those marks are the review list, and on public
telephone recordings they landed on real missed speech 61 times out of 62, covering 92 per
cent of the words the published engine had dropped, for 18 per cent of the audio listened to.

The full design, the measurements on public corpora, the engine and licence landscape, and
the engineering traps already paid for are in
[docs/Design_and_Findings_2026-09-06.md](docs/Design_and_Findings_2026-09-06.md).

## What it produces, per recording

Beside the recording, named by its stem:

- `.docx` and `.txt`: timestamped, speaker-labelled, with a speaker summary at the top giving
  the word count assigned to each speaker, because a participant can vanish from a transcript
  while the overall accuracy looks normal;
- `.srt`: a subtitle file, so the transcript plays against the recording line by line in any
  player;
- `.review.json`: the review list, with a screen for working through it with the audio at
  each mark;
- `.run.json`: the run record: engines and versions, settings, the file's digest, when it ran,
  how long it took, and any failure and why;
- `.transcript.json`: the document the application reads back and every renderer works from.

## The application

One window, in the manner of a media player: drop recordings or folders in, each plays at
once, and the ones that have been transcribed show their transcript following the audio, the
review marks on the timeline and between the lines, and a button that opens the verification
screen. Choose a quality level, press Transcribe, and the pipeline runs over everything not
yet done with progress per file. While a recording is being transcribed its lines appear as
they are decoded; they can be read without being pulled to the newest line, clicked to move
the playhead, and played from. Nothing in the package reaches the network.

```
python -m twinscribe app [recordings or folders]
python -m twinscribe run <recordings or folders> [--quality standard] [--out FOLDER] [--device auto]
python -m twinscribe check [--verify]
python -m twinscribe export <transcript.json>
python -m twinscribe verify <review.json>
```

In a checkout with the project's virtual environment beside it, `twinscribe.cmd` runs the same
command line (`twinscribe.cmd run <folder>` transcribes with progress in the console; with no
arguments it opens the window), and `twinscribe-app.cmd` opens the window without a console.
A window started without a console writes anything it would have printed, and any unhandled
error, to `twinscribe.log` under the application home.

Models live in one folder (`TWINSCRIBE_MODELS`, or `models` beside the package);
`tools/fetch_models.py` fetches them from the sources the catalogue names and pins their
digests. A folder that runs with nothing installed is described in
[docs/Portable_Layout.md](docs/Portable_Layout.md). Which platforms and accelerators are
covered, and how the machine is probed for the best available option, is in
[docs/Platforms.md](docs/Platforms.md). Specifications and the notes recorded while building
from them are under `docs/specs/`.

## Components and licences

Engines and models are open and run on the processor: faster-whisper and CTranslate2 (MIT),
Whisper weights (MIT), sherpa-onnx (Apache-2.0), NVIDIA Parakeet TDT 0.6B v2 and TitaNet
(CC BY 4.0), Silero VAD (MIT), pyannote segmentation-3.0 (MIT). The interface is PySide6
(LGPL-3.0). Attributions are in [NOTICE](NOTICE). Test material is public research audio
under CC BY 4.0; no private recording of any kind is used in development or testing.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).

Copyright 2026 Uniflux Technology Inc.
