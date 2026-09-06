# Offline transcription with speaker labels: design and findings handoff

**Uniflux Technology Inc. · 2026-09-06 · Own-account R&D · Prepared by Abdirahman Kulmiye**

## 0. What this document is, and what it deliberately is not

This is a record of what I know about building an offline speech transcription and
speaker-labelling tool for long recordings that will be relied on, written down so that the
tool can be built fresh in this workspace. It carries design, findings, the engine and licence
landscape, corpus knowledge, and the engineering traps I have already paid for.

It carries no code, no configuration files, no evaluation fixtures and no measurements taken on
anyone else's hardware. Everything measured here was measured on public research corpora with
open-licence engines, and accuracy on such material is a property of the engine and the audio,
not of the machine that ran it; I checked that by running identical tests on two different
computers and they agreed to within half an error per hundred words on an adequate sample.
Speed is a property of the machine and is not carried over. It is to be re-measured on lab
hardware, and until then no throughput figure in this document should be quoted.

The implementation is to be a fresh artifact under charter section 13: fresh history,
private first, public flip as a separate confirmed step, gate on contents and history, stranger
clone after release.

## 1. The problem

Take audio and video recordings (MP3, MP4, AVI and the other common containers), on an
ordinary laptop with no network and no administrator rights, and produce a timestamped,
speaker-labelled transcript in plain text and in Word, with a record of how it was produced.
Two workloads shape every decision: batches of roughly twenty-minute two-party telephone
recordings at narrowband quality, and older interview-room video with a single far-field
microphone and several speakers.

The output will be relied on by people who were not present, so the governing rule is that a
transcript is a draft until verified against the recording, and the tool's job is to make that
verification cheap and targeted rather than a read-through.

## 2. The finding that drives the design

There are two families of open transcription engine and they fail in opposite directions.

The Whisper family (encoder-decoder, autoregressive) predicts each word partly from the words
it has already written, so it always produces something, and on hard or silent audio it can
produce fluent text that was never said. The transducer family (NVIDIA Parakeet TDT, via
sherpa-onnx) can only emit a word while consuming audio frames, so it cannot write into
silence, and its characteristic failure is the opposite: on hard audio it produces nothing.

Measured on 24 public telephone calls, 2,341 reference words, at production settings:

| Engine | Words dropped entirely | Words wrong | Error rate per 100 words |
|---|---|---|---|
| Whisper large-v3-turbo (CTranslate2, int8) | 36 | 64 | 6.4 |
| Distil-Whisper large-v3 | 58 | 85 | 9.0 |
| Parakeet TDT 0.6B v2 | 165 | 99 | 12.3 |

The transducer drops three to five times as much speech as either Whisper engine, silently.
On a four-speaker meeting recording with close microphones the ranking inverts: Parakeet 16
errors per hundred words against 22 and 23 for the two Whisper engines. On the same meeting
from one far-field microphone, Distil-Whisper went from 22 to 26.

Two indicators of invented text that I built did not survive checking and must not be
reused as built. Words placed where the reference is silent turned out to be the same words
timed differently, because the reference word timings in the telephone corpus are
approximations; checked against the reference text, the counts collapse and the difference
between engine families disappears. Runs of five or more consecutive inserted words were all
postcodes and telephone numbers transcribed correctly and split into digits by the scoring
normaliser. The invention risk is real and documented in the Whisper model card and in the
published audits, but on 24 minutes of clean role-played calls it was not observed, and any
indicator for it has to be checked against what the reference says, not only when it says it.

## 3. The two-engine design

Run both engines. Publish only the transducer's transcript, because it does not invent. Run
the Whisper engine alongside, never show its text to anybody, and use it for one purpose: to
mark every span where it heard speech and the published engine heard nothing. Those marks are
the review list. They turn the published engine's silent failure into a marked one, for no
extra dependency, because both engines are needed anyway.

Rule that worked: a mark is raised where the published transcript has no word for at least
0.8 seconds and the detector placed at least two words inside that span; play 0.4 seconds of
run-in either side. Tested against the human reference on the same 24 calls:

| Marks raised | On real missed speech | Of the 154 words the publisher dropped, covered | Audio a person listens to |
|---|---|---|---|
| 62 | 61 (98 per cent) | 141 (92 per cent) | 18 per cent |

One false alarm in sixty-two. That result is the reason the design exists and it should be
re-verified on any new corpus before the thresholds are trusted.

Per file, the tool produces: a Word document and a plain text file (timestamped, speaker
labelled); a subtitle file so the transcript can be played against the recording line by
line; the review list; a speaker summary at the top giving the word count assigned to each
speaker; and a run record naming the engines and versions, the settings, the file's digest,
when it ran, how long it took, and every file that failed and why.

The speaker summary is not decoration. On the meeting recording, one speaker's 38 words were
produced by nobody at all under one engine's production settings while the overall error
rate read a normal-looking 23 per hundred. A participant can vanish from a transcript with
nothing on the page to show it, and a word count against each name is the cheapest way to
notice.

## 4. Speaker labelling

Embedding-and-clustering diarization (pyannote segmentation-3.0 for speech regions, TitaNet
embeddings, clustering, all through sherpa-onnx) works and is the weakest part of the system.
With the speaker count supplied in advance, the share of the recording carrying the wrong
label was 15 per cent on the four-speaker meeting and 32 per cent on telephone calls.

The 15 per cent flatters it. Forcing the count to four, the system returned four clusters but
two of the four speakers were never separated: their speech was absorbed into other labels and
the count was satisfied by a degenerate cluster a few seconds long. Diarization error rate is
duration-weighted, so a speaker who talks for forty seconds of twenty minutes can vanish at
almost no cost to the score. Consequences for the build:

- do not force the speaker count in production; use a clustering threshold so the failure is
  visible rather than silent;
- score with Jaccard error rate beside DER, because JER weights every speaker equally;
- report per-speaker word counts and per-speaker recall, not only overall figures;
- labelling costs about the same as transcription, not double.

The quiet-and-overlapped case is structural. On the meeting, the genuinely quiet speaker sat
about 6 dB below the loudest and was talked over a third of the time. Level normalisation was
measured and ruled out: the gain filters close only 1 to 2 dB of the gap, the best arm was
plain uniform gain (which changes no relative level at all), and time-varying gain is neutral
to harmful because a gain that is a function of time alone cannot separate two people talking
at the same moment. Fixing overlap needs separate microphones, separate call legs where a
recorder stores them, or source separation, and separation synthesises audio that was not in
the recording, which rules it out for anything evidential.

Verify the instrument before interpreting the measurement. The other apparently quiet speaker
on that meeting turned out to be the loudest person in the room through the shared room
microphone; the corpus's own published defect list records that participant's headset was worn
improperly. Read a corpus's known-defects page before building on its audio.

## 5. Audio pre-processing

Convert to 16 kHz mono PCM and nothing else. Standard noise reduction (afftdn) made the
transcript of genuinely far-field audio worse by three errors per hundred words, agreeing with
the published work. Level normalisation does not help (section 4). Silence stripping is
dangerous for evidential audio because it silently drops quiet speech; if used at all, log
every removed region. Never transcribe from an enhanced copy; keep any enhancement as a
separate, documented, listening-only artifact.

Container handling is not the hard part: ten containers (MP3, M4A, WMA, FLAC, OGG, AVI, MP4,
MKV, WMV, MOV) all decoded through PyAV inside faster-whisper with at most 1.4 points of
accuracy drift. Render container test files with a hard duration cut, because a synthetic video
track pads the file and inflates any speed figure. MP3 inside AVI cannot carry the gapless tag
and decodes about 0.1 s long; that is a known exception, not a defect.

## 6. Engine and licence landscape

Usable without procurement or a signature:

| Component | Licence | Role |
|---|---|---|
| faster-whisper 1.2.x, CTranslate2 4.8.x | MIT | Whisper family on CPU, int8 |
| Whisper weights (tiny.en to large-v3, large-v3-turbo, Distil-Whisper) | MIT | CTranslate2 conversions from Systran and mobiuslabsgmbh repositories |
| sherpa-onnx 1.13.x | Apache-2.0 | transducer decoding, VAD, segmentation, embeddings, clustering |
| NVIDIA Parakeet TDT 0.6B v2 | CC BY 4.0 | the published engine; attribution required |
| NVIDIA Parakeet TDT 0.6B v3 | CC BY 4.0 | multilingual fallback, slightly worse on English |
| Silero VAD | MIT | speech detection for the transducer's long-form recipe |
| pyannote segmentation-3.0 (ONNX export via sherpa-onnx) | MIT | speech regions for diarization |
| NVIDIA TitaNet-large | CC BY 4.0 | speaker embeddings; attribution required |
| OpenAI Whisper English text normaliser | MIT | vendored, stdlib only |
| PySide6 (full, with QtMultimedia) | LGPL-3.0 | interface and playback; dynamic linking, ship the notice |

Blocked or conditional, recorded so nobody re-derives them:

- CC BY-NC, unusable commercially: NVIDIA `diar_sortformer_4spk-v1` (v2 is CC BY 4.0),
  DiariZen, the original `canary-1b` (the 180m-flash variant is CC BY 4.0), torchaudio's
  `MMS_FA` aligner, WhisperX's non-English aligners (passing `--language fr` silently pulls a
  non-commercial model; pin `--align_model` explicitly if WhisperX is ever used).
- NVIDIA Open Model License (`parakeet-unified-en-0.6b`, Sortformer v2.1, Multitalker
  Parakeet): royalty-free and unsigned but revocable, with an indemnity to NVIDIA and Delaware
  law. A lawyer's call, not an engineer's. `parakeet-unified-en-0.6b` is the strongest
  candidate to beat v2 on telephone audio because its training set names Switchboard as well as
  Fisher, so the licence question is worth asking.
- Gated downloads: `pyannote/speaker-diarization-community-1` (CC BY 4.0 but behind a
  Hugging Face account and a use-case declaration), Cohere Transcribe. Fix is a one-time
  supervised download by a named person, weights vendored, acceptance recorded and dated.
- Third-party mirrors of NVIDIA weights (the sherpa-onnx redistributions) carry no licence
  metadata, and at least one asserts the wrong licence. Record the licence file and commit hash
  of the exact repository the bytes came from.
- Integrated graphics on Windows laptops: no official Vulkan build of whisper.cpp exists, and
  DirectML through sherpa-onnx measures roughly fifty times slower than the CPU on diarization.
  Settled negative; do not reopen.
- A dedicated NPU on the same silicon exists and was not examined; the driver-level install it
  needs rules it out on managed machines anyway.

Candidates for a second round of measurement, in order: `parakeet-unified-en-0.6b` (licence
permitting), Parakeet v3 as the insurance if not, Streaming Sortformer 4spk v2 (CC BY 4.0,
end-to-end, no clustering step and so no absorption failure, but degrades past four speakers
and the official ONNX export is broken), pyannote community-1 as the overflow path past four
speakers, VBx (BUT, Apache-2.0) as a resegmentation stage over any embedding extractor,
`facebook/wav2vec2-large-robust-ft-swbd-300h` (Apache-2.0, the only clean model actually
fine-tuned on telephone speech) as an independent arbiter, and a CTC forced aligner
(`facebook/wav2vec2-large-960h-lv60-self`, Apache-2.0) to make word timestamps a
post-processing step that cannot invent words because the text is an input.

## 7. Metrics, as specified

- Word error rate on both raw tokens (lower case, punctuation stripped, underscore treated as
  a separator so spelled acronyms match) and on the vendored English normaliser; report both,
  with the reference word count beside every figure, because short clips have coarse
  resolution.
- Alignment by unit-cost Levenshtein with backtrace, tie-break substitution before deletion
  before insertion; report substitutions, deletions and insertions separately, always. Deletion
  is the transducer's characteristic failure and normalised WER hides it.
- Character error rate inside chunks between runs of five or more aligned words; an upper
  bound on the whole-string distance, exact in practice.
- Diarization error rate on a merged boundary timeline, collar 0.25 s per side (the pyannote
  0.5 convention) and 0, overlap included and excluded, optimal mapping by exhaustive search up
  to six speakers, miss, false alarm and confusion split out, and JER beside it.
- Per-speaker recall, using the reference's speaker labels, so a lost participant is visible.
- Timestamp offsets, median and 90th percentile, engine word times against reference word
  times, only where the engine produced its own times.
- Statistical power: 252 reference words could not rank three engines (three word errors
  apart); 2,341 could. Do not quote a ranking from under about two thousand words, and
  bootstrap confidence intervals when a claim will be relied on.
- Every timing row carries a contention verdict (other processes' CPU time over wall time
  against an idle baseline sampled just before the cell) and a mains-power flag; contended
  rows are excluded from speed tables and counted.

## 8. Public corpora, and their limits

| Corpus | Licence | What it gives |
|---|---|---|
| AMI Meeting Corpus, meeting ES2002a, headset mix and single distant microphone | CC BY 4.0 | four-speaker meeting, close-mic and genuinely far-field versions of the same words; word-level reference |
| HarperValleyBank | CC BY 4.0 | genuine 8 kHz telephone audio, two channels per call, human transcripts with word offsets |
| LibriSpeech test-other | CC BY 4.0 | clean read speech; smoke tests and container decode tests only |

Not usable without procurement or an account: the LDC corpora (Switchboard, Fisher,
CALLHOME), TalkBank CallHome, Earnings-22 (transcripts CC BY-SA, audio licence unnamed).

Limits that belong in any report built on these: AMI participants are recruited volunteers
role-playing a design team, largely non-native English speakers; HarperValleyBank calls are
role-played bank enquiries from a 59-speaker pool; LibriSpeech is volunteers reading novels.
None was recorded under stressful or adversarial conditions, none contains documented
Canadian English, and none can detect the accuracy disparities across accent and dialect that
the published research is unambiguous about. What they establish is how engines differ from
one another and how they fail; what they cannot establish is accuracy on a specific
organisation's audio.

Corpus mechanics worth keeping: the AMI 180 s window at 780 to 960 s is the densest speech;
word times come from forced alignment and are least reliable under overlap; the corpus's
defects page records ES2002a participant 1 (agent A) wearing the headset improperly. For
HarperValleyBank, use the `human_transcript` field, never the machine `transcript` field; mix
agent and caller channels with the agent delayed by the channel length difference; pin call ids
per item so that widening the download set never silently redefines an item that was already
measured.

## 9. The verification screen

PySide6 with QtMultimedia ran with nothing installed beyond the wheels: a timeline bar of the
whole recording with every mark on it, a list of marks, the published transcript either side
of the gap, what the second engine heard as a hint of what to listen for, and three actions per
mark (play the span, nothing was said, type what was said). Keys: Space play or pause, J and K
next and previous mark, Enter play the span, N nothing said, T type it, arrows nudge five
seconds. A session file records every resolution.

pywebview is ruled out as built: its Windows backend hardcodes the modern .NET runtime
(`PYTHONNET_RUNTIME = 'coreclr'`), which Windows 11 does not ship, so on a managed laptop it
is an install request. PySide6-Essentials lacks QtMultimedia; the full PySide6 with Addons is
the cost of real playback and it is a size cost only. `pip download` for PySide6 needs
`--abi abi3` stated explicitly.

## 10. Engineering traps already paid for

- sherpa-onnx returns tokens with a leading-space word marker, not U+2581, and a bare marker
  token can appear alone. Build segment text from the grouped words, not from the engine's
  `text` field, which can differ by a token.
- CTranslate2's compression ratio is per 30 s window; use zlib, not gzip, to match its scale.
- Container test renders must be cut at the canonical audio length (`-t`), or the video track
  pads the file.
- On Windows, `if errorlevel 1` is a signed comparison and misses status codes such as
  0xC000013A; test `"%ERRORLEVEL%"=="0"` instead. Quote every `set`, or an ampersand in a path
  truncates the value. Robocopy exit codes 0 to 7 are success.
- Compiled Python caches embed the absolute build path; strip every `__pycache__` from anything
  that ships, and screen the shipped tree rather than the source tree.
- Never screenshot Qt under `QT_QPA_PLATFORM=offscreen`; it has no font database and renders
  every glyph as a box.
- `proxy_tools` is sdist-only; wheel it once and pin the hash for any offline install.
- A count assertion in a test suite ("plan has 16 entries") is the wrong shape; assert the
  invariant it was standing in for (every item a tier runs has a rendered arm).
- A provenance guard that decides by heuristic before it decides by identifier fails in the
  safe direction and still costs the whole answer; test the explicit name first.
- Widening a corpus download must never redefine an item already measured; pin the item's own
  ids and rebuild byte-identical.

## 11. Roadmap

1. Repository skeleton under the fresh-history rule, private, identity
   `abe.kulmiye@uniflux.io`, licence to be chosen (Apache-2.0 is the natural fit beside the
   Apache-2.0 and MIT dependencies; the CC BY attributions and the LGPL notice ship in a
   NOTICE file regardless).
2. Bench: corpus fetch and pinning, variants, the two engines, the metrics above, a report.
   Re-measure speed on lab hardware. Reproduce the accuracy table in section 2 as the first
   milestone; anything materially different is a bug.
3. Review list and verification screen, reproducing the section 3 table.
4. Batch application: one window, drop files, choose a quality level, progress, the six outputs
   per file, no network access in the code rather than by policy.
5. Second-round diarization from the section 6 candidates, and per-leg handling where a
   recorder stores channels separately, which makes speaker labelling exact and free for that
   material.
6. Release gate: contents and history screened, pre-push hook, stranger clone, then the public
   flip as its own confirmed step.
