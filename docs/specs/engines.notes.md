# engines: notes on the delivery

Deviations from `engines.md`, with the reason for each, and the list of functions that could
not be exercised because a library or model was absent.

## Deviations and additions

1. **`Diarization` and `SpeakerTurn` live in `base.py`.** The specification lists three
   records for `base.py` and has `diarize()` return a `Diarization` without saying where it
   is defined. All records are kept together in `base.py`; `diarize.py` imports them.
2. **`base.py` also carries `Stopwatch`, `elapsed_since` and `default_threads`.** Both
   engines need the same thread default (min of 8 and the core count) and the same timing
   pattern, so they are defined once and unit-tested there.
3. **`transcribe()` and `diarize()` accept the preset as a name or a mapping.** The
   specification defines presets as dictionaries but types `Transcript.preset` as a string.
   A name is looked up in the preset table and recorded as given; an explicit mapping is
   recorded under the name `custom` so a transcript always states what produced it.
4. **ffmpeg is invoked with `-hide_banner -nostdin -loglevel error -y` before the specified
   arguments.** The specified list is passed unchanged; the extra flags make the call
   non-interactive, let it overwrite a stale output, and keep stderr to real errors. `-t`
   is placed as an output option immediately before the destination.
5. **Benchmark preset thresholds.** The specification fixes beam, voice filter, minimum
   silence, conditioning and word timestamps for `benchmark` and is silent on the three
   thresholds and the temperature ladder. They are written explicitly at the library
   defaults (no-speech 0.6, compression ratio 2.4, log probability -1.0, default ladder) so
   the preset is self-describing. `production` uses no-speech 0.7 as specified.
6. **Whisper presets are validated against the library's parameter names** before the model
   loads, so a misspelt key fails in under a millisecond rather than after a model load.
7. **`WhisperModel` receives `local_files_only=True`** in addition to the offline
   environment variables, and the environment variables are forced to `1` rather than set
   only when absent, because a stray `0` in the environment would otherwise permit a hub
   lookup. The model directory is checked for `model.bin` and `config.json` before import.
8. **Transducer model files.** `resolve_model_files()` accepts the int8 export names first
   (`encoder.int8.onnx` and so on) and the fp32 names as a fallback; int8 wins when both
   are present because it is the form the design measured.
9. **Trailing partial voice-detector window.** Audio shorter than one window at the end of a
   file is padded with zeros to a full window and fed, rather than discarded, so no audio is
   dropped silently. The detector is then flushed and drained as specified.
10. **Word probability for the transducer is `None`.** Greedy transducer search reports no
    per-token probability, so `Word.prob` is `None` and `Segment.quality` is empty for that
    engine.
11. **Short timestamp lists are padded, not rejected.** If the recogniser returns fewer
    timestamps than tokens, the last known time is repeated so that no token is lost; an
    empty timestamp list places every token at the segment start.
12. **Empty voice-detected segments are kept.** A detected speech region that decoded to no
    words remains in `segments` with empty text; it is honest bookkeeping and the count is
    recorded in `extras["empty_segments"]`. `Transcript.text` skips them.
13. **Detected language is not stored.** `Transcript.extras` is typed numeric, so only
    `language_probability` is recorded; the language code itself is not.
14. **Diarization labels** are formatted `speaker_00`, `speaker_01`, ... from the library's
    integer cluster index. Segment ordering is by start, end, then label so the output is
    the same regardless of the order the library returns them in.
15. **Windows binary registration** is one helper, `parakeet.register_windows_dlls()`,
    which adds the library package's `lib` directory (where the compiled extension and its
    runtime sit) and the package directory to the process DLL search path when they contain
    DLLs. It is a no-op on other platforms and when the package is not installed. `diarize.py`
    reuses it through `parakeet.import_sherpa_onnx()`.
16. **Extras record the decode share.** For the transducer, `extras` carries the decode
    seconds and the remainder attributed to voice detection beside `transcribe_s`, so every
    timing has what it measured beside it.

## Functions not exercised live

Neither engine library was installed in the development environment and no model directory
was present, so the following ran only as far as their input validation:

- `whisper_ct2.transcribe()`: the model-directory and audio checks are tested; the library
  call, the lazy generator consumption and the timing split are written against the
  documented interface and not executed.
- `parakeet.transcribe()`: model-file resolution, the voice-detector preset and the word
  grouping are tested; recogniser and detector construction, the window feed loop, the
  final flush and the per-segment decode are not executed.
- `diarize.diarize()`: model-file checks, argument validation, label formatting and the
  seconds-per-label arithmetic are tested; configuration, validation, `process()` and the
  result iteration are not executed.
- `parakeet.register_windows_dlls()` and both `library_versions()` helpers ran on their
  library-absent paths only.

The live tests `test_whisper_live`, `test_parakeet_live` and `test_diarize_live` skip with
the reason (library not installed, or the corresponding environment variable
`TWINSCRIBE_WHISPER_MODEL_DIR`, `TWINSCRIBE_PARAKEET_MODEL_DIR`, `TWINSCRIBE_SILERO_VAD`,
`TWINSCRIBE_SEGMENTATION_MODEL`, `TWINSCRIBE_EMBEDDING_MODEL` unset). Once models are
present they exercise the whole path on a synthetic tone and check the bookkeeping, not
the words.

## Later additions (batch application)

17. **Progress callbacks.** `whisper_ct2.transcribe()` and `parakeet.transcribe()` accept an
    optional `progress` argument, called with the fraction of the audio reached: after every
    decoded segment for Whisper, at most a few hundred times per file plus once with 1.0
    after the final flush for the transducer. An exception raised inside it propagates and
    abandons the run, which is how the application cancels. The default is None and nothing
    else about the wrappers changed. Not run inside a live decode on the development machine.
18. **The `quick` Whisper preset.** Production with greedy decoding (beam 1) and word times
    on. The benchmark preset has word times off and so cannot feed the review list; the quick
    quality level of the application needed a preset that can.

## Exercised live, on the development machine

After the first delivery the libraries were installed: sherpa-onnx natively on the 64-bit
ARM interpreter, and the whole engines extra in a second, x64 environment under the platform's
emulation, because CTranslate2 publishes no wheel for the native architecture. The
standard-profile models were fetched with `tools/fetch_models.py` and verified against the
lock. Nothing in the three wrappers needed changing to run live.

- The three live tests pass: `test_parakeet_live` and `test_diarize_live` natively and under
  emulation, `test_whisper_live` under emulation.
- Called directly on a synthesised two-voice conversation (two speech-synthesiser voices
  reading a 28-line script; 143 seconds; 335 script words), the transducer returned 324 words
  from 46 voice-detected segments covering 96 seconds of speech, Whisper large-v3-turbo 335
  words in 31 segments, and the diarizer two labels of 56.9 and 42.9 seconds. Scored with the
  project's own metrics against the script:

  | Engine | WER raw | WER normalised | S / D / I, normalised |
  |---|---|---|---|
  | transducer (published) | 7.5 per 100 words | 3.6 | 5 / 6 / 1 |
  | Whisper large-v3-turbo (detector) | 5.7 per 100 words | 1.5 | 4 / 0 / 1 |

  The deletions sit with the transducer and none with Whisper, the direction the design
  describes. Review list at the design's rule: 5 marks, all five on speech, 7.0 per cent of
  the audio to listen to; every mark sat on a phrase the transducer had dropped, the detector's
  hints being "Is it the same", "It does.", "The booking", "What do" and "No, that". Against
  script word times spread evenly inside each line (approximate, so the dropped-word count
  of 30 is inflated by timing), the marks covered 7 dropped words. Diarization against the
  scripted line spans: DER 15.4 per cent, all of it miss (turns shorter than the scripted
  spans, which include the synthesiser's padding), no false alarm, no confusion, JER 15.2 per
  cent, both speakers mapped and none unmatched.
- Timings under emulation for the 143 second file, recorded as observed and not comparable
  with any other machine: transducer load 2.3 s and decode 19.1 s; Whisper load 4.4 s and
  decode 97.8 s; diarizer 22.8 s. The run records flagged other load on the machine. No figure
  here is to be quoted; the design document reserves speed for the lab hardware.
- `library_versions()` reports the onnxruntime version sherpa-onnx bundles (1.27.1 here),
  which differs from the onnxruntime package installed beside it (1.29.0); the run record
  therefore names the runtime the transducer actually used, which is the right one.
- Both `progress` callbacks fired as specified and drove the window's progress display.
- Synthesised speech is studio-clean, two voices, no overlap: these figures show that the
  chain works, not what it does on recordings.

## Device and provider parameters, and the ONNX detector

19. **`whisper_ct2.transcribe()`** accepts `device` ("cpu" or "cuda"), `compute_type` and
    `device_index`; the defaults are the measured configuration. **`parakeet.transcribe()`**
    and **`diarize()`** accept `provider` (the onnxruntime execution provider; "cuda" only
    does anything with the CUDA build of the library, and falls back with a warning
    otherwise). All three record what they were asked in the transcript's new `settings`
    field, which the run record carries. On a CUDA request the CUDA runtime packages, when
    installed as packages, are registered with the process first (`hardware.py`).
20. **`engines/whisper_onnx.py`** is a second detector, Whisper through sherpa-onnx, for the
    platform where CTranslate2 has no wheel (Windows on ARM). It decodes windows of up to
    thirty seconds cut at the quietest moment before each limit, never the voice detector's
    utterances (the transducer decodes those, and a detector that saw the same boundaries
    failed in the same places: measured, zero marks). The library asks for token timestamps;
    the published exports have no cross-attention outputs and return none, so the recogniser
    is built again without the request (to stop a warning per window) and words are spread
    inside each segment's timestamps, confined to the voiced parts the voice detector finds.
    Run natively on the synthesised conversation with the turbo export: 330 words, normalised
    WER 1.8 per hundred (4 substitutions, 0 deletions, 2 insertions), 6 windows, load 3 s and
    decode 32 s on this machine (not quotable); review list 3 marks, all on real dropped
    phrases, against 5 for the CTranslate2 detector. The transcript's settings say
    `word_timing: segment` so every document downstream can say the times are approximate.
