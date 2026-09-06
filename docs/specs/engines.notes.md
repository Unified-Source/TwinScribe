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
