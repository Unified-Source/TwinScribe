# Specification: audio and engines

Read `CONVENTIONS.md` first. The design record this specification follows is held outside
the repository.

Deliver `twinscribe/audio.py` and `twinscribe/engines/` with tests under
`tests/test_audio.py` and `tests/test_engines_logic.py`. The engine libraries are not
installed in the development environment; write against their documented interfaces, guard
every import, and make tests skip cleanly when a library or model is absent. `ffmpeg` is on
the path.

## 1. `twinscribe/audio.py`

- `read_wav_mono16k(path) -> numpy.ndarray[float32]`: standard-library `wave` reader;
  rejects anything but 16 kHz mono 16-bit with a clear error.
- `decode_to_wav(src, dst, ffmpeg=None, duration_s=None) -> float`: invokes `ffmpeg` with
  arguments `-i src -vn -ac 1 -ar 16000 -c:a pcm_s16le dst`, adding `-t duration_s` when
  given, as an argument list (no shell); returns the decoded duration read back from the
  file. Locate `ffmpeg` from the argument, then the path; raise a clear error if absent.
- `duration_s(path) -> float` for WAV files.
- `sha256_of(path) -> str`.
- `synthetic_wav(path, seconds, tone_hz=440.0)`: writes a sine-tone WAV; a test helper kept
  in the module so the application's own smoke check can use it.

## 2. `twinscribe/engines/base.py`

```python
@dataclass(frozen=True)
class Word:      text: str; start: float; end: float; prob: float | None
@dataclass(frozen=True)
class Segment:   start: float; end: float; text: str; words: tuple[Word, ...]; quality: dict[str, float | None]
@dataclass(frozen=True)
class Transcript:
    engine: str; model: str; preset: str
    segments: tuple[Segment, ...]
    audio_s: float
    load_s: float; transcribe_s: float
    versions: dict[str, str]
    extras: dict[str, float | int | None]
    @property
    def words(self) -> list[Word]        # flattened, ordered by start
    @property
    def text(self) -> str
```

Presets are plain dicts in `twinscribe/engines/presets.py`: for Whisper, `production`
(beam 5, no voice filter, no-speech threshold 0.7, no conditioning on previous text, word
timestamps on, compression ratio threshold 2.4, log-probability threshold -1.0, the default
temperature ladder) and `benchmark` (beam 1, voice filter on with 500 ms minimum silence,
conditioning on, word timestamps off); for the transducer, `vad` (Silero voice detection,
threshold 0.5, minimum silence 0.5 s, minimum speech 0.25 s, maximum speech 20 s, 512-sample
windows, greedy search). Both Whisper presets are kept because the design measured both.

## 3. `twinscribe/engines/whisper_ct2.py`

`transcribe(audio_path, model_dir, preset, threads=None) -> Transcript` using
`faster_whisper.WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=...)`
with threads defaulting to the smaller of 8 and the core count. Set the environment variables
that keep the Hugging Face hub offline before the import. Time the load and the transcription
separately, and make the transcription timer enclose full consumption of the segment
generator, because the library decodes lazily. Record per segment the average log
probability, no-speech probability and compression ratio the library reports, and word times
when the preset asks for them. Record the library's reported duration after voice filtering
in `extras`.

## 4. `twinscribe/engines/parakeet.py`

`transcribe(audio_path, model_dir, vad_model_path, preset, threads=None) -> Transcript` using
`sherpa_onnx.OfflineRecognizer.from_transducer(...)` with `model_type="nemo_transducer"` and
`decoding_method="greedy_search"`, and `sherpa_onnx.VoiceActivityDetector` configured from
the preset. Feed the waveform in windows of the configured size, drain detected segments after
every window and after the final flush, then decode each speech segment on its own stream.

Word grouping: the recogniser returns tokens with per-token timestamps. A token that begins
with a space starts a new word; that leading space is the word marker. A token consisting of
the marker alone can occur and starts an empty word that is dropped if nothing follows. A
word's start is its first token's time; its end is the next word's start, or the segment end.
Build the segment text from the grouped words rather than from the recogniser's own text
field. Record the number of voice-detected segments and the seconds they cover in `extras`.

On Windows the library's binary dependencies must be registered with the process before
import; keep that in one helper that is a no-op elsewhere.

## 5. `twinscribe/engines/diarize.py`

`diarize(audio_path, segmentation_model, embedding_model, threads=None, num_speakers=None, threshold=0.5) -> Diarization`
using `sherpa_onnx.OfflineSpeakerDiarization`. Clustering by threshold is the default;
`num_speakers` is an explicit opt-in, and the docstring says why forcing the count can hide a
failure. Return segments `(start, end, label)`, seconds per label, load and diarize timings,
and versions.

## 5a. `twinscribe/engines/tagging.py`

`tag_regions(audio_path, model_dir, regions, threads=None, provider="cpu", top_k=8, progress=None) -> Tagging`
using `sherpa_onnx.AudioTagging` with a CED model (`model.int8.onnx` and
`class_labels_indices.csv` in the folder). For each region (start and end in seconds) of one
16 kHz mono WAV, the top classes by probability in descending order; a region shorter than a
tenth of a second yields no events. Returns the regions, the events per region, load and
tagging timings, versions and the settings asked for. The model decides nothing itself; the
scene pass (`batch_app.md`, section 3a) turns its events into markers and into the decision to
set aside words the published engine wrote where there was no speech.

## 6. Tests

`synthetic_wav` for the audio tests; the `ffmpeg` test skips when it is absent. For the
engines, unit-test the pure parts (preset dictionaries, token-to-word grouping on hand-built
token lists including the bare-marker case, timing bookkeeping) and skip the live calls when
a library or model directory is absent, with the reason in the skip message.

## 7. Notes file

`docs/specs/engines.notes.md`: deviations, and which functions could not be exercised because
a library or model was absent.

## 8. The speaker stage in a child process

`twinscribe/engines/isolate.py` runs a function in a process started afresh and returns its
result, the caller's thread waiting on a pipe. `diarize_in_child` is `diarize` run that way,
and it is the diarizer the pipeline uses by default: the speaker library holds the interpreter
lock for the whole of a labelling call, which runs for minutes on a processor, so a window
whose thread shares the process stops answering until it ends. The function and its arguments
travel by pickling; an exception in the child is raised again in the caller. The frozen
executables call `multiprocessing.freeze_support()` first thing, which the child start needs.
