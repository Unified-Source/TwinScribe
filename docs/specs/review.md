# Specification: review list, run record, load measurement

Read `CONVENTIONS.md` first, then sections 2, 3 and 7 of
`../Design_and_Findings_2026-09-06.md`, and `engines.md` for the `Word` and `Transcript`
types.

Deliver `twinscribe/review.py`, `twinscribe/runrecord.py`, `twinscribe/load.py` and tests
under `tests/test_review.py`, `tests/test_runrecord.py`, `tests/test_load.py`.

## 1. `twinscribe/review.py`: the two-engine review list

```python
@dataclass(frozen=True)
class Mark:
    start: float; end: float               # what to play, padded
    span_start: float; span_end: float     # the publisher's silent span
    detector_words: int; detector_text: str

def build_review(published: list[Word], detector: list[Word], audio_s: float,
                 min_silence_s: float = 0.8, min_detector_words: int = 2, pad_s: float = 0.4) -> list[Mark]
```

A mark is raised for every maximal span of at least `min_silence_s` in which the published
words have no coverage and in which at least `min_detector_words` detector words fall; a word
falls in a span if its interval overlaps it. Leading and trailing silence count as spans.
`start` and `end` are the span padded by `pad_s` and clamped to the audio.

```python
@dataclass(frozen=True)
class ReviewEvaluation:
    marks: int; marks_on_speech: int; precision: float
    dropped_words: int; dropped_covered: int; recall: float
    audio_to_review_s: float; audio_to_review_fraction: float

def evaluate_review(marks, published: list[Word], reference: list[Word], audio_s: float) -> ReviewEvaluation
```

A dropped word is a reference word with no published word overlapping it; a mark is on speech
if any reference word overlaps its span; a dropped word is covered if any mark's span overlaps
it. Precision and recall are 0.0 when their denominators are zero.

```python
def review_set(published: Transcript, detector: Transcript, audio_path: str,
               marks: list[Mark], reference: list[Word] | None = None,
               reference_speakers: list[str] | None = None) -> dict
```

Produces the document the verification screen reads:

```json
{
  "schema": "twinscribe.review.v1",
  "audio": "<path as given>",
  "duration_s": 0.0,
  "publisher": {"engine": "", "model": "", "preset": ""},
  "detector":  {"engine": "", "model": "", "preset": ""},
  "transcript": [{"s": 0.0, "e": 0.0, "w": ""}],
  "marks": [{"start": 0.0, "end": 0.0, "span_start": 0.0, "span_end": 0.0,
             "detector_words": 0, "detector_text": "",
             "reference_words": null, "reference_speakers": null}],
  "evaluation": null
}
```

`reference_words`, `reference_speakers` and `evaluation` are filled only when a reference is
supplied; the screen treats their presence as test mode. Provide `write_review_set(doc, path)`
writing JSON atomically (temporary file in the same folder, then rename).

Tests: synthetic word lists with known gaps; one boundary case per rule (a span exactly
`min_silence_s` long, a detector word touching the span edge, leading and trailing gaps, a
detector word count one below the threshold); a fixture that reproduces precision and recall
by hand; the JSON round-trips.

## 2. `twinscribe/runrecord.py`

`machine_facts(include_hostname: bool = False) -> dict`: processor name, logical cores,
memory in GB, operating system, Python version; the hostname only on request, default off.

`RunRecord`: engines and versions, settings, input path and digest, audio duration, load and
transcribe seconds, real-time factor, start and end times in UTC, load verdict (see section
3), and failures as `(path, error_class, message)`. `to_dict()`, and `write(path)` atomically.
The input digest is computed once and reused by whatever else needs it.

Tests: construct, serialise, write, read back; the digest matches `audio.sha256_of`.

## 3. `twinscribe/load.py`: was the machine busy with something else?

A timing is only comparable if the machine was not doing other work at the time. Provide
`Snapshot` (wall seconds, total CPU seconds consumed by all processes, CPU seconds consumed by
this process) taken from the operating system's standard counters (Windows: the system and
process time counters; Linux: `/proc/stat` and `/proc/self/stat`), and

```python
def other_load(before: Snapshot, after: Snapshot) -> float | None
```

returning the average number of cores other processes kept busy between the two snapshots,
`None` when either snapshot is unavailable, and `busy(before, after, threshold_cores=0.5) -> bool | None`.
Also `on_mains_power() -> bool | None`, `None` where it cannot be determined. Probes never
raise; they return `None`.

Tests: the arithmetic on hand-built snapshots; on the current platform the probes return the
documented shapes or `None`.

## 4. Notes file

`docs/specs/review.notes.md`: deviations, and anything about the load probe on this platform
that the specification did not anticipate.
