# Specification: text normalisation and metrics

Read `CONVENTIONS.md` first, then sections 2, 4 and 7 of `../Design_and_Findings_2026-09-06.md`.

Deliver the package `twinscribe/metrics/` and `twinscribe/text/`, with tests under
`tests/test_text.py`, `tests/test_align.py`, `tests/test_diarization.py`,
`tests/test_speakers.py`, `tests/test_timing.py`.

## 1. `twinscribe/text/`

### 1.1 Vendored normaliser

Vendor the OpenAI Whisper English text normaliser from its upstream repository
(`whisper/normalizers/english.py`, `english.json`, `basic.py`; MIT, Copyright (c) 2022
OpenAI) into `twinscribe/text/_whisper_normalizer/` with the upstream `LICENSE` beside it.
Fetch it from upstream; do not take it from anywhere else. Make it standard-library only:
inline the one `more_itertools` helper it uses, and drop the branch that needs the third-party
`regex` package if present. Record exactly what was changed in a `VENDORED.md` in that folder
with the upstream commit hash.

Expose in `twinscribe/text/normalize.py`:

- `normalize(text: str) -> str`: the English normaliser, tagged `NORMALISER_VERSION = "v1"`.
- `tokenize_raw(text: str) -> list[str]`: lower case; punctuation removed except the
  apostrophe; the underscore treated as a separator (so a reference that writes a spelled
  acronym as `C_D_` yields the same tokens the normaliser produces); split on whitespace.
- `tokenize_norm(text: str) -> list[str]`: `normalize` then split.

Tests: derive expected outputs by running the vendored normaliser on sentences that exercise
spelled numbers, contractions, fillers, currency and percentages, and capture them as
fixtures; include the asymmetry that a compound spelled number becomes digits while a lone
"one" stays a word. Test `tokenize_raw` on punctuation, apostrophes and underscores.

## 2. `twinscribe/metrics/align.py`

```python
@dataclass(frozen=True)
class Op:            # one alignment step
    kind: str        # "equal" | "sub" | "del" | "ins"
    ref: int | None  # index into the reference token list, None for "ins"
    hyp: int | None  # index into the hypothesis token list, None for "del"

def edit_ops(ref: list[str], hyp: list[str]) -> list[Op]
```

Unit-cost Levenshtein with a full backtrace. Tie-break, when costs are equal, prefers
substitution, then deletion, then insertion, and the choice is documented in the docstring
because it changes the S/D/I split on ties. `O(len(ref) * len(hyp))` memory is acceptable for
the sizes here (tens of thousands of tokens); do not stream.

```python
@dataclass(frozen=True)
class WordErrors:
    n_ref: int; n_hyp: int; hits: int; sub: int; dele: int; ins: int
    @property
    def wer(self) -> float | None      # (sub + dele + ins) / n_ref; None when n_ref == 0 and ins > 0; 0.0 when both empty

def word_errors(ref: list[str], hyp: list[str]) -> tuple[WordErrors, list[Op]]
```

```python
def cer_chunked(ref: list[str], hyp: list[str], ops: list[Op], anchor_run: int = 5) -> float
```

Character error rate computed inside chunks delimited by runs of at least `anchor_run`
consecutive `equal` ops, summing character Levenshtein distance per chunk over the total
reference characters. Document in the docstring that the chunk sum is an upper bound on the
whole-string distance and is exact when the anchors sit where a character-optimal alignment
would put them.

Tests: a 10-word reference with exactly one substitution, one deletion and one insertion
gives `wer == 0.3` with the S/D/I split (1, 1, 1); an asymmetric case that distinguishes
deletion from insertion direction; identical sequences give zero; empty hypothesis gives
`wer == 1.0`; chunked CER equals whole-string character distance on a fixture where the
anchors are clean; `edit_ops` is total (every ref and hyp index appears exactly once).

## 3. `twinscribe/metrics/diarization.py`

Segments are `(start: float, end: float, label: str)`. Labels on the two sides are unrelated
strings.

```python
@dataclass(frozen=True)
class DerResult:
    der: float; miss: float; false_alarm: float; confusion: float   # all as fractions of scored reference speech
    scored_ref_s: float
    n_ref_speakers: int; n_hyp_speakers: int
    mapping: dict[str, str]            # reference label -> hypothesis label, for mapped speakers
    unmatched_ref: tuple[str, ...]     # reference speakers mapped to a hypothesis label they never overlap, or unmapped

def der(ref: list[Seg], hyp: list[Seg], collar: float = 0.25, skip_overlap: bool = False) -> DerResult
def jer(ref: list[Seg], hyp: list[Seg], collar: float = 0.25) -> float
```

Method: build a merged timeline of every boundary from both sides; on each elementary
interval count reference speakers and hypothesis speakers; a collar of `collar` seconds is
excluded either side of every reference boundary; with `skip_overlap` the intervals with more
than one reference speaker are excluded. Choose the speaker mapping that minimises total error
by exhaustive search over injective partial mappings from reference labels to hypothesis
labels (both sides up to six speakers; raise `ValueError` beyond that with a clear message).
Miss is reference speech with fewer hypothesis speakers than reference speakers; false alarm
the reverse; confusion the speech where counts match but labels do not under the mapping.
`unmatched_ref` names any reference speaker whose mapped hypothesis label shares no time with
it, or which received no label; the design document explains why this must be surfaced.

`jer` is the Jaccard error rate: for each reference speaker under the optimal mapping, one
minus the intersection over union of its time with its mapped hypothesis speaker's time,
averaged over reference speakers with unmapped speakers counting as 1.0.

Tests: identical inputs give 0; swapped labels give 0; a reference speaker absent from the
hypothesis gives `miss` equal to that speaker's share and names it in `unmatched_ref`; the
collar removes exactly `2 * collar` of scored time per interior boundary on a fixture built to
show it; a hypothesis with more speakers than the reference maps injectively; `skip_overlap`
changes the scored duration by exactly the overlapped span; `jer` on a fixture where one of
four speakers is absorbed is far worse than `der` on the same fixture (assert the ordering and
the approximate values by hand).

## 4. `twinscribe/metrics/speakers.py`

```python
def per_speaker_recall(ref_tokens: list[str], ref_speakers: list[str], hyp_tokens: list[str]) -> dict[str, SpeakerRecall]
```

Aligns with `edit_ops` and attributes every `sub` and `del` to the speaker of the reference
token, returning per speaker the counts of tokens, recovered (`equal`), substituted and
deleted, and the recovered fraction. This is the measurement that shows a participant
dropping out of a transcript while the overall figure looks normal.

Tests: a two-speaker fixture where every deletion falls on one speaker.

## 5. `twinscribe/metrics/timing.py`

```python
def timestamp_offsets(ops: list[Op], ref_times: list[tuple[float, float]], hyp_times: list[tuple[float, float]]) -> TimingResult
```

Over `equal` ops only, the signed onset offset (hyp start minus ref start); report count,
median and 90th percentile of the absolute offset. Also
`words_in_gaps(ref_times, hyp_times, min_gap: float = 2.0) -> int`: reference words whose
midpoint falls inside a hypothesis gap at least `min_gap` long, counting the unannotated lead
and tail as gaps.

Tests: hand fixtures for both.

## 6. Notes file

`docs/specs/metrics.notes.md`: deviations, the vendored normaliser's upstream commit, and the
exact tie-break rule as implemented.
