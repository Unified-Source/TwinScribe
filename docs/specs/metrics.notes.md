# Notes: text normalisation and metrics

Companion to `metrics.md`. Records the vendored normaliser's provenance, the exact tie-break
rule as implemented, and every point where the implementation chose something the
specification left open or departs from it.

## Vendored normaliser

Upstream commit `86098128c0b4f24f0e2aa2994de830614b474227` (branch `main`, 2026-08-31).
File digests and the full change list are in
`twinscribe/text/_whisper_normalizer/VENDORED.md`. In summary: `more_itertools.windowed`
is replaced by a private `_windowed` with the same behaviour at the one call site; the
`split_letters` option of `BasicTextNormalizer` (the only user of the third-party `regex`
package) is removed together with its parameter; the spelling table is opened with an
explicit UTF-8 encoding inside a `with` block; each module gained a docstring. Nothing else
changed. `english.json` and `LICENSE` are byte-for-byte upstream.

Two things in the vendored files are upstream text kept verbatim and may need a gate
exemption or a conscious pass: the `EnglishSpellingNormalizer` docstring cites an external
URL, and the title-expansion table maps the abbreviation `asst` to the word `assistant`.
The vendored `LICENSE` (SHA-256 `b5d65a59...e9d`) is not listed in
`.githooks/gate_exempt.sha256`; that file was not touched because it is outside this
specification.

`pyproject.toml` has no package-data entry, so a built wheel would omit `english.json`.
Tests run from the source tree and are unaffected. Not changed here for the same reason.

## Tie-break rule in `edit_ops`, as implemented

The table is filled row by row; the backtrace starts at cell `(len(ref), len(hyp))` and
walks to `(0, 0)`. At each cell, among the predecessor moves whose cost equals the cell's
optimal cost, the order of preference is:

1. diagonal from `(i - 1, j - 1)`: an `equal` op when `ref[i - 1] == hyp[j - 1]`,
   otherwise `sub`;
2. vertical from `(i - 1, j)`: `del` of `ref[i - 1]`;
3. horizontal from `(i, j - 1)`: `ins` of `hyp[j - 1]`.

Because the backtrace runs from the end, a tie between two equally good placements of a
deletion resolves towards keeping the later matching token (see the
`["a", "b"]` against `["c"]` case in the tests, which yields `del a, sub b->c`). Row 0 is
all insertions and column 0 all deletions. When the tokens at a cell match, the diagonal
is always optimal, so a matching pair is never reported as a substitution. The rule fixes
only how ties split between S, D and I; the total edit count is the Levenshtein distance
and is verified against a textbook implementation in the tests. A consequence recorded in
the tests: swapping reference and hypothesis preserves the total but does not in general
mirror the S/D/I split, because substitution is preferred in whichever direction the
alignment runs.

## Decisions and deviations

### `twinscribe/text/normalize.py`

- `normalize` strips leading and trailing whitespace. The upstream class can return a
  string with a trailing space (its final step collapses runs of whitespace to one space
  without stripping); the tokens are identical either way.
- `tokenize_raw` treats "punctuation" as Unicode general category P. So `%` (category Po)
  is removed while `$` (category Sc, a symbol) is kept. Hyphens are removed, not turned
  into separators: `well-known` becomes `wellknown`, whereas the normaliser produces
  `well known`. The typographic apostrophes U+2019 and U+2018 are folded to the ASCII
  apostrophe before stripping so that the two spellings of `don't` tokenise alike.

### `twinscribe/metrics/align.py`

- `levenshtein_distance(a, b)` is exposed as a public helper. `cer_chunked` needs a
  character-level distance and the tests use it to cross-check the word alignment.
- The dynamic programme runs on numpy rows (the insertion chain is a running minimum),
  with a `uint8` backpointer table of `(len(ref) + 1) * (len(hyp) + 1)` bytes, which is
  the O(n * m) memory the specification allows.
- `cer_chunked` joins tokens with single spaces and gives each token its following space
  inside the chunk strings, so that chunks and anchors concatenate exactly to the full
  strings; the denominator is `len(" ".join(ref))`. With an empty reference and a
  non-empty hypothesis it returns `math.inf` (the return type is `float`, so `None` was not
  available); both sides empty gives `0.0`. `anchor_run < 1` raises `ValueError`.

### `twinscribe/metrics/diarization.py`

- The denominator `scored_ref_s` is reference speaker-time: on an interval where two
  reference speakers overlap, each second counts twice. This is the usual DER
  convention. Consequently `skip_overlap` reduces the scored duration by twice a
  two-speaker overlapped span; the test states this in its derivation.
- The collar is applied around the start and end of every reference segment, on both
  sides of the boundary, and excludes the interval from every component (miss, false
  alarm and confusion alike). Timeline boundaries include the collar edges so that each
  elementary interval is wholly inside or wholly outside a collar.
- Miss, false alarm and confusion follow the per-interval formulas
  `max(n_ref - n_hyp, 0)`, `max(n_hyp - n_ref, 0)` and `min(n_ref, n_hyp) - n_correct`.
  The last covers the specification's "counts match but labels do not" case and also the
  mixed case where, for example, two reference speakers face one hypothesis speaker whose
  mapped label matches neither. Confusion is accumulated per interval under the chosen
  mapping rather than as the difference of two totals, so identical inputs give exactly
  zero.
- The mapping search enumerates every injective partial mapping (at most six labels per
  side; `ValueError` beyond, with `MAX_SPEAKERS` exported). Ties are broken towards fewer
  mapped pairs and then by sorted label order, so a reference speaker that shares no time
  with any free hypothesis label is left out of `mapping` and appears in `unmatched_ref`
  rather than being paired with an arbitrary label. The `unmatched_ref` check for a mapped
  speaker with zero overlap is kept for robustness but cannot fire under this tie-break.
- With no scored reference speech: rates are `0.0` when there is no error and `math.inf`
  when there is (a hypothesis speaking over a silent reference).
- `jer` uses the mapping chosen by `der` at the same collar with overlap included; it does
  not run a separate Jaccard-optimal search. Intersection and union are measured on scored
  intervals only. An empty reference returns `0.0`.
- Segments with `end < start` raise `ValueError`; zero-length segments contribute nothing
  but their label counts towards the speaker totals. A negative collar raises `ValueError`.

### `twinscribe/metrics/speakers.py`

- The result dictionary is keyed by speaker in order of first appearance in the
  reference. `recall` is `None` for a speaker with no reference tokens (cannot occur from
  `per_speaker_recall` itself, which only creates entries for speakers that appear).
- A length mismatch between `ref_tokens` and `ref_speakers` raises `ValueError`.

### `twinscribe/metrics/timing.py`

- `TimingResult` carries `offsets_s`, the signed offsets in alignment order, beside `n`,
  `median_abs_s` and `p90_abs_s`. The specification asks for the three statistics; the
  raw offsets are included so that a report can bootstrap an interval or plot them without
  recomputing the alignment. The 90th percentile is numpy's default linear interpolation.
- `words_in_gaps` defines the annotated extent as the span from the earliest start to the
  latest end over both sides. The lead (extent start to first hypothesis word) and the
  tail (last hypothesis word to extent end) are gaps subject to the same `min_gap` as
  interior gaps. With no hypothesis words the whole extent is one gap. A reference
  midpoint exactly on a gap edge counts as inside. The gap list itself is exposed as
  `hypothesis_gaps` so the review list can show it.

## Tests

76 tests across the five files. Every metric has hand-derived fixtures and at least one
property test: alignment totality and agreement with a textbook distance on random
inputs, length-difference and bound invariants, chunked CER as an upper bound, DER
component sum and relabelling invariance, JER bounds, per-speaker counts partitioning the
reference, offset sign flip, and gap-count monotonicity. No test writes files.
