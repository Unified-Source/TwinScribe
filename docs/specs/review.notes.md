# Notes: review list, run record, load measurement

Deviations from `review.md`, and what the load probe found on the platform it was built on.

## Delivered

- `twinscribe/review.py`, `twinscribe/runrecord.py`, `twinscribe/load.py`
- `tests/test_review.py`, `tests/test_runrecord.py`, `tests/test_load.py`
- `tests/_types_stub.py`: frozen dataclasses equivalent to `Word`, `Segment` and
  `Transcript` from `engines.md` section 2, used by the review tests only when
  `twinscribe.engines.base` cannot be imported. The tests import the real types first. Both
  paths were exercised: the suite passed against the real engines package, and again with
  that import blocked so the stub was used.

## Deviations and interpretations

### review.py

- `Word` and `Transcript` are imported from `twinscribe.engines.base` under `TYPE_CHECKING`
  only. The module reads attributes (`start`, `end`, `text`, `engine`, `model`, `preset`,
  `words`, `audio_s`) and never constructs those types, so it imports cleanly without the
  engines package. This keeps the review list usable in a test or tooling context that has
  no engine libraries installed.
- Overlap is strict: two intervals overlap when they share a positive-length intersection.
  A detector word that ends exactly where a span starts does not fall in that span. A
  zero-length interval overlaps another only when it lies strictly inside it; two
  zero-length intervals never overlap. One rule serves the mark builder and the evaluation.
- The span-length test carries a tolerance of 1e-9 s so that a span whose end points come
  from decimal word times (2.0 - 1.2) still counts as exactly 0.8 s long.
- Published words are clamped to `[0, audio_s]`; words entirely outside the audio, and
  zero-length words, cover nothing. Overlapping or unsorted published words are merged
  before the silent spans are derived, so every span is maximal by construction.
- `audio_s <= 0` returns no marks. Negative `min_silence_s`, `pad_s` or
  `min_detector_words` raise `ValueError`; `min_detector_words = 0` is allowed and marks
  every qualifying silent span.
- `detector_text` joins the detector words in start order with single spaces after
  stripping each word, because Whisper word tokens usually carry a leading space.
- `audio_to_review_s` is the length of the union of the padded play windows, so windows
  that overlap after padding are not counted twice. The fraction is 0.0 when `audio_s` is 0.
- "On speech" and "covered" are decided against the mark's span, not its padded window.
- In the review-set document, `reference_words` is the count of reference words that fall
  in the mark's span (an integer, matching `detector_words`) and `reference_speakers` is the
  list of distinct speaker labels of those words in order of first appearance. This matches
  what the verification screen states for a mark in test mode. When a reference is supplied
  without speaker labels, `reference_words` and `evaluation` are filled and
  `reference_speakers` stays `null`. Supplying labels without a reference, or a label list
  of the wrong length, raises `ValueError`.
- The transcript entries carry the published word text as given, without stripping.
- The atomic JSON writer is shared: `write_review_set` delegates to
  `twinscribe.runrecord.write_json_atomic`.
- Interval queries use a sorted index with a running maximum of end times, so the
  evaluation is `O((n + m) log n)` rather than quadratic over published and reference words.

### runrecord.py

- The digest helper `input_digest(path)` is implemented locally with `hashlib` and is
  intentionally standalone: the record module keeps no dependency on the audio module, so a
  record can still be written for an input the audio module refused to open. `RunRecord`
  takes the digest as a value, so whoever already holds it (normally from
  `audio.sha256_of`) passes it in and the file is hashed once. The test checks the local
  digest against `hashlib` directly and against `audio.sha256_of` (skipping when that module
  is absent); both agree.
- `RunRecord` carries, beyond the listed fields, `machine` (default `machine_facts()`, so a
  record identifies the class of machine and never the machine) and a `schema` key
  `twinscribe.runrecord.v1` in `to_dict()`.
- `failures` accepts plain `(path, error_class, message)` tuples, dicts or `Failure`
  instances and normalises them to `Failure` records at construction.
  `Failure.from_exception(path, exc)` builds one from a caught exception.
- `real_time_factor` is a property, `transcribe_s / audio_s` with load time excluded, and
  `None` when `audio_s` is 0. It is serialised beside the timings.
- Start and end times are ISO 8601 strings in UTC with second resolution; `utc_now()` makes
  them.
- `machine_facts()` returns `processor`, `logical_cores`, `memory_gb`, `os`, `architecture`
  and `python`; `architecture` is an addition (the same processor name can appear on 32-bit
  and 64-bit builds). The `hostname` key is absent unless `include_hostname=True`. Every
  fact is probed separately and is `None` when its probe fails. Memory is reported in GiB
  rounded to one decimal.
- `write_json_atomic` writes to a temporary file created in the target folder, flushes and
  fsyncs it, then renames over the target; the temporary file is removed on any failure.
  Values `json` cannot serialise are coerced: numpy scalars via `.item()`, dataclasses via
  `asdict`, tuples and sets to lists, path-like objects to strings; anything else raises
  `TypeError` and leaves no file behind.

### load.py

- `LoadVerdict` (other-load cores, busy flag, mains flag, threshold) and
  `verdict(before, after, threshold_cores)` are additions; they are the "load verdict" that
  `RunRecord` carries and that the design record's section on contention calls the contention
  verdict.
- `busy` is `True` when the other-load figure is strictly greater than `threshold_cores`.
- `other_load` clamps at 0.0: counter granularity can make the process delta exceed the
  system delta by a tick.
- The "idle baseline sampled just before the cell" in the design record's section on contention is
  not a separate subtraction here. `other_load` already isolates other processes' CPU time
  through the process counter; a caller who wants a background baseline takes a snapshot
  pair before the cell and compares the two figures.
- Platforms other than Windows and Linux get `None` from both probes.

## The load probe on this platform

Built and tested on Windows on a 10-core ARM64 machine (from `machine_facts()`; no hostname
was read or written).

- Counters: `GetSystemTimes` (kernel time includes idle time, which is subtracted) and
  `GetProcessTimes` on the current process pseudo-handle, both through `ctypes` with
  explicit argument and return types so the pseudo-handle is not truncated on 64-bit.
- Both counters advance at the scheduler tick, about 15.6 ms. Cells shorter than roughly one
  second give a coarse other-load figure; the 50 ms test cell is enough for a monotonic
  check but not for a trustworthy verdict. Bracket real transcription cells, not toy ones.
- Observed while the sibling packages were being built and tested on the same machine: a
  0.5 s cell read 1.25 other cores and `busy=True`. That is the intended behaviour: the
  probe reports concurrent work as contention, and the timing rows it qualifies are
  excluded from speed tables and counted.
- `GetSystemPowerStatus` reported mains power; an `ACLineStatus` of 255 maps to `None`.
- The Linux path (`/proc/stat` with idle and iowait excluded from busy time, `/proc/self/stat`
  parsed after the last closing parenthesis so command names with spaces do not shift the
  fields, `/sys/class/power_supply` for mains) is written from the documented counter
  layouts and was not executed here; the probe-shape tests run there unchanged.
