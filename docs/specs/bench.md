# Specification: the bench

Read `CONVENTIONS.md` first. The design record this bench reproduces is held outside the
repository; the figures it set are restated in `bench.notes.md` beside what was measured.

Deliver the package `twinscribe/bench/`, the tools `tools/fetch_corpora.py` and
`tools/bench.py`, and `tests/test_bench.py`. The bench reproduces the design's accuracy tables
on the public corpora the design names, scores every engine with the metrics package, and
measures speed with a contention verdict beside every timed cell. Nothing in the package
reaches the network; the fetch tool is the one place that does, beside the model fetch tool.

## 1. Corpora, pinned (`twinscribe/bench/corpora.py`)

A catalogue of three corpora, each with its licence, credit line and the exact upstream
revision its bytes come from:

- HarperValleyBank (CC BY 4.0): the repository commit is pinned, and the calls measured are
  listed by id, the first calls in the sorted order of their ids. The list is only ever
  appended to. Each call has four files: the agent channel, the caller channel, the
  transcript and the metadata.
- AMI ES2002a (CC BY 4.0): the headset mix and one array microphone from the corpus mirror,
  and, from the manual annotation release, the meeting map and the words and segments files
  of the four speakers.
- LibriSpeech test-other (CC BY 4.0): the archive is large, so a selection rule pins what is
  extracted: the first chapter folders in sorted order, the first utterances of each, plus
  each chapter's transcript file.

The store is a root folder with one folder per corpus and a lock, `corpora.lock.json`,
recording every file's digest, size, URL, archive member, revision and licence. The store
reader reports which corpora are complete; the verifier digests every file against the lock
and reports verified, mismatch, unpinned or missing, as the model store does.

## 2. References (`twinscribe/bench/references.py`)

A reference is what an item's annotation says: speakers, segments (start, end, speaker,
text), words (text, speaker, and times when the annotation has them) and a statement of how
good the word times are: annotated by the corpus, spread evenly inside each segment, or
absent. Builders:

- HarperValleyBank: every segment's human transcript (never the machine transcript) with its
  offset and duration in the recording; word times spread. The two channels are mixed so
  that both end together, the shorter padded with silence at the front; the offset minus
  start of a role's segments is the delay of that role's channel and is checked against the
  channel lengths.
- AMI: timed words from the words files, punctuation and vocal sounds left out; transcriber
  segments from the segments files; agent letters mapped to global speaker names through the
  meeting map. A window cuts words wholly inside it and clips segments to it, shifting both
  to start at zero.
- LibriSpeech: one reader, the transcript line lower-cased, no word times.

## 3. Items (`twinscribe/bench/items.py`)

An item is one 16 kHz mono WAV and its reference. Ids carry the upstream ids: `hvb/<sid>`,
`ami/ES2002a/<headset|sdm>/<start>-<end>` and `.../full`, `librispeech/<utterance>`. Items
are built into a work folder through the audio module's decoder; a WAV already there is
reused.

## 4. Scoring (`twinscribe/bench/scoring.py`)

- Transcript: word error rate on raw and normalised tokens with the substitution, deletion and
  insertion split, chunked character error rate, per-speaker recall on normalised tokens, and
  onset offsets (median and 90th percentile) only where the reference word times are
  annotated.
- Diarization: error rate with its split at collars 0.25 s and 0, overlap included, and the
  Jaccard error rate. Labels beyond the mapping's limit are folded: the largest by time are
  kept and the rest become one label, and the fold is noted.
- Review: the marks at the design's rule from the published and detector words, scored
  against the reference words; flagged approximate when the word times are spread.
- Pooling: counts summed across items, never rates averaged.

## 5. Report (`twinscribe/bench/report.py`)

Rendered from the results document alone: accuracy pooled per corpus group and variant, the
per-item detail, speaker labelling, the review list, and speed. In the speed table every row
whose verdict says the machine was busy, or that has no verdict, is excluded from the pooled
figures and counted. The limits of the corpora (design section 8) and the licences and
credits close the report.

## 6. Tools

- `tools/fetch_corpora.py --root ROOT [--only KEY ...] | --verify`: downloads files and archive
  members, applies the selection rule, records the lock, skips anything already present with
  a matching digest.
- `tools/bench.py --out FOLDER [--corpora ROOT] [--models ROOT] [--items PATTERN ...]
  [--variants ...] [--device ...] [--threads N] [--rescore]`: builds the items, runs the
  variants (the transducer at the production preset; each Whisper conversion at the
  production preset through CTranslate2 on the plan's device; the diarizer by threshold and
  with the reference speaker count; the review list from the transducer and the turbo
  detector), takes a load snapshot before and after every engine call, keeps every engine
  output as JSON so a run resumes and can be re-scored, and writes `results.json` and
  `report.md`.

## 7. Tests

Hand-derived values for every scoring wrapper, the alignment tie-break kept in mind when
choosing them; the channel mixing and window arithmetic on arrays; the parsers on small
annotation strings; the selection rule on a fake member list; the lock statuses on a
temporary store; the report's exclusion count. No test needs the network or an engine.
