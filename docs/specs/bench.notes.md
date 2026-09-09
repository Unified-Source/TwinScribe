# Notes: the bench

Companion to `bench.md`. Records what the implementation chose where the specification and
the design left room, what was observed on the first run, and what was deliberately not done.

## Choices

- The telephone set is the first 24 calls of the corpus in the sorted order of their ids,
  pinned in `corpora.py`. The design's own 24 calls are not identified anywhere, so the
  section 2 table can be reproduced in shape (three engines, one pooled telephone set of a
  comparable size) but not on the same calls; a difference is a difference in the sample
  until the original ids are known. Widening the set appends ids and never reorders them.
- The channel rule was verified on the corpus before it was coded: in every call inspected
  the caller's segments have offset equal to start, and the agent's offset minus start equals
  the caller channel length minus the agent channel length, to the millisecond. The mixer
  pads the shorter channel at the front and the item builder checks the transcript's implied
  delay against the channel lengths; a disagreement over 50 ms is written into the item's
  notes rather than corrected.
- HarperValleyBank word times are spread evenly inside each segment because the human
  transcript carries none (the machine transcript has word times and is never used). Every
  review figure on that set is therefore marked approximate, in the same way the design
  qualified its own dropped-word counts.
- The meeting reference uses the corpus's transcriber segments for speaker turns and its
  forced-alignment word times for the words; punctuation entries (zero length, `punc="true"`)
  and vocal sounds are dropped. The 780 to 960 s window holds 626 words over four speakers,
  which matches the design's note that it is the densest speech.
- The read-speech selection is a rule applied to the archive's member list at fetch time
  (first two chapter folders in string order, ten utterances each) and the lock pins what
  came out; item identity is the utterance id, so a changed rule adds or removes items and
  redefines none.
- Diarization scoring folds surplus labels: the exhaustive speaker mapping stops at six a
  side, and the threshold clustering can return more than that on a two-voice recording.
  The largest labels by time are kept and the rest become one label; the fold is recorded in
  the scores and the raw count is reported beside it.
- Every engine output is written as JSON before it is scored, so an interrupted bench resumes
  where it stopped and `--rescore` re-renders without decoding. The cost is one model load
  per item per variant; keeping models loaded across items would cut a first pass by minutes
  and is left for the engine wrappers to offer.
- The plan chooses devices as it does for a run: the detector on the CUDA device the library
  sees, the transducer and the speaker models on the processor with the processor build of
  sherpa-onnx. A second device is selected by restricting what the runtime can see, which
  the bench does not do itself.

## Deviations from the specification

- The report pools counts, never rates, and excludes contended rows from the speed pool; a
  row with no verdict at all (the review variant, which times nothing) is counted with the
  excluded rows so that the count is honest about what the pool lacks.
- `tools/bench.py` names the review variant after the two engines it is built from
  (transducer published, turbo checking) rather than exposing the detector choice; the other
  detector variants are scored as transcripts only.

## Not done, and why

- The full-meeting items are not in the first pass's item filter; at twenty minutes each
  they are the throughput items. A second pass into the same output folder ran them, with the
  cached engine outputs of the first pass reused.
- The full-size Whisper conversion is a variant (`whisper-large`) outside the default set, so
  a first pass costs the design's three engines only; the second pass added it.
- Bootstrap confidence intervals over the pooled counts (design section 7) are not rendered;
  the per-item counts are in the results document for that.

## First run, observed

Forty-eight items, seven variants, the detector conversions in float16 on a CUDA device, the
transducer and the speaker models on the processor with eight threads. Every timed cell
carried a contention verdict of busy (another process held the machine throughout), so the
speed pool is empty by the report's rule and nothing here is a speed figure.

- The pinned telephone set pools to 2,341 reference words on normalised tokens, the count the
  design gives for its own 24 calls; the sets are almost certainly the same. On it the turbo
  conversion scores 6.4 per hundred with 150 errors (41 deleted, 59 substituted, 50 inserted;
  the design: 36, 64, 50), the transducer 12.6 (173 deleted against the design's 165) and
  the full-size conversion 6.4. Distil-Whisper scores 7.8 against the design's 9.0, the one
  material difference; a processor-only pass in int8, the design's configuration, is the
  first thing to compare it with.
- The review list at the design's rule: 60 marks, 59 on speech, 18.5 per cent of the audio to
  listen to (the design: 62, 61, 18). The dropped-word coverage under spread word times is
  167 of 254 and is not comparable with the design's 141 of 154, which counted against the
  reference text; the figure is marked approximate in the report.
- The meeting window reproduces the inversion: transducer 16.2 on the headset mix against
  23.3 and 22.4 for the two Whisper conversions; on the far-field microphone Distil-Whisper
  goes from 22.4 to 24.9. The vanishing participant is visible in the per-speaker recall:
  the turbo conversion recovered 2 per cent of one speaker's words on the headset window at
  a normal-looking overall rate, and the transducer recovered none of another speaker's on
  the far-field microphone.
- Speaker labelling with the count supplied: 14.9 per cent error on the headset window (the
  design: 15) with two of the four speakers unmatched, the degenerate-cluster failure the
  design describes; by threshold, 14 labels on the window and over a hundred on the whole
  meeting. On the telephone set the count-supplied error pools to 13.8 per cent at the
  0.25 s collar against the design's 32; the design does not record its collar or pooling,
  so this stays a difference in method until it does.
- Whole meetings (1,273 s each): transducer 18.4 per hundred on the headset mix and 42.8 on
  the far-field microphone; the review list on the far-field meeting raises 77 marks, 59 on
  speech, covering 469 of 517 dropped words for 47 per cent of the audio. The rule was tuned
  on telephone calls; far-field multi-speaker audio needs its own thresholds.
- Onset offsets against the corpus alignment: transducer median 0.06 s, 90th percentile 0.15
  to 0.17 s; turbo median 0.04 to 0.05 s, 90th percentile 0.31 to 0.51 s.
- Read speech, 20 utterances: 1.3 to 2.7 per hundred, as a smoke test should read.
- Processor-only pass in int8, the design's own arm, over the telephone set: turbo 6.5 per
  hundred (34 deleted, 67 substituted, 50 inserted; the design: 36, 64, 50), transducer 12.6
  (the same output as the device pass, as it must be), Distil-Whisper 7.7. The distil
  difference survives the change of precision and device, so it is not that; the fetched
  conversion is the upstream repository's `main` at fetch time, its model file's digest is in
  `models.lock.json`, and the design records no digest for its own. Review list on this arm:
  63 marks, 62 on speech, 19.3 per cent of the audio (the design: 62, 61, 18).
