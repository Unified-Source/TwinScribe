"""Tests for the bench: the corpus catalogue and lock, the reference builders and their audio
arithmetic, the scoring wrappers with hand-derived values, and the report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from twinscribe.bench import corpora
from twinscribe.bench.corpora import (
    CATALOGUE,
    HVB_CALLS,
    KEY_AMI,
    KEY_HVB,
    KEY_LIBRISPEECH,
    LIBRISPEECH_FOLDER,
    STATUS_MISMATCH,
    STATUS_MISSING,
    STATUS_UNPINNED,
    STATUS_VERIFIED,
    find_corpora,
    librispeech_members,
    lock_entry,
    read_lock,
    spec_for,
    verify_store,
    write_lock,
)
from twinscribe.bench.items import KIND_AMI_MEETING, KIND_HVB_CALL, list_items, safe_name
from twinscribe.bench.references import (
    WORD_TIMES_ANNOTATED,
    WORD_TIMES_NONE,
    WORD_TIMES_SPREAD,
    RefSegment,
    RefWord,
    ami_reference,
    ami_segments,
    ami_speakers,
    ami_words,
    cut_reference,
    cut_window,
    hvb_alignment_notes,
    hvb_delays,
    hvb_reference,
    librispeech_reference,
    librispeech_transcripts,
    mix_channels,
    read_pcm16,
    spread_words,
    write_pcm16,
)
from twinscribe.bench.report import render_report
from twinscribe.bench.scoring import diarization_scores, fold_labels, pooled_word_errors, review_scores, transcript_scores
from twinscribe.engines.base import SpeakerTurn, Word

# --- catalogue and lock ---------------------------------------------------------------------


def test_catalogue_is_consistent() -> None:
    keys = [spec.key for spec in CATALOGUE]
    assert len(keys) == len(set(keys))
    for spec in CATALOGUE:
        assert spec.licence and spec.credit and spec.title and spec.revision
        targets = [file.target for file in spec.files]
        assert len(targets) == len(set(targets)), spec.key
        for file in spec.files:
            assert file.url.startswith("https://"), file.url
            if file.archive is not None:
                assert file.url.endswith(file.archive) and file.member
        if spec.selection is not None:
            assert not spec.files and spec.archive_url
    assert list(HVB_CALLS) == sorted(HVB_CALLS) and len(set(HVB_CALLS)) == len(HVB_CALLS)
    assert len(spec_for(KEY_HVB).files) == 4 * len(HVB_CALLS)
    with pytest.raises(KeyError):
        spec_for("nothing")


def test_librispeech_selection_is_first_chapters_first_utterances() -> None:
    names = []
    for speaker, chapter in (("2033", "164914"), ("1688", "142285"), ("1688", "9"), ("1998", "15444")):
        folder = f"LibriSpeech/test-other/{speaker}/{chapter}"
        names.append(f"{folder}/{speaker}-{chapter}.trans.txt")
        names += [f"{folder}/{speaker}-{chapter}-{i:04d}.flac" for i in range(12)]
    names.append("LibriSpeech/test-other/README.TXT")
    pairs = librispeech_members(names, chapters=2, per_chapter=3)
    members = [member for member, _ in pairs]
    # Sorted chapter folders: 1688/142285 before 1688/9 (string order), both before 1998.
    assert members[0] == "LibriSpeech/test-other/1688/142285/1688-142285.trans.txt"
    assert members[1:4] == [f"LibriSpeech/test-other/1688/142285/1688-142285-{i:04d}.flac" for i in range(3)]
    assert members[4] == "LibriSpeech/test-other/1688/9/1688-9.trans.txt"
    assert len(pairs) == 8
    assert all(target.startswith(LIBRISPEECH_FOLDER + "/") for _, target in pairs)


def test_find_corpora_reports_incomplete_and_present(tmp_path: Path) -> None:
    assert not find_corpora(tmp_path).present
    ami = tmp_path / KEY_AMI
    for file in spec_for(KEY_AMI).files[:3]:
        (ami / file.target).parent.mkdir(parents=True, exist_ok=True)
        (ami / file.target).write_bytes(b"x")
    store = find_corpora(tmp_path)
    assert KEY_AMI in store.incomplete and not store.has(KEY_AMI)
    for file in spec_for(KEY_AMI).files:
        (ami / file.target).parent.mkdir(parents=True, exist_ok=True)
        (ami / file.target).write_bytes(b"x")
    utterances = tmp_path / KEY_LIBRISPEECH / LIBRISPEECH_FOLDER
    utterances.mkdir(parents=True)
    (utterances / "1688-142285-0000.flac").write_bytes(b"x")
    assert KEY_LIBRISPEECH in find_corpora(tmp_path).incomplete
    (utterances / "1688-142285.trans.txt").write_text("1688-142285-0000 HELLO\n")
    store = find_corpora(tmp_path)
    assert store.has(KEY_AMI) and store.has(KEY_LIBRISPEECH)
    assert "present" in store.describe()


def test_lock_verify_statuses(tmp_path: Path) -> None:
    spec = spec_for(KEY_AMI)
    folder = tmp_path / spec.key
    lock: dict = {}
    for file in spec.files:
        path = folder / file.target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file.target.encode())
        lock[f"{spec.key}/{file.target}"] = lock_entry(spec, file, path)
    write_lock(tmp_path, lock)
    assert read_lock(tmp_path) == lock
    first = spec.files[0]
    assert lock[f"{spec.key}/{first.target}"]["sha256"] == hashlib.sha256(first.target.encode()).hexdigest()
    (folder / spec.files[1].target).write_bytes(b"changed")
    (folder / spec.files[2].target).unlink()
    del lock[f"{spec.key}/{spec.files[3].target}"]
    write_lock(tmp_path, lock)
    checks = {c.file: c.status for c in verify_store(tmp_path, keys=(KEY_AMI,))}
    assert checks[spec.files[0].target] == STATUS_VERIFIED
    assert checks[spec.files[1].target] == STATUS_MISMATCH
    assert checks[spec.files[2].target] == STATUS_MISSING
    assert checks[spec.files[3].target] == STATUS_UNPINNED
    assert read_lock(tmp_path / "nowhere") == {}


# --- references -------------------------------------------------------------------------------

HVB_TRANSCRIPT = [
    {"index": 2, "offset_ms": 6890, "start_ms": 4839, "duration_ms": 1140, "speaker_role": "agent", "human_transcript": "my name is elizabeth"},
    {"index": 1, "offset_ms": 3720, "start_ms": 1669, "duration_ms": 2670, "speaker_role": "agent", "human_transcript": "hello this is the bank"},
    {"index": 3, "offset_ms": 12890, "start_ms": 12890, "duration_ms": 330, "speaker_role": "caller", "human_transcript": "hi"},
    {"index": 4, "offset_ms": 14000, "start_ms": 14000, "duration_ms": 1000, "speaker_role": "caller", "human_transcript": ""},
]


def test_hvb_reference_orders_segments_and_spreads_words() -> None:
    ref = hvb_reference("hvb/x", HVB_TRANSCRIPT, audio_s=51.11)
    assert ref.speakers == ("agent", "caller")
    assert [s.text for s in ref.segments] == ["hello this is the bank", "my name is elizabeth", "hi"]
    assert ref.segments[0].start == pytest.approx(3.72) and ref.segments[0].end == pytest.approx(6.39)
    assert ref.word_times == WORD_TIMES_SPREAD
    words = [w for w in ref.words if w.speaker == "agent"][:5]
    assert [w.text for w in words] == ["hello", "this", "is", "the", "bank"]
    assert words[0].start == pytest.approx(3.72) and words[4].end == pytest.approx(6.39)
    assert words[1].start == pytest.approx(3.72 + 2.67 / 5)
    assert ref.text == "hello this is the bank my name is elizabeth hi"
    assert len(ref.timed_words()) == 10


def test_hvb_delays_per_role() -> None:
    assert hvb_delays(HVB_TRANSCRIPT) == {"agent": (2.051,), "caller": (0.0,)}


def test_hvb_alignment_notes_tolerate_rounding_jitter() -> None:
    # The caller channel is 2.051 s longer, so the agent's delay should be 2.051 s.
    assert hvb_alignment_notes({"agent": (2.051,), "caller": (0.0,)}, 2.051) == []
    assert hvb_alignment_notes({"agent": (2.050, 2.051), "caller": (0.0, 0.001)}, 2.051) == []
    notes = hvb_alignment_notes({"agent": (2.051, 3.5), "caller": (0.0,)}, 2.051)
    assert len(notes) == 1 and notes[0].startswith("agent channel delay in the transcript is 3.500 s against 2.051 s")
    # A longer agent channel puts the delay on the caller.
    assert hvb_alignment_notes({"agent": (0.0,), "caller": (1.2,)}, -1.2) == []
    assert hvb_alignment_notes({"agent": (1.2,), "caller": (0.0,)}, -1.2)[0].startswith("agent channel delay")


def test_spread_words_empty_segment() -> None:
    assert spread_words(RefSegment(0.0, 1.0, "a", "   ")) == []


def test_mix_channels_pads_the_shorter_at_the_front_and_clips() -> None:
    agent = np.array([1000, 2000, 3000], dtype=np.int16)
    caller = np.array([10, 20, 30, 40, 50], dtype=np.int16)
    mixed = mix_channels(agent, caller)
    assert mixed.dtype == np.int16
    assert mixed.tolist() == [10, 20, 1030, 2040, 3050]
    loud = np.array([30000, -30000], dtype=np.int16)
    assert mix_channels(loud, loud).tolist() == [32767, -32768]
    assert mix_channels(np.array([], dtype=np.int16), caller).tolist() == caller.tolist()


def test_pcm16_round_trip(tmp_path: Path) -> None:
    samples = (np.sin(np.linspace(0, 20, 800)) * 12000).astype(np.int16)
    path = tmp_path / "tone.wav"
    write_pcm16(path, samples, 8000)
    back, rate = read_pcm16(path)
    assert rate == 8000 and back.tolist() == samples.tolist()


AMI_WORDS_A = (
    '<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>\n'
    '<nite:root nite:id="M.A.words" xmlns:nite="http://nite.sourceforge.net/">\n'
    '  <w nite:id="w0" starttime="77.44" endtime="77.74">Hi</w>\n'
    '  <w nite:id="w1" starttime="77.74" endtime="77.74" punc="true">,</w>\n'
    '  <w nite:id="w2" starttime="77.74" endtime="78.16">I&#39;m</w>\n'
    '  <vocalsound nite:id="v0" starttime="78.2" endtime="78.5" type="laugh"/>\n'
    '  <w nite:id="w3" starttime="780.5" endtime="780.9">Okay</w>\n'
    '  <w nite:id="w4" starttime="959.8" endtime="960.2">late</w>\n'
    "</nite:root>\n"
)
AMI_SEGMENTS_A = (
    '<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>\n'
    '<nite:root nite:id="M.A.segs" xmlns:nite="http://nite.sourceforge.net/">\n'
    '  <segment nite:id="s0" channel="0" transcriber_start="77.408" transcriber_end="80.955"/>\n'
    '  <segment nite:id="s1" channel="0" transcriber_start="779.0" transcriber_end="781.0"/>\n'
    "</nite:root>\n"
)
AMI_MEETINGS = (
    '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
    '<nite:root xmlns:nite="http://nite.sourceforge.net/">\n'
    '  <meeting nite:id="meet_1" observation="ES2001a"><speaker nxt_agent="A" global_name="XXX001"/></meeting>\n'
    '  <meeting nite:id="meet_2" observation="ES2002a">\n'
    '    <speaker nite:id="ES2002a_2" channel="1" nxt_agent="B" global_name="FEE005" role="PM"/>\n'
    '    <speaker nite:id="ES2002a_1" channel="0" nxt_agent="A" global_name="MEE006" role="ID"/>\n'
    "  </meeting>\n"
    "</nite:root>\n"
)


def test_ami_parsers_skip_punctuation_and_vocal_sounds() -> None:
    words = ami_words(AMI_WORDS_A, "MEE006")
    assert [w.text for w in words] == ["Hi", "I'm", "Okay", "late"]
    assert words[0].start == 77.44 and words[0].speaker == "MEE006"
    assert ami_words(AMI_WORDS_A.encode("iso-8859-1"), "A") == [RefWord(w.text, "A", w.start, w.end) for w in words]
    segments = ami_segments(AMI_SEGMENTS_A, "MEE006")
    assert [(s.start, s.end) for s in segments] == [(77.408, 80.955), (779.0, 781.0)]
    assert ami_speakers(AMI_MEETINGS, "ES2002a") == {"B": "FEE005", "A": "MEE006"}
    assert ami_speakers(AMI_MEETINGS, "ES9999z") == {}


def test_ami_reference_window_cuts_and_shifts() -> None:
    words = {"MEE006": ami_words(AMI_WORDS_A, "MEE006")}
    segments = {"MEE006": ami_segments(AMI_SEGMENTS_A, "MEE006")}
    ref = ami_reference("ami/x", words, segments, audio_s=180.0, window=(780.0, 960.0))
    assert ref.word_times == WORD_TIMES_ANNOTATED
    # "Okay" lies inside the window; "late" ends after it and "Hi" precedes it.
    assert [w.text for w in ref.words] == ["Okay"]
    assert ref.words[0].start == pytest.approx(0.5) and ref.words[0].end == pytest.approx(0.9)
    # The segment 779-781 is clipped to 780-781 and shifted to 0-1, carrying the word inside it.
    assert len(ref.segments) == 1
    assert (ref.segments[0].start, ref.segments[0].end, ref.segments[0].text) == (pytest.approx(0.0), pytest.approx(1.0), "Okay")
    assert ref.speakers == ("MEE006",)
    whole = ami_reference("ami/y", words, segments, audio_s=1242.0)
    assert [w.text for w in whole.words] == ["Hi", "I'm", "Okay", "late"]
    assert whole.segments[0].text == "Hi I'm"
    with pytest.raises(ValueError):
        cut_reference([], [], (10.0, 10.0))


def test_cut_window_on_samples() -> None:
    samples = np.arange(100, dtype=np.int16)
    assert cut_window(samples, 10, (2.0, 4.0)).tolist() == list(range(20, 40))
    assert cut_window(samples, 10, (9.0, 20.0)).tolist() == list(range(90, 100))
    with pytest.raises(ValueError):
        cut_window(samples, 10, (20.0, 30.0))


def test_librispeech_reference() -> None:
    transcripts = librispeech_transcripts("1688-142285-0000 THERE WAS A KING\n\n1688-142285-0001 WHO REIGNED\n")
    assert transcripts == {"1688-142285-0000": "THERE WAS A KING", "1688-142285-0001": "WHO REIGNED"}
    ref = librispeech_reference("librispeech/1688-142285-0000", "1688-142285-0000", transcripts["1688-142285-0000"], 3.5)
    assert ref.text == "there was a king" and ref.word_times == WORD_TIMES_NONE
    assert ref.timed_words() == [] and ref.speakers == ("reader",)
    assert ref.segments[0].end == 3.5


# --- items ------------------------------------------------------------------------------------


def test_list_items_follows_the_store(tmp_path: Path) -> None:
    hvb = tmp_path / KEY_HVB
    for file in spec_for(KEY_HVB).files:
        (hvb / file.target).parent.mkdir(parents=True, exist_ok=True)
        (hvb / file.target).write_bytes(b"x")
    ami = tmp_path / KEY_AMI
    for file in spec_for(KEY_AMI).files:
        (ami / file.target).parent.mkdir(parents=True, exist_ok=True)
        (ami / file.target).write_bytes(b"x")
    items = list_items(find_corpora(tmp_path))
    ids = [item.id for item in items]
    assert ids[: len(HVB_CALLS)] == [f"hvb/{sid}" for sid in HVB_CALLS]
    assert ids[len(HVB_CALLS):] == [
        "ami/ES2002a/headset/780-960", "ami/ES2002a/sdm/780-960", "ami/ES2002a/headset/full", "ami/ES2002a/sdm/full",
    ]
    assert {item.kind for item in items} == {KIND_HVB_CALL, KIND_AMI_MEETING}
    assert safe_name("ami/ES2002a/sdm/full") == "ami__ES2002a__sdm__full"
    assert list_items(find_corpora(tmp_path / "empty")) == []


# --- scoring ----------------------------------------------------------------------------------


def _reference(segments: list[tuple[float, float, str, str]], word_times: str = WORD_TIMES_SPREAD):
    from twinscribe.bench.references import Reference

    segs = tuple(RefSegment(s, e, spk, text) for s, e, spk, text in segments)
    words = tuple(w for seg in segs for w in spread_words(seg))
    return Reference("item", "corpus", 10.0, tuple(dict.fromkeys(s.speaker for s in segs)), segs, words, word_times)


def _words(spec: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(text=t, start=s, end=e, prob=None) for t, s, e in spec]


def test_transcript_scores_hand_derived() -> None:
    # Reference: nine words over two speakers. Hypothesis: "quick" substituted, "fox" deleted,
    # "big" inserted; the deletion and the insertion are separated by four matching words, so
    # no alignment of equal cost can trade them for two substitutions. Normalised and raw
    # tokens coincide here.
    ref = _reference([(0.0, 3.0, "a", "the quick brown fox"), (4.0, 8.0, "b", "jumps over the lazy dog")])
    hyp = _words([
        ("the", 0.0, 0.5), ("quack", 0.5, 1.0), ("brown", 1.0, 1.5),
        ("jumps", 4.0, 4.5), ("over", 4.5, 5.0), ("the", 5.0, 5.5), ("lazy", 5.5, 6.0), ("big", 6.0, 6.5), ("dog", 6.5, 7.0),
    ])
    scores = transcript_scores(ref, hyp)
    norm = scores["norm"]
    assert (norm["n_ref"], norm["n_hyp"], norm["sub"], norm["del"], norm["ins"], norm["hits"]) == (9, 9, 1, 1, 1, 7)
    assert norm["wer"] == pytest.approx(3 / 9)
    assert scores["raw"]["wer"] == pytest.approx(3 / 9)
    recall = scores["per_speaker"]
    assert recall["a"]["n_tokens"] == 4 and recall["a"]["recovered"] == 2 and recall["a"]["recall"] == pytest.approx(0.5)
    assert recall["a"]["substituted"] == 1 and recall["a"]["deleted"] == 1
    assert recall["b"]["n_tokens"] == 5 and recall["b"]["recall"] == pytest.approx(1.0)
    assert scores["timing"] is None
    perfect = transcript_scores(ref, _words([(w.text, w.start, w.end) for w in ref.words]))
    assert perfect["norm"]["wer"] == 0.0 and perfect["raw"]["wer"] == 0.0


def test_transcript_scores_timing_when_annotated() -> None:
    ref = _reference([(0.0, 3.0, "a", "one two three")], word_times=WORD_TIMES_ANNOTATED)
    shifted = _words([(w.text, w.start + 0.2, w.end + 0.2) for w in ref.words])
    timing = transcript_scores(ref, shifted)["timing"]
    assert timing["n"] == 3 and timing["median_abs_s"] == pytest.approx(0.2) and timing["p90_abs_s"] == pytest.approx(0.2)


def test_fold_labels_keeps_the_largest() -> None:
    turns = [SpeakerTurn(i * 10.0, i * 10.0 + 10.0 - i, f"s{i}") for i in range(8)]
    folded, found, note = fold_labels(turns, limit=3)
    assert found == 8 and note is not None
    labels = [label for _, _, label in folded]
    assert labels[:2] == ["s0", "s1"] and set(labels[2:]) == {"other"}
    assert fold_labels(turns[:2], limit=3) == ([(0.0, 10.0, "s0"), (10.0, 19.0, "s1")], 2, None)


def test_diarization_scores_identity_and_confusion() -> None:
    ref = _reference([(0.0, 4.0, "a", "x x x x"), (5.0, 9.0, "b", "y y y y")])
    same = [SpeakerTurn(0.0, 4.0, "spk1"), SpeakerTurn(5.0, 9.0, "spk2")]
    scores = diarization_scores(ref, same)
    assert scores["labels_found"] == 2 and scores["der_0"]["der"] == pytest.approx(0.0) and scores["jer"] == pytest.approx(0.0)
    assert scores["der_0.25"]["mapping"] == {"a": "spk1", "b": "spk2"}
    # One label for both speakers: whichever speaker the tie leaves unmapped, four of the eight
    # scored seconds are confusion at collar 0.
    merged = diarization_scores(ref, [SpeakerTurn(0.0, 4.0, "spk1"), SpeakerTurn(5.0, 9.0, "spk1")])
    assert merged["der_0"]["confusion"] == pytest.approx(0.5)
    assert len(merged["der_0"]["unmatched_ref"]) == 1 and merged["der_0"]["unmatched_ref"][0] in ("a", "b")
    assert merged["labels_found"] == 1 and merged["der_0"]["n_hyp_speakers"] == 1
    assert diarization_scores(ref, [])["der_0"] is None


def test_review_scores_finds_a_dropped_span() -> None:
    # The publisher missed "help me" at 4-6 s; the detector heard two words there.
    ref = _reference([(0.0, 2.0, "a", "hello there"), (4.0, 6.0, "b", "help me"), (8.0, 10.0, "a", "goodbye now")])
    published = _words([("hello", 0.0, 1.0), ("there", 1.0, 2.0), ("goodbye", 8.0, 9.0), ("now", 9.0, 10.0)])
    detector = published + _words([("help", 4.0, 5.0), ("me", 5.0, 6.0)])
    scores = review_scores(ref, published, detector)
    assert scores["marks"] == 1 and scores["marks_on_speech"] == 1 and scores["precision"] == 1.0
    assert scores["dropped_words"] == 2 and scores["dropped_covered"] == 2 and scores["recall"] == 1.0
    assert scores["approximate"] is True and scores["evaluated"] is True
    none = review_scores(_reference([(0.0, 2.0, "a", "hello")], word_times=WORD_TIMES_NONE), published, detector)
    assert none["evaluated"] is False and none["marks"] == 1


def test_pooled_word_errors_sums_counts() -> None:
    rows = [{"n_ref": 10, "n_hyp": 9, "hits": 8, "sub": 1, "del": 1, "ins": 0}, {"n_ref": 20, "n_hyp": 21, "hits": 18, "sub": 1, "del": 1, "ins": 2}]
    pooled = pooled_word_errors(rows)
    assert pooled["items"] == 2 and pooled["n_ref"] == 30 and pooled["wer"] == pytest.approx(6 / 30)
    assert pooled_word_errors([])["wer"] is None


# --- report -----------------------------------------------------------------------------------


def test_report_renders_and_counts_contended_rows(tmp_path: Path) -> None:
    ref = _reference([(0.0, 3.0, "a", "the quick brown fox"), (4.0, 6.0, "b", "jumps over")])
    hyp = _words([("the", 0.0, 0.5), ("quick", 0.5, 1.0), ("brown", 1.0, 1.5), ("fox", 1.5, 2.0), ("jumps", 4.0, 4.5), ("over", 5.0, 6.0)])
    scores = transcript_scores(ref, hyp)
    variant = {"kind": "transcript", "scores": scores, "timing": {"audio_s": 10.0, "load_s": 1.0, "transcribe_s": 2.0}}
    results = {
        "schema": "twinscribe.bench.v1",
        "meta": {"produced_utc": "2026-01-01T00:00:00+00:00", "machine": {"processor": "p", "logical_cores": 8, "memory_gb": 16, "os": "o"}, "versions": {"lib": "1"}, "plan": {"lines": ["Detector on cpu"]}, "credits": ["corpus: credit (CC BY 4.0)"]},
        "items": {
            "hvb/aaa": {"audio_s": 10.0, "speakers": ["a", "b"], "notes": ["note one"], "variants": {
                "parakeet": {**variant, "verdict": {"busy": False, "other_load_cores": 0.1}},
                "whisper-turbo": {**variant, "verdict": {"busy": True, "other_load_cores": 3.0}},
                "speakers": {"kind": "diarization", "scores": diarization_scores(ref, [SpeakerTurn(0.0, 3.0, "x"), SpeakerTurn(4.0, 6.0, "y")]), "timing": {"audio_s": 10.0, "load_s": 0.5, "transcribe_s": 1.0}, "verdict": {"busy": None}},
                "review": {"kind": "review", "scores": review_scores(ref, hyp, hyp)},
            }},
            "librispeech/1-2-0000": {"audio_s": 4.0, "speakers": ["reader"], "variants": {"parakeet": {**variant, "verdict": {"busy": False, "other_load_cores": 0.0}}}},
        },
    }
    report = render_report(results)
    assert "| telephone calls (HarperValleyBank) | parakeet | 1 | 6 | 0.0 |" in report
    assert "| read speech (LibriSpeech test-other) | parakeet | 1 | 6 |" in report
    # Both parakeet rows carry the same 10 s cell, so the pool is 20 s of audio in 4 s.
    assert "| parakeet | 2 | 20 | 4.0 | 5.00 | 0 |" in report
    assert "| whisper-turbo | 0 | 0 | 0.0 | - | 1 |" in report
    assert "| speakers | 0 | 0 | 0.0 | - | 1 |" in report
    assert "note one" in report and "corpus: credit" in report and "Limits" in report
    (tmp_path / "report.md").write_text(report, encoding="utf-8")
    assert json.dumps(results)


def test_corpora_default_root_honours_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(corpora.CORPORA_ENV, str(tmp_path))
    assert corpora.default_corpora_root() == tmp_path


def test_bench_has_a_targeted_checking_arm() -> None:
    import importlib.util
    import sys
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location("bench_tool", _Path(__file__).resolve().parent.parent / "tools" / "bench.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("bench_tool", module)
    spec.loader.exec_module(module)
    assert "whisper-turbo-gaps" in module.ALL_VARIANTS and "review-gaps" in module.ALL_VARIANTS
    assert module.TRANSCRIPT_VARIANTS["whisper-turbo-gaps"] == module.KEY_WHISPER_TURBO
    assert module.ALL_VARIANTS.index("whisper-turbo-gaps") < module.ALL_VARIANTS.index("review-gaps")
