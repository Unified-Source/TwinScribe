"""Tests for the two-engine review list, its evaluation and the review-set document."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from twinscribe.review import (
    REVIEW_SCHEMA,
    Mark,
    ReviewEvaluation,
    build_review,
    evaluate_review,
    review_set,
    write_review_set,
)

try:
    from twinscribe.engines.base import Segment, Transcript, Word
except ImportError:  # engines package not present yet; equivalent test-only types
    from tests._types_stub import Segment, Transcript, Word


def word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, prob=None)


def transcript(engine: str, model: str, preset: str, words: list[Word], audio_s: float) -> Transcript:
    segment = Segment(
        start=words[0].start if words else 0.0,
        end=words[-1].end if words else 0.0,
        text=" ".join(w.text for w in words),
        words=tuple(words),
        quality={},
    )
    return Transcript(
        engine=engine,
        model=model,
        preset=preset,
        segments=(segment,),
        audio_s=audio_s,
        load_s=0.1,
        transcribe_s=0.5,
        versions={"lib": "1.0"},
        extras={},
    )


# ---------------------------------------------------------------- build_review


def test_single_gap_with_two_detector_words_raises_one_mark():
    published = [word("a", 0.0, 1.0), word("b", 1.0, 2.0), word("c", 4.0, 5.0), word("d", 9.0, 10.0)]
    detector = [word("x", 2.5, 2.8), word("y", 3.0, 3.3), word("z", 6.0, 6.5)]
    marks = build_review(published, detector, audio_s=10.0)
    assert len(marks) == 1
    mark = marks[0]
    assert (mark.span_start, mark.span_end) == (2.0, 4.0)
    assert mark.start == pytest.approx(1.6)
    assert mark.end == pytest.approx(4.4)
    assert mark.detector_words == 2
    assert mark.detector_text == "x y"


def test_detector_word_count_one_below_threshold_raises_nothing():
    published = [word("a", 0.0, 2.0), word("b", 6.0, 10.0)]
    detector = [word("z", 3.0, 3.5)]
    assert build_review(published, detector, audio_s=10.0) == []
    marks = build_review(published, detector, audio_s=10.0, min_detector_words=1)
    assert len(marks) == 1
    assert (marks[0].span_start, marks[0].span_end) == (2.0, 6.0)
    assert marks[0].detector_text == "z"


def test_span_exactly_min_silence_long_qualifies():
    published = [word("a", 0.0, 1.2), word("b", 2.0, 3.0)]
    detector = [word("x", 1.3, 1.5), word("y", 1.6, 1.8)]
    marks = build_review(published, detector, audio_s=3.0)
    assert len(marks) == 1
    assert (marks[0].span_start, marks[0].span_end) == (1.2, 2.0)
    assert build_review(published, detector, audio_s=3.0, min_silence_s=0.81) == []
    shorter = [word("a", 0.0, 1.25), word("b", 2.0, 3.0)]
    assert build_review(shorter, detector, audio_s=3.0) == []


def test_detector_word_touching_span_edge_does_not_fall_in_span():
    published = [word("a", 0.0, 2.0), word("b", 4.0, 6.0)]
    touching = [word("x", 1.5, 2.0), word("y", 3.0, 3.2), word("z", 4.0, 4.5)]
    assert build_review(published, touching, audio_s=6.0) == []
    crossing = [word("x", 1.9, 2.1), word("y", 3.0, 3.2), word("z", 4.0, 4.5)]
    marks = build_review(published, crossing, audio_s=6.0)
    assert len(marks) == 1
    assert marks[0].detector_words == 2
    assert marks[0].detector_text == "x y"


def test_leading_silence_counts_and_start_is_clamped_to_zero():
    published = [word("a", 2.0, 3.0), word("b", 3.0, 10.0)]
    detector = [word("x", 0.2, 0.5), word("y", 0.6, 0.9)]
    marks = build_review(published, detector, audio_s=10.0)
    assert len(marks) == 1
    assert (marks[0].span_start, marks[0].span_end) == (0.0, 2.0)
    assert marks[0].start == 0.0
    assert marks[0].end == pytest.approx(2.4)


def test_trailing_silence_counts_and_end_is_clamped_to_audio():
    published = [word("a", 0.0, 8.0)]
    detector = [word("x", 8.5, 8.8), word("y", 9.0, 9.9)]
    marks = build_review(published, detector, audio_s=10.0)
    assert len(marks) == 1
    assert (marks[0].span_start, marks[0].span_end) == (8.0, 10.0)
    assert marks[0].start == pytest.approx(7.6)
    assert marks[0].end == 10.0


def test_no_published_words_makes_the_whole_recording_one_span():
    detector = [word("x", 1.0, 2.0), word("y", 3.0, 4.0)]
    marks = build_review([], detector, audio_s=5.0)
    assert len(marks) == 1
    assert (marks[0].span_start, marks[0].span_end) == (0.0, 5.0)
    assert (marks[0].start, marks[0].end) == (0.0, 5.0)
    assert marks[0].detector_words == 2


def test_empty_detector_or_empty_audio_raises_nothing():
    published = [word("a", 0.0, 1.0)]
    assert build_review(published, [], audio_s=10.0) == []
    assert build_review([], [word("x", 0.0, 1.0)], audio_s=0.0) == []


def test_overlapping_and_unsorted_published_words_merge_into_one_coverage():
    published = [word("b", 2.0, 5.0), word("a", 0.0, 3.0), word("c", 7.0, 9.0)]
    detector = [word("x", 5.2, 5.4), word("y", 6.0, 6.5)]
    marks = build_review(published, detector, audio_s=9.0)
    assert [(m.span_start, m.span_end) for m in marks] == [(5.0, 7.0)]


def test_published_words_outside_the_audio_are_clamped():
    published = [word("a", -1.0, 1.0), word("b", 8.0, 12.0)]
    detector = [word("x", 2.0, 3.0), word("y", 4.0, 5.0)]
    marks = build_review(published, detector, audio_s=10.0)
    assert [(m.span_start, m.span_end) for m in marks] == [(1.0, 8.0)]


def test_detector_text_is_stripped_and_joined_in_time_order():
    published = [word("a", 0.0, 1.0), word("b", 5.0, 6.0)]
    detector = [word(" world", 3.0, 3.5), word(" hello", 2.0, 2.5), word("  ", 4.0, 4.1)]
    marks = build_review(published, detector, audio_s=6.0)
    assert marks[0].detector_words == 3
    assert marks[0].detector_text == "hello world"


def test_negative_parameters_are_rejected():
    with pytest.raises(ValueError):
        build_review([], [], audio_s=1.0, pad_s=-0.1)
    with pytest.raises(ValueError):
        build_review([], [], audio_s=1.0, min_silence_s=-1.0)


def _random_words(rng: np.random.RandomState, audio_s: float, count: int, prefix: str) -> list[Word]:
    starts = np.sort(rng.uniform(0.0, audio_s, size=count))
    words = []
    for index, start in enumerate(starts):
        length = float(rng.uniform(0.05, 0.6))
        words.append(word(f"{prefix}{index}", float(start), min(audio_s, float(start) + length)))
    return words


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mark_invariants_on_random_fixture(seed: int):
    rng = np.random.RandomState(seed)
    audio_s = 120.0
    published = _random_words(rng, audio_s, 150, "p")
    detector = _random_words(rng, audio_s, 250, "d")
    marks = build_review(published, detector, audio_s)
    previous_end = -1.0
    for mark in marks:
        assert 0.0 <= mark.start <= mark.span_start < mark.span_end <= mark.end <= audio_s
        assert mark.span_start >= previous_end
        previous_end = mark.span_end
        assert mark.span_end - mark.span_start >= 0.8 - 1e-9
        assert mark.detector_words >= 2
        assert mark.start == pytest.approx(max(0.0, mark.span_start - 0.4))
        assert mark.end == pytest.approx(min(audio_s, mark.span_end + 0.4))
        for w in published:
            assert not (w.start < mark.span_end and mark.span_start < w.end)
    unpadded = build_review(published, detector, audio_s, pad_s=0.0)
    assert all(m.start == m.span_start and m.end == m.span_end for m in unpadded)
    stricter = build_review(published, detector, audio_s, min_detector_words=3)
    assert {(m.span_start, m.span_end) for m in stricter} <= {(m.span_start, m.span_end) for m in marks}
    looser = build_review(published, detector, audio_s, min_silence_s=0.4)
    assert {(m.span_start, m.span_end) for m in marks} <= {(m.span_start, m.span_end) for m in looser}


# ------------------------------------------------------------ evaluate_review


REFERENCE = [
    word("r1", 1.0, 2.0),
    word("r2", 3.0, 4.0),
    word("r3", 6.0, 7.0),
    word("r4", 10.0, 11.0),
    word("r5", 15.0, 16.0),
]
PUBLISHED = [word("p1", 1.0, 2.0), word("p2", 3.5, 4.5)]
HAND_MARKS = [
    Mark(start=4.6, end=8.4, span_start=5.0, span_end=8.0, detector_words=2, detector_text="x y"),
    Mark(start=11.6, end=13.4, span_start=12.0, span_end=13.0, detector_words=2, detector_text="q q"),
    Mark(start=14.1, end=17.4, span_start=14.5, span_end=17.0, detector_words=3, detector_text="a b c"),
]


def test_evaluation_reproduces_hand_derived_precision_and_recall():
    # Dropped: r3, r4, r5 (r1 and r2 overlap a published word). Mark 1 covers r3, mark 2 is a
    # false alarm, mark 3 covers r5. Padded windows: 3.8 + 1.8 + 3.3 = 8.9 s of 20 s.
    evaluation = evaluate_review(HAND_MARKS, PUBLISHED, REFERENCE, audio_s=20.0)
    assert evaluation.marks == 3
    assert evaluation.marks_on_speech == 2
    assert evaluation.precision == pytest.approx(2 / 3)
    assert evaluation.dropped_words == 3
    assert evaluation.dropped_covered == 2
    assert evaluation.recall == pytest.approx(2 / 3)
    assert evaluation.audio_to_review_s == pytest.approx(8.9)
    assert evaluation.audio_to_review_fraction == pytest.approx(0.445)


def test_zero_denominators_give_zero():
    no_marks = evaluate_review([], PUBLISHED, REFERENCE, audio_s=20.0)
    assert no_marks.marks == 0
    assert no_marks.precision == 0.0
    assert no_marks.dropped_words == 3
    assert no_marks.recall == 0.0
    assert no_marks.audio_to_review_s == 0.0
    fully_published = [word(w.text, w.start, w.end) for w in REFERENCE]
    nothing_dropped = evaluate_review(HAND_MARKS, fully_published, REFERENCE, audio_s=20.0)
    assert nothing_dropped.dropped_words == 0
    assert nothing_dropped.dropped_covered == 0
    assert nothing_dropped.recall == 0.0
    assert nothing_dropped.marks_on_speech == 2
    assert evaluate_review(HAND_MARKS, PUBLISHED, REFERENCE, audio_s=0.0).audio_to_review_fraction == 0.0


def test_on_speech_uses_the_span_not_the_padded_window():
    reference = [word("r", 4.7, 4.9)]
    marks = [Mark(start=4.6, end=8.4, span_start=5.0, span_end=8.0, detector_words=2, detector_text="x y")]
    evaluation = evaluate_review(marks, [], reference, audio_s=10.0)
    assert evaluation.marks_on_speech == 0
    assert evaluation.dropped_words == 1
    assert evaluation.dropped_covered == 0


def test_overlapping_padded_windows_are_counted_once():
    marks = [
        Mark(start=1.6, end=3.4, span_start=2.0, span_end=3.0, detector_words=2, detector_text=""),
        Mark(start=3.1, end=4.9, span_start=3.5, span_end=4.5, detector_words=2, detector_text=""),
    ]
    evaluation = evaluate_review(marks, [], [], audio_s=10.0)
    assert evaluation.audio_to_review_s == pytest.approx(3.3)
    assert evaluation.audio_to_review_fraction == pytest.approx(0.33)


def test_end_to_end_review_on_a_known_fixture():
    # The publisher drops the words between 4 s and 6 s; the detector hears them.
    reference = [word("one", 1.0, 1.5), word("two", 4.2, 4.6), word("three", 4.8, 5.4), word("four", 8.0, 8.5)]
    published = [word("one", 1.0, 1.5), word("four", 8.0, 8.5)]
    detector = [word("one", 1.0, 1.5), word("two", 4.2, 4.6), word("three", 4.8, 5.4), word("four", 8.0, 8.5)]
    marks = build_review(published, detector, audio_s=10.0)
    # Spans: [0, 1.0] holds one detector word, [1.5, 8.0] holds two, [8.5, 10.0] holds none.
    assert [(m.span_start, m.span_end) for m in marks] == [(1.5, 8.0)]
    evaluation = evaluate_review(marks, published, reference, audio_s=10.0)
    assert (evaluation.marks, evaluation.marks_on_speech, evaluation.precision) == (1, 1, 1.0)
    assert (evaluation.dropped_words, evaluation.dropped_covered, evaluation.recall) == (2, 2, 1.0)
    assert evaluation.audio_to_review_s == pytest.approx(7.3)


@pytest.mark.parametrize("seed", [3, 4])
def test_evaluation_bounds_on_random_fixture(seed: int):
    rng = np.random.RandomState(seed)
    audio_s = 90.0
    reference = _random_words(rng, audio_s, 120, "r")
    published = [w for w in reference if rng.uniform() > 0.3]
    detector = _random_words(rng, audio_s, 200, "d")
    marks = build_review(published, detector, audio_s)
    evaluation = evaluate_review(marks, published, reference, audio_s)
    assert 0.0 <= evaluation.precision <= 1.0
    assert 0.0 <= evaluation.recall <= 1.0
    assert 0.0 <= evaluation.audio_to_review_fraction <= 1.0
    assert evaluation.marks_on_speech <= evaluation.marks == len(marks)
    assert evaluation.dropped_covered <= evaluation.dropped_words <= len(reference)


# ------------------------------------------------------------------ review_set


def _fixture_transcripts() -> tuple[Transcript, Transcript, list[Mark]]:
    published_words = [word("one", 1.0, 1.5), word("four", 8.0, 8.5)]
    detector_words = [word("one", 1.0, 1.5), word("two", 4.2, 4.6), word("three", 4.8, 5.4), word("four", 8.0, 8.5)]
    published = transcript("transducer", "model-p", "vad", published_words, 10.0)
    detector = transcript("whisper", "model-d", "production", detector_words, 10.0)
    marks = build_review(published.words, detector.words, published.audio_s)
    return published, detector, marks


def test_review_set_without_reference_has_null_test_fields():
    published, detector, marks = _fixture_transcripts()
    doc = review_set(published, detector, "sample.wav", marks)
    assert doc["schema"] == REVIEW_SCHEMA
    assert doc["audio"] == "sample.wav"
    assert doc["duration_s"] == 10.0
    assert doc["publisher"] == {"engine": "transducer", "model": "model-p", "preset": "vad"}
    assert doc["detector"] == {"engine": "whisper", "model": "model-d", "preset": "production"}
    assert doc["transcript"] == [{"s": 1.0, "e": 1.5, "w": "one"}, {"s": 8.0, "e": 8.5, "w": "four"}]
    assert len(doc["marks"]) == 1
    entry = doc["marks"][0]
    assert entry["span_start"] == 1.5 and entry["span_end"] == 8.0
    assert entry["start"] == pytest.approx(1.1) and entry["end"] == pytest.approx(8.4)
    assert entry["detector_words"] == 2 and entry["detector_text"] == "two three"
    assert entry["reference_words"] is None
    assert entry["reference_speakers"] is None
    assert doc["evaluation"] is None


def test_review_set_with_reference_fills_test_fields():
    published, detector, marks = _fixture_transcripts()
    reference = [word("one", 1.0, 1.5), word("two", 4.2, 4.6), word("three", 4.8, 5.4), word("four", 8.0, 8.5)]
    speakers = ["A", "B", "B", "A"]
    doc = review_set(published, detector, "sample.wav", marks, reference, speakers)
    entry = doc["marks"][0]
    assert entry["reference_words"] == 2
    assert entry["reference_speakers"] == ["B"]
    expected = asdict(evaluate_review(marks, published.words, reference, published.audio_s))
    assert doc["evaluation"] == expected
    assert doc["evaluation"]["recall"] == 1.0
    without_labels = review_set(published, detector, "sample.wav", marks, reference)
    assert without_labels["marks"][0]["reference_words"] == 2
    assert without_labels["marks"][0]["reference_speakers"] is None
    assert without_labels["evaluation"] == expected


def test_review_set_speakers_in_order_of_first_appearance():
    published, detector, marks = _fixture_transcripts()
    reference = [word("two", 4.2, 4.6), word("three", 4.8, 5.4), word("more", 6.0, 6.3)]
    doc = review_set(published, detector, "sample.wav", marks, reference, ["Z", "A", "Z"])
    assert doc["marks"][0]["reference_words"] == 3
    assert doc["marks"][0]["reference_speakers"] == ["Z", "A"]


def test_review_set_rejects_inconsistent_speakers():
    published, detector, marks = _fixture_transcripts()
    reference = [word("two", 4.2, 4.6)]
    with pytest.raises(ValueError):
        review_set(published, detector, "sample.wav", marks, reference, ["A", "B"])
    with pytest.raises(ValueError):
        review_set(published, detector, "sample.wav", marks, None, ["A"])


def test_review_set_round_trips_through_json(tmp_path):
    published, detector, marks = _fixture_transcripts()
    reference = [word("two", 4.2, 4.6), word("three", 4.8, 5.4)]
    doc = review_set(published, detector, "sample.wav", marks, reference, ["A", "A"])
    target = tmp_path / "review.json"
    write_review_set(doc, target)
    with open(target, encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded == doc
    assert ReviewEvaluation(**loaded["evaluation"]) == evaluate_review(
        marks, published.words, reference, published.audio_s
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["review.json"]
    doc["audio"] = "other.wav"
    write_review_set(doc, target)
    with open(target, encoding="utf-8") as handle:
        assert json.load(handle)["audio"] == "other.wav"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["review.json"]


def test_checking_windows_widen_merge_and_clamp() -> None:
    from twinscribe.review import checking_windows

    words = [word("a", 0.0, 1.0), word("b", 1.5, 2.0), word("c", 5.0, 6.0)]
    # Silent spans: 1.0-1.5 (0.5 s, under the rule), 2.0-5.0 and 6.0-10.0. With a 1.0 s margin
    # the two that count become 1.0-6.0 and 5.0-10.0, which overlap and merge.
    assert checking_windows(words, 10.0) == [(1.0, 10.0)]
    # A 0.2 s margin keeps them apart: 1.8-5.2 and 5.8-10.0.
    assert checking_windows(words, 10.0, margin_s=0.2) == [(1.8, 5.2), (5.8, 10.0)]
    # Touching windows merge: a 0.5 s margin gives 1.5-5.5 and 5.5-10.0.
    assert checking_windows(words, 10.0, margin_s=0.5) == [(1.5, 10.0)]
    # Leading silence counts and the margin clamps at zero; trailing silence clamps at the end.
    late = [word("a", 2.0, 3.0)]
    assert checking_windows(late, 4.0) == [(0.0, 4.0)]
    assert checking_windows(late, 4.0, margin_s=0.0) == [(0.0, 2.0), (3.0, 4.0)]
    # No silence long enough: nothing to decode. No audio: nothing.
    dense = [word("a", 0.0, 4.0), word("b", 4.5, 8.0), word("c", 8.5, 10.0)]
    assert checking_windows(dense, 10.0) == []
    assert checking_windows(words, 0.0) == []
    with pytest.raises(ValueError):
        checking_windows(words, 10.0, margin_s=-1.0)
