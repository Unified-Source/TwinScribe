"""Tests for speaker attribution, line building and the speaker summary."""

from __future__ import annotations

import pytest

from tests._fixtures import PUBLISHED, turns, words
from twinscribe.engines.base import Segment, SpeakerTurn, Word
from twinscribe.labelling import (
    UNLABELLED_NAME,
    build_lines,
    default_names,
    display_name,
    label_words,
    smooth_labels,
    summarise_speakers,
)


def word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, prob=None)


def test_smoothing_absorbs_a_one_word_flicker_inside_an_utterance() -> None:
    ws = [word(f"w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(8)]
    utterance = Segment(0.0, 2.35, "", tuple(ws))
    smoothed, changed = smooth_labels(ws, ["A", "A", "A", "B", "A", "A", "A", "A"], [utterance])
    assert smoothed == ["A"] * 8 and changed == 1
    # Two flickers in one utterance are both absorbed.
    smoothed, changed = smooth_labels(ws, ["A", "B", "A", "A", "A", "B", "A", "A"], [utterance])
    assert smoothed == ["A"] * 8 and changed == 2
    # A run of two words is a change of speaker; so is a single long word.
    steady = ["A", "A", "A", "B", "B", "A", "A", "A"]
    assert smooth_labels(ws, steady, [utterance]) == (steady, 0)
    long_words = [word("x", 0.0, 0.2), word("y", 0.2, 0.9), word("z", 0.9, 1.2)]
    assert smooth_labels(long_words, ["A", "B", "A"], [Segment(0.0, 1.2, "", tuple(long_words))]) == (["A", "B", "A"], 0)
    # Different labels on the two sides are a real change, not a flicker.
    mixed = ["A", "A", "A", "B", "C", "C", "C", "C"]
    assert smooth_labels(ws, mixed, [utterance]) == (mixed, 0)


def test_smoothing_leaves_utterance_edges_alone() -> None:
    ws = [word(f"w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(8)]
    utterance = Segment(0.0, 2.35, "", tuple(ws))
    edges = ["B", "A", "A", "A", "A", "A", "A", "B"]
    assert smooth_labels(ws, edges, [utterance]) == (edges, 0)
    # The flicker word is the last word of the first utterance: an edge, left as it is.
    two = [Segment(0.0, 1.15, "", tuple(ws[:4])), Segment(1.2, 2.35, "", tuple(ws[4:]))]
    crossing = ["A", "A", "A", "B", "A", "A", "A", "A"]
    assert smooth_labels(ws, crossing, two) == (crossing, 0)
    # Words no utterance covers are not smoothed at all.
    assert smooth_labels(ws, crossing, []) == (crossing, 0)
    with pytest.raises(ValueError):
        smooth_labels(ws, ["A"], [utterance])
    with pytest.raises(ValueError):
        smooth_labels(ws, crossing, two, min_run_words=0)


def test_words_inside_turns_take_their_label() -> None:
    labels = label_words(words(PUBLISHED), turns())
    assert labels[:7] == ["speaker_00"] * 7
    assert labels[7:11] == ["speaker_01"] * 4
    assert labels[11:15] == ["speaker_00"] * 4
    assert labels[15:] == ["speaker_01"] * 3


def test_largest_overlap_wins_and_ties_go_to_the_earlier_turn() -> None:
    two = (SpeakerTurn(0.0, 2.0, "A"), SpeakerTurn(1.5, 4.0, "B"))
    assert label_words([word("x", 1.0, 3.0)], two) == ["B"]
    assert label_words([word("x", 1.0, 2.5)], two) == ["A"]


def test_gap_words_attach_to_a_nearby_turn_or_inherit() -> None:
    result = turns()
    near = word("um", 4.0, 4.3)          # 0.4 s after speaker_00 ends at 3.6
    far = word("erm", 9.0, 9.3)          # 1.3 s after speaker_01 ends at 7.7
    labels = label_words([near, far], result)
    assert labels == ["speaker_00", "speaker_00"]
    labels = label_words(words(PUBLISHED[:7]) + [far] + words(PUBLISHED[7:]), result)
    assert labels[7] == "speaker_00"


def test_leading_unlabelled_words_inherit_the_next_label() -> None:
    late = (SpeakerTurn(5.0, 8.0, "B"),)
    labels = label_words([word("a", 0.5, 0.9), word("b", 5.5, 5.9)], late)
    assert labels == ["B", "B"]


def test_no_turns_leaves_everything_unlabelled() -> None:
    assert label_words(words(PUBLISHED), ()) == [None] * len(PUBLISHED)


def test_negative_attach_distance_rejected() -> None:
    with pytest.raises(ValueError):
        label_words([], (), max_attach_s=-1.0)


def test_lines_split_on_speaker_gap_and_length() -> None:
    ws = words(PUBLISHED)
    labels = label_words(ws, turns())
    lines = build_lines(ws, labels)
    assert [(line.speaker, len(line.words)) for line in lines] == [
        ("speaker_00", 7), ("speaker_01", 4), ("speaker_00", 4), ("speaker_01", 3),
    ]
    assert lines[0].text == "good morning this is the first call"
    assert lines[0].start == 0.5 and lines[0].end == 3.5 and lines[0].duration == pytest.approx(3.0)

    same_speaker = [word("a", 0.0, 0.5), word("b", 2.5, 3.0)]
    assert len(build_lines(same_speaker, ["A", "A"])) == 2        # 2.0 s pause splits
    assert len(build_lines(same_speaker, ["A", "A"], max_gap_s=2.5)) == 1
    many = [word(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(5)]
    assert [len(line.words) for line in build_lines(many, ["A"] * 5, max_words=2)] == [2, 2, 1]


def test_lines_reject_bad_input() -> None:
    with pytest.raises(ValueError):
        build_lines([word("a", 0.0, 1.0)], [])
    with pytest.raises(ValueError):
        build_lines([], [], max_words=0)


def test_lines_partition_the_words_in_order() -> None:
    ws = words(PUBLISHED)
    labels = label_words(ws, turns())
    flattened = [w for line in build_lines(ws, labels) for w in line.words]
    assert flattened == ws


def test_speaker_summary_counts_and_order() -> None:
    ws = words(PUBLISHED)
    lines = build_lines(ws, label_words(ws, turns()))
    summary = summarise_speakers(lines)
    assert [(s.label, s.words) for s in summary] == [("speaker_00", 11), ("speaker_01", 7)]
    assert summary[0].seconds == pytest.approx(3.0 + 1.9)
    assert sum(s.words for s in summary) == len(ws)

    mixed = build_lines(ws[:3], [None, "B", "B"])
    labels = [s.label for s in summarise_speakers(mixed)]
    assert labels == ["B", None]


def test_default_and_display_names() -> None:
    names = default_names(["speaker_01", None, "speaker_00", "speaker_01"])
    assert names == {"speaker_01": "Speaker 1", "speaker_00": "Speaker 2"}
    assert display_name("speaker_00", names) == "Speaker 2"
    assert display_name("speaker_07", names) == "speaker_07"
    assert display_name(None, names) == UNLABELLED_NAME
