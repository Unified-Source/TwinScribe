"""Tests for the scene pass: spans and gaps, windows, levels, classification from tagger
events, the decision to set words aside, merging, word filtering, and the analysis of a
synthetic recording with a fake tagger and without one."""

from __future__ import annotations

import numpy as np
import pytest

from twinscribe import scenes
from twinscribe.engines.base import Segment, Transcript, Word
from twinscribe.scenes import (
    Event,
    Scene,
    analyse,
    classify,
    gaps_between,
    level_dbfs,
    merge_scenes,
    merge_spans,
    seconds_by_kind,
    should_suppress,
    split_windows,
    without_segments,
    words_outside,
)

RATE = 16000


def ev(*pairs: tuple[str, float]) -> tuple[Event, ...]:
    return tuple(Event(name, prob) for name, prob in pairs)


def utterance(start: float, end: float, text: str) -> Segment:
    tokens = text.split()
    step = (end - start) / max(1, len(tokens))
    words = tuple(Word(token, start + i * step, start + (i + 1) * step) for i, token in enumerate(tokens))
    return Segment(start, end, text, words)


def test_merge_spans_and_gaps() -> None:
    assert merge_spans([(5, 6), (1, 3), (2, 4), (6, 7)]) == [(1.0, 4.0), (5.0, 7.0)]
    assert merge_spans([(3, 3), (2, 1)]) == []
    assert gaps_between([(3.0, 6.0), (7.0, 12.0)], 30.0) == [(0.0, 3.0), (12.0, 30.0)]   # 6 to 7 is too short
    assert gaps_between([(3.0, 6.0)], 30.0, min_gap_s=4.0) == [(6.0, 30.0)]
    assert gaps_between([], 5.0) == [(0.0, 5.0)]
    assert gaps_between([], 1.0) == []
    assert gaps_between([(0.0, 40.0)], 30.0) == []                                       # utterance past the end
    with pytest.raises(ValueError):
        gaps_between([], 5.0, min_gap_s=-1.0)


def test_split_windows_into_equal_pieces() -> None:
    assert split_windows(0.0, 10.0) == [(0.0, 10.0)]
    pieces = split_windows(0.0, 25.0)
    assert len(pieces) == 3 and pieces[0][0] == 0.0 and pieces[-1][1] == 25.0
    assert all(abs((e - s) - 25.0 / 3) < 1e-9 for s, e in pieces)
    assert split_windows(4.0, 4.0) == []
    assert split_windows(2.0, 5.0, window_s=1.0) == [(2.0, 3.0), (3.0, 4.0), (4.0, 5.0)]
    with pytest.raises(ValueError):
        split_windows(0.0, 1.0, window_s=0.0)


def test_level_in_decibels() -> None:
    quiet = np.zeros(RATE * 2, dtype=np.float32)
    assert level_dbfs(quiet, 0.0, 2.0) == scenes.QUIET_FLOOR_DBFS
    tone = (0.5 * np.sin(np.linspace(0.0, 2000.0, RATE * 2))).astype(np.float32)
    assert level_dbfs(tone, 0.0, 2.0) == pytest.approx(-9.03, abs=0.1)                  # 0.5 / sqrt(2)
    assert level_dbfs(tone, 5.0, 6.0) == scenes.QUIET_FLOOR_DBFS                          # past the end
    assert level_dbfs(np.full(RATE, 10 ** (-55 / 20), dtype=np.float32), 0.0, 1.0) == pytest.approx(-55.0, abs=0.01)


def test_classify_by_level_then_by_events() -> None:
    assert classify(-60.0, None) == ("silence", "", 1.0)
    assert classify(-60.0, ev(("Silence", 0.3), ("White noise", 0.2))) == ("silence", "", 0.3)
    assert classify(-20.0, None) == ("sound", "", 0.0)
    assert classify(-20.0, ev(("Speech synthesizer", 0.9), ("Speech", 0.5)))[0] is None
    assert classify(-20.0, ev(("Speech", 0.31),))[0] is None
    assert classify(-20.0, ev(("Music", 0.82), ("Synthesizer", 0.45), ("Speech", 0.05))) == ("music", "", 0.82)
    assert classify(-20.0, ev(("Synthesizer", 0.6), ("Speech", 0.1))) == ("music", "Synthesizer", 0.6)
    assert classify(-20.0, ev(("Video game music", 0.4),)) == ("music", "Video game music", 0.4)
    assert classify(-26.0, ev(("Waterfall", 0.6), ("Stream", 0.33), ("Pink noise", 0.19))) == ("noise", "Waterfall", 0.6)
    assert classify(-30.0, ev(("White noise", 0.35), ("Noise", 0.16))) == ("noise", "White noise", 0.35)
    assert classify(-30.0, ev(("Noise", 0.4),)) == ("noise", "", 0.4)
    assert classify(-30.0, ev(("Music", 0.4), ("Rain", 0.5))) == ("noise", "Rain", 0.5)          # noise outranks music
    assert classify(-20.0, ev(("Siren", 0.8), ("Speech", 0.02))) == ("sound", "Siren", 0.8)
    assert classify(-20.0, ev(("Siren", 0.1),)) == ("sound", "", 0.1)
    assert classify(-20.0, ()) == ("sound", "", 0.0)


def test_words_are_set_aside_only_with_length_and_confident_non_speech() -> None:
    music = ev(("Music", 0.8), ("Speech", 0.05))
    assert should_suppress(music, 3.0) and not should_suppress(music, 1.0)
    assert not should_suppress(None, 3.0)
    assert not should_suppress(ev(("Music", 0.8), ("Speech", 0.2)), 3.0)                # speech is present
    assert not should_suppress(ev(("Music", 0.45), ("Speech", 0.05)), 3.0)              # not confident
    assert should_suppress(ev(("Waterfall", 0.6),), 2.0)
    assert not should_suppress(ev(("Siren", 0.9),), 3.0)                                # other sound keeps its words


def test_merge_scenes_joins_neighbours_of_one_kind() -> None:
    merged = merge_scenes(
        [
            Scene(0.0, 5.0, "music", "", 0.6),
            Scene(5.02, 10.0, "music", "Synthesizer", 0.8),
            Scene(10.0, 12.0, "noise"),
            Scene(12.5, 14.0, "noise"),
            Scene(20.0, 22.0, "music"),
        ]
    )
    assert [(s.start, s.end, s.kind) for s in merged] == [(0.0, 10.0, "music"), (10.0, 12.0, "noise"), (12.5, 14.0, "noise"), (20.0, 22.0, "music")]
    assert merged[0].probability == 0.8 and merged[0].label == ""                       # the longer part names it
    assert merged[0].duration == 10.0


def test_words_outside_and_without_segments() -> None:
    words = [Word("a", 0.0, 1.0), Word("b", 4.0, 5.0), Word("c", 9.0, 9.8)]
    kept = words_outside(words, [Scene(3.0, 6.0, "music"), Scene(10.0, 12.0, "silence")])
    assert [w.text for w in kept] == ["a", "c"]
    assert words_outside(words, []) == words
    first = Segment(0.0, 1.0, "a", (words[0],))
    second = Segment(4.0, 5.0, "b", (words[1],))
    transcript = Transcript("e", "m", "p", (first, second), 10.0, 0.1, 0.2)
    assert without_segments(transcript, [second]).segments == (first,)
    assert without_segments(transcript, []) is transcript
    assert without_segments(transcript, [second]).engine == "e"


def test_seconds_by_kind_in_fixed_order() -> None:
    totals = seconds_by_kind([Scene(0.0, 2.0, "noise"), Scene(5.0, 8.0, "silence"), Scene(9.0, 10.0, "noise")])
    assert list(totals) == ["silence", "noise"] and totals["noise"] == 3.0 and totals["silence"] == 3.0
    assert seconds_by_kind([]) == {}


def _recording() -> tuple[np.ndarray, list[Segment]]:
    audio_s = 60.0
    samples = np.zeros(int(audio_s * RATE), dtype=np.float32)
    rng = np.random.default_rng(1)

    def paint(start: float, end: float, level_db: float) -> None:
        count = int((end - start) * RATE)
        samples[int(start * RATE): int(start * RATE) + count] = (rng.standard_normal(count) * 10 ** (level_db / 20)).astype(np.float32)

    paint(0.0, 60.0, -70.0)          # room tone throughout
    paint(5.0, 10.0, -20.0)          # speech
    paint(12.0, 24.0, -18.0)         # a music passage nobody speaks over
    paint(26.0, 31.0, -20.0)         # speech
    paint(31.5, 36.0, -22.0)         # an utterance the voice detector passed that is music
    paint(45.0, 50.0, -20.0)         # speech
    utterances = [
        utterance(5.0, 10.0, "good afternoon everyone thank you"),
        utterance(26.0, 31.0, "before we start a question"),
        utterance(31.5, 36.0, "la la la la"),
        utterance(45.0, 50.0, "let us move on"),
    ]
    return samples, utterances


def test_analyse_with_a_fake_tagger_and_without() -> None:
    samples, utterances = _recording()
    seen: list[list[tuple[float, float]]] = []

    def overlap(s: float, e: float, a: float, b: float) -> float:
        return max(0.0, min(e, b) - max(s, a))

    def tag(regions):
        seen.append(list(regions))
        out = []
        for s, e in regions:
            if overlap(s, e, 12.0, 24.0) > 0.5 * (e - s):
                out.append(ev(("Music", 0.8), ("Synthesizer", 0.4)))
            elif overlap(s, e, 31.5, 36.0) > 0.5 * (e - s):
                out.append(ev(("Music", 0.7), ("Speech", 0.05)))
            else:
                out.append(ev(("Speech", 0.9),))
        return out

    analysis = analyse(samples, utterances, 60.0, tag)
    # Gaps 0 to 5, 10 to 26 (two windows), 36 to 45 and 50 to 60, then the four utterances.
    assert len(seen) == 1 and len(seen[0]) == 9
    assert [(round(s.start, 3), round(s.end, 3), s.kind) for s in analysis.scenes] == [
        (0.0, 5.0, "silence"),
        (10.0, 26.0, "music"),
        (31.5, 36.0, "music"),
        (36.0, 45.0, "silence"),
        (50.0, 60.0, "silence"),
    ]
    assert [u.text for u in analysis.suppressed] == ["la la la la"]
    assert analysis.tagged and analysis.regions_tagged == 9
    totals = seconds_by_kind(analysis.scenes)
    assert totals["silence"] == pytest.approx(24.0) and totals["music"] == pytest.approx(20.5)
    assert analysis.settings["silence_dbfs"] == scenes.SILENCE_DBFS

    # Without a tagger the level alone decides: a loud pause is sound, a quiet one silence, and
    # no utterance loses its words.
    plain = analyse(samples, utterances, 60.0, None)
    assert [(s.start, s.end, s.kind) for s in plain.scenes] == [
        (0.0, 5.0, "silence"), (10.0, 26.0, "sound"), (36.0, 45.0, "silence"), (50.0, 60.0, "silence"),
    ]
    assert plain.suppressed == () and not plain.tagged and plain.regions_tagged == 0

    with pytest.raises(ValueError):
        analyse(samples, utterances, 60.0, lambda regions: [])


def test_short_utterances_are_never_tagged() -> None:
    samples, _ = _recording()
    regions_seen: list[tuple[float, float]] = []

    def tag(regions):
        regions_seen.extend(regions)
        return [ev(("Music", 0.9),) for _ in regions]

    analysis = analyse(samples, [utterance(5.0, 6.0, "hi")], 60.0, tag)
    assert (5.0, 6.0) not in regions_seen and analysis.suppressed == ()
    assert all(kind in ("silence", "music") for kind in (s.kind for s in analysis.scenes))
