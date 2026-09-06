"""Tests for twinscribe.metrics.speakers: per-speaker recall from an alignment."""

import random

import pytest

from twinscribe.metrics.speakers import per_speaker_recall

# Two speakers. A says eleven words in two turns, B says four in between. The hypothesis
# drops every one of B's words and gets one of A's wrong (here -> there). B's words do not
# recur in A's turns, so the alignment cannot recover one of them from an A token.
REF_TOKENS = "good morning how are you fine thanks and yourself great to see you all here".split()
REF_SPEAKERS = ["A"] * 5 + ["B"] * 4 + ["A"] * 6
HYP_TOKENS = "good morning how are you great to see you all there".split()


def test_every_deletion_falls_on_one_speaker() -> None:
    result = per_speaker_recall(REF_TOKENS, REF_SPEAKERS, HYP_TOKENS)
    assert list(result) == ["A", "B"]

    a = result["A"]
    assert (a.n_tokens, a.recovered, a.substituted, a.deleted) == (11, 10, 1, 0)
    assert a.recall == pytest.approx(10 / 11)

    b = result["B"]
    assert (b.n_tokens, b.recovered, b.substituted, b.deleted) == (4, 0, 0, 4)
    assert b.recall == 0.0


def test_overall_rate_hides_what_per_speaker_shows() -> None:
    # 5 errors over 15 words reads as a third; the per-speaker view says B is gone.
    result = per_speaker_recall(REF_TOKENS, REF_SPEAKERS, HYP_TOKENS)
    total_errors = sum(r.substituted + r.deleted for r in result.values())
    assert total_errors == 5
    assert result["B"].recall == 0.0
    assert result["A"].recall > 0.9


def test_insertions_are_not_attributed() -> None:
    hyp = "good morning how are you well fine thanks and yourself great to see you all here".split()
    result = per_speaker_recall(REF_TOKENS, REF_SPEAKERS, hyp)
    assert sum(r.n_tokens for r in result.values()) == len(REF_TOKENS)
    assert all(r.recall == 1.0 for r in result.values())


def test_length_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError):
        per_speaker_recall(["a", "b"], ["A"], ["a", "b"])


def test_empty_reference() -> None:
    assert per_speaker_recall([], [], ["x"]) == {}


def test_counts_partition_the_reference() -> None:
    # Property: per speaker, recovered + substituted + deleted == n_tokens, and the token
    # counts sum to the reference length, whatever the hypothesis.
    rng = random.Random(6)
    for _ in range(100):
        n = rng.randint(0, 20)
        ref = [rng.choice("abc") for _ in range(n)]
        speakers = [rng.choice("XY") for _ in range(n)]
        hyp = [rng.choice("abc") for _ in range(rng.randint(0, 20))]
        result = per_speaker_recall(ref, speakers, hyp)
        assert set(result) == set(speakers)
        assert sum(r.n_tokens for r in result.values()) == n
        for r in result.values():
            assert r.recovered + r.substituted + r.deleted == r.n_tokens
            assert r.n_tokens == speakers.count(r.speaker)
            assert r.recall is not None and 0.0 <= r.recall <= 1.0
