"""Tests for twinscribe.metrics.diarization: DER with its split, the speaker mapping, JER."""

import random

import pytest

from twinscribe.metrics.diarization import MAX_SPEAKERS, der, jer


def test_identical_inputs_score_zero() -> None:
    ref = [(0.0, 10.0, "A"), (10.0, 20.0, "B"), (25.0, 30.0, "A")]
    for collar in (0.0, 0.25):
        res = der(ref, list(ref), collar=collar)
        assert res.der == 0.0
        assert (res.miss, res.false_alarm, res.confusion) == (0.0, 0.0, 0.0)
        assert res.mapping == {"A": "A", "B": "B"}
        assert res.unmatched_ref == ()
        assert (res.n_ref_speakers, res.n_hyp_speakers) == (2, 2)
        assert jer(ref, list(ref), collar=collar) == 0.0
    assert der(ref, list(ref), collar=0.0).scored_ref_s == pytest.approx(25.0)


def test_swapped_labels_score_zero() -> None:
    ref = [(0.0, 10.0, "A"), (10.0, 20.0, "B")]
    hyp = [(0.0, 10.0, "B"), (10.0, 20.0, "A")]
    res = der(ref, hyp, collar=0.0)
    assert res.der == 0.0
    assert res.mapping == {"A": "B", "B": "A"}
    assert jer(ref, hyp, collar=0.0) == 0.0


def test_absent_reference_speaker_is_a_miss_and_is_named() -> None:
    # C speaks 5 s of 25 s of reference speech; the hypothesis never labels that time.
    ref = [(0.0, 10.0, "A"), (10.0, 20.0, "B"), (20.0, 25.0, "C")]
    hyp = [(0.0, 10.0, "x"), (10.0, 20.0, "y")]
    res = der(ref, hyp, collar=0.0)
    assert res.miss == pytest.approx(5.0 / 25.0)
    assert res.false_alarm == 0.0
    assert res.confusion == 0.0
    assert res.der == pytest.approx(0.2)
    assert res.unmatched_ref == ("C",)
    assert "C" not in res.mapping
    assert res.mapping == {"A": "x", "B": "y"}
    # JER: A and B perfect, C unmapped counts 1.0 -> (0 + 0 + 1) / 3.
    assert jer(ref, hyp, collar=0.0) == pytest.approx(1.0 / 3.0)


def test_collar_removes_two_collars_per_interior_boundary() -> None:
    # One speaker for 10 s has boundaries at 5 and 15 only; each removes `collar` of speech
    # on the inside. Splitting the same 10 s between two speakers adds one interior boundary
    # at 10, and that boundary removes `collar` from each side: exactly 2 * collar more.
    collar = 0.25
    one = [(5.0, 15.0, "A")]
    two = [(5.0, 10.0, "A"), (10.0, 15.0, "B")]
    assert der(one, list(one), collar=0.0).scored_ref_s == pytest.approx(10.0)
    assert der(one, list(one), collar=collar).scored_ref_s == pytest.approx(10.0 - 2 * collar)
    assert der(two, list(two), collar=collar).scored_ref_s == pytest.approx(10.0 - 4 * collar)
    removed_by_interior = der(one, list(one), collar=collar).scored_ref_s - der(two, list(two), collar=collar).scored_ref_s
    assert removed_by_interior == pytest.approx(2 * collar)


def test_collar_excludes_hypothesis_speech_near_reference_boundaries() -> None:
    # The hypothesis overshoots by 0.1 s at both ends: false alarm 0.2 / 10 without a collar,
    # nothing once the 0.25 s collar swallows the overshoot.
    ref = [(5.0, 15.0, "A")]
    hyp = [(4.9, 15.1, "a")]
    assert der(ref, hyp, collar=0.0).false_alarm == pytest.approx(0.02)
    assert der(ref, hyp, collar=0.25).der == 0.0


def test_more_hypothesis_speakers_than_reference_maps_injectively() -> None:
    # B's 10 s is split into q (6 s) and r (4 s); the mapping takes q and the rest is confusion.
    ref = [(0.0, 10.0, "A"), (10.0, 20.0, "B")]
    hyp = [(0.0, 10.0, "p"), (10.0, 16.0, "q"), (16.0, 20.0, "r")]
    res = der(ref, hyp, collar=0.0)
    assert res.mapping == {"A": "p", "B": "q"}
    assert len(set(res.mapping.values())) == len(res.mapping)
    assert set(res.mapping.values()) <= {"p", "q", "r"}
    assert (res.n_ref_speakers, res.n_hyp_speakers) == (2, 3)
    assert res.confusion == pytest.approx(4.0 / 20.0)
    assert res.miss == 0.0
    assert res.false_alarm == 0.0
    assert res.der == pytest.approx(0.2)
    assert res.unmatched_ref == ()
    # JER for B: intersection 6, union 10 -> 0.4; A perfect -> mean 0.2.
    assert jer(ref, hyp, collar=0.0) == pytest.approx(0.2)


def test_skip_overlap_removes_the_overlapped_span() -> None:
    # A and B overlap from 5 to 10. The denominator is reference speaker-time, so the 5 s
    # overlapped span carries 2 * 5 = 10 speaker-seconds: 20 with overlap included, 10 once
    # the overlapped intervals are skipped. The hypothesis hears only A there, so the miss
    # of 5 s (one speaker short over 5 s) disappears with the span.
    ref = [(0.0, 10.0, "A"), (5.0, 15.0, "B")]
    hyp = [(0.0, 10.0, "a"), (10.0, 15.0, "b")]
    with_overlap = der(ref, hyp, collar=0.0, skip_overlap=False)
    without = der(ref, hyp, collar=0.0, skip_overlap=True)
    assert with_overlap.scored_ref_s == pytest.approx(20.0)
    assert without.scored_ref_s == pytest.approx(10.0)
    assert with_overlap.scored_ref_s - without.scored_ref_s == pytest.approx(2 * 5.0)
    assert with_overlap.miss == pytest.approx(5.0 / 20.0)
    assert with_overlap.der == pytest.approx(0.25)
    assert without.der == 0.0


# Four reference speakers; D talks for 10 s of 310. The hypothesis absorbs D into C's
# cluster and satisfies a count of four with a 2 s degenerate cluster inside B's turn.
ABSORBED_REF = [(0.0, 100.0, "A"), (100.0, 200.0, "B"), (200.0, 300.0, "C"), (300.0, 310.0, "D")]
ABSORBED_HYP = [
    (0.0, 100.0, "h1"),
    (100.0, 150.0, "h2"),
    (150.0, 152.0, "h4"),
    (152.0, 200.0, "h2"),
    (200.0, 310.0, "h3"),
]


def test_jer_exposes_an_absorbed_speaker_that_der_hides() -> None:
    # DER (collar 0): every second has exactly one speaker on each side, so no miss and no
    # false alarm; confusion is the 10 s of D under h3 plus the 2 s of B under h4:
    # 12 / 310 = 0.0387. Mapping A->h1, B->h2, C->h3; D shares no time with h4 and is
    # left unmapped.
    res = der(ABSORBED_REF, ABSORBED_HYP, collar=0.0)
    assert res.mapping == {"A": "h1", "B": "h2", "C": "h3"}
    assert res.unmatched_ref == ("D",)
    assert res.miss == 0.0
    assert res.false_alarm == 0.0
    assert res.confusion == pytest.approx(12.0 / 310.0)
    assert res.der == pytest.approx(12.0 / 310.0)

    # JER: A 0; B 1 - 98/100 = 0.02; C 1 - 100/110 = 0.0909; D unmapped 1.0; mean 0.2777.
    expected_jer = (0.0 + (1 - 98 / 100) + (1 - 100 / 110) + 1.0) / 4
    assert jer(ABSORBED_REF, ABSORBED_HYP, collar=0.0) == pytest.approx(expected_jer)
    assert expected_jer > 0.27
    assert jer(ABSORBED_REF, ABSORBED_HYP, collar=0.0) > 5 * res.der

    # The ordering survives the default collar.
    assert jer(ABSORBED_REF, ABSORBED_HYP) > 5 * der(ABSORBED_REF, ABSORBED_HYP).der


def test_der_components_sum_and_bounds() -> None:
    # Property: the split sums to the total, every component is non-negative, and JER lies
    # in [0, 1], on random segmentations.
    rng = random.Random(4)
    for _ in range(50):
        ref = _random_segments(rng, "ABC", 6)
        hyp = _random_segments(rng, "xyzw", 6)
        for skip in (False, True):
            res = der(ref, hyp, collar=0.1, skip_overlap=skip)
            assert res.miss >= 0.0 and res.false_alarm >= 0.0 and res.confusion >= -1e-9
            assert res.der == pytest.approx(res.miss + res.false_alarm + res.confusion)
            assert len(set(res.mapping.values())) == len(res.mapping)
        assert 0.0 <= jer(ref, hyp, collar=0.1) <= 1.0 + 1e-9


def test_der_is_invariant_under_hypothesis_relabelling() -> None:
    rng = random.Random(5)
    for _ in range(30):
        ref = _random_segments(rng, "ABC", 6)
        hyp = _random_segments(rng, "xyz", 6)
        table = {"x": "s1", "y": "s2", "z": "s3"}
        relabelled = [(s, e, table[label]) for s, e, label in hyp]
        assert der(ref, hyp).der == pytest.approx(der(ref, relabelled).der)
        assert jer(ref, hyp) == pytest.approx(jer(ref, relabelled))
        assert der(ref, list(ref)).der == 0.0


def test_too_many_speakers_is_an_error() -> None:
    labels = [chr(ord("A") + k) for k in range(MAX_SPEAKERS + 1)]
    ref = [(float(k), float(k + 1), label) for k, label in enumerate(labels)]
    with pytest.raises(ValueError, match="at most"):
        der(ref, [(0.0, 1.0, "x")])
    with pytest.raises(ValueError, match="at most"):
        der([(0.0, 1.0, "A")], ref)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        der([(5.0, 4.0, "A")], [])
    with pytest.raises(ValueError):
        der([(0.0, 1.0, "A")], [], collar=-0.1)


def test_empty_sides() -> None:
    assert der([], [], collar=0.0).der == 0.0
    assert der([(0.0, 1.0, "A")], [], collar=0.0).miss == 1.0
    assert der([], [(0.0, 1.0, "x")], collar=0.0).der == float("inf")
    assert jer([], [(0.0, 1.0, "x")]) == 0.0


def _random_segments(rng: random.Random, labels: str, count: int) -> list[tuple[float, float, str]]:
    out = []
    for _ in range(count):
        start = rng.uniform(0.0, 30.0)
        out.append((start, start + rng.uniform(0.5, 8.0), rng.choice(labels)))
    return out
