"""Tests for twinscribe.metrics.timing: onset offsets over matched words and gap counts."""

import pytest

from twinscribe.metrics.align import edit_ops
from twinscribe.metrics.timing import hypothesis_gaps, timestamp_offsets, words_in_gaps

# Twelve reference words, one per second. The hypothesis substitutes word 5 and inserts an
# extra word after word 8, so eleven words match. Their signed onset offsets alternate in
# sign with magnitudes 0.0, 0.1, ..., 1.0; the substituted word carries a 5.0 s offset that
# must be ignored. Sorted magnitudes 0.0 .. 1.0 in steps of 0.1: median 0.5, and the 90th
# percentile by linear interpolation sits at index 0.9 * 10 = 9, exactly 0.9.
REF_TOKENS = [f"w{k}" for k in range(12)]
HYP_TOKENS = REF_TOKENS[:5] + ["zzz"] + REF_TOKENS[6:9] + ["extra"] + REF_TOKENS[9:]
SIGNED = [0.0, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0]


def _fixture_times() -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    ref_times = [(float(k), k + 0.5) for k in range(12)]
    hyp_times = []
    offsets = iter(SIGNED)
    for token in HYP_TOKENS:
        if token == "zzz":
            hyp_times.append((5.0 + 5.0, 5.0 + 5.5))
        elif token == "extra":
            hyp_times.append((8.6, 8.9))
        else:
            k = int(token[1:])
            d = next(offsets)
            hyp_times.append((k + d, k + d + 0.5))
    return ref_times, hyp_times


def test_timestamp_offsets_fixture() -> None:
    ref_times, hyp_times = _fixture_times()
    ops = edit_ops(REF_TOKENS, HYP_TOKENS)
    assert [op.kind for op in ops].count("equal") == 11
    result = timestamp_offsets(ops, ref_times, hyp_times)
    assert result.n == 11
    assert result.median_abs_s == pytest.approx(0.5)
    assert result.p90_abs_s == pytest.approx(0.9)
    assert list(result.offsets_s) == pytest.approx(SIGNED)


def test_timestamp_offsets_empty() -> None:
    ops = edit_ops(["a"], ["b"])
    result = timestamp_offsets(ops, [(0.0, 1.0)], [(0.0, 1.0)])
    assert result.n == 0
    assert result.median_abs_s is None
    assert result.p90_abs_s is None
    assert result.offsets_s == ()


def test_timestamp_offsets_sign_flips_when_sides_swap() -> None:
    # Property: swapping the two time lists negates every offset and leaves the absolute
    # statistics unchanged.
    ref_times, hyp_times = _fixture_times()
    ops = edit_ops(REF_TOKENS, HYP_TOKENS)
    fwd = timestamp_offsets(ops, ref_times, hyp_times)
    # Swapping sides also swaps the indices in the ops.
    swapped_ops = edit_ops(HYP_TOKENS, REF_TOKENS)
    rev = timestamp_offsets(swapped_ops, hyp_times, ref_times)
    assert rev.n == fwd.n
    assert sorted(rev.offsets_s) == pytest.approx(sorted(-x for x in fwd.offsets_s))
    assert rev.median_abs_s == pytest.approx(fwd.median_abs_s)
    assert rev.p90_abs_s == pytest.approx(fwd.p90_abs_s)


# Hypothesis words cover 0.0-1.0 (two words 0.1 s apart) and 5.0-6.0; the reference runs
# to 8.5. Gaps of at least 2 s: 1.0-5.0 (4.0 s) and the tail 6.0-8.5 (2.5 s). The two tiny
# 0.1 s holes between adjacent words are not gaps.
HYP_TIMES = [(0.0, 0.5), (0.6, 1.0), (5.0, 5.5), (5.6, 6.0)]
REF_TIMES = [
    (0.0, 0.4),  # midpoint 0.2, covered
    (0.7, 0.9),  # midpoint 0.8, covered
    (2.0, 2.4),  # midpoint 2.2, in the 1.0-5.0 gap
    (3.0, 3.5),  # midpoint 3.25, in the 1.0-5.0 gap
    (4.9, 5.2),  # midpoint 5.05, covered by 5.0-5.5
    (5.7, 5.9),  # midpoint 5.8, covered
    (8.0, 8.5),  # midpoint 8.25, in the tail gap
]


def test_hypothesis_gaps_fixture() -> None:
    assert hypothesis_gaps(REF_TIMES, HYP_TIMES, min_gap=2.0) == [(1.0, 5.0), (6.0, 8.5)]
    assert hypothesis_gaps(REF_TIMES, HYP_TIMES, min_gap=0.05) == [(0.5, 0.6), (1.0, 5.0), (5.5, 5.6), (6.0, 8.5)]
    assert hypothesis_gaps(REF_TIMES, HYP_TIMES, min_gap=3.0) == [(1.0, 5.0)]
    assert hypothesis_gaps(REF_TIMES, HYP_TIMES, min_gap=5.0) == []


def test_words_in_gaps_fixture() -> None:
    assert words_in_gaps(REF_TIMES, HYP_TIMES, min_gap=2.0) == 3
    assert words_in_gaps(REF_TIMES, HYP_TIMES, min_gap=3.0) == 2
    assert words_in_gaps(REF_TIMES, HYP_TIMES, min_gap=5.0) == 0


def test_words_in_gaps_counts_the_lead() -> None:
    # The hypothesis starts at 10 s; the reference has two words before that, one just
    # before the first hypothesis word. The lead 1.0-10.0 is a 9 s gap; the tail 10.5-10.9
    # is too short to count.
    hyp = [(10.0, 10.5)]
    ref = [(1.0, 1.5), (9.5, 9.8), (10.1, 10.4), (10.6, 10.9)]
    assert hypothesis_gaps(ref, hyp, min_gap=2.0) == [(1.0, 10.0)]
    assert words_in_gaps(ref, hyp, min_gap=2.0) == 2


def test_words_in_gaps_with_no_hypothesis() -> None:
    ref = [(0.0, 0.5), (1.0, 1.5), (2.0, 2.5)]
    assert words_in_gaps(ref, [], min_gap=2.0) == 3
    assert words_in_gaps(ref, [], min_gap=3.0) == 0
    assert words_in_gaps([], [], min_gap=2.0) == 0
    assert words_in_gaps([], [(0.0, 1.0)], min_gap=0.0) == 0


def test_words_in_gaps_bounds() -> None:
    # Property: never more than the reference word count, never negative, and a larger
    # minimum gap never counts more words.
    counts = [words_in_gaps(REF_TIMES, HYP_TIMES, min_gap=g) for g in (0.0, 0.5, 1.0, 2.0, 3.0, 10.0)]
    assert all(0 <= c <= len(REF_TIMES) for c in counts)
    assert counts == sorted(counts, reverse=True)
    with pytest.raises(ValueError):
        words_in_gaps(REF_TIMES, HYP_TIMES, min_gap=-1.0)
