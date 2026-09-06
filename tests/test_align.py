"""Tests for twinscribe.metrics.align: edit operations, word error counts and chunked CER."""

import math
import random

import pytest

from twinscribe.metrics.align import Op, cer_chunked, edit_ops, levenshtein_distance, word_errors


def _naive_distance(a: list[str], b: list[str]) -> int:
    """Textbook Levenshtein for cross-checking the vectorised table on small inputs."""
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j - 1] + (a[i - 1] != b[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1]


def _assert_total(ref: list[str], hyp: list[str], ops: list[Op]) -> None:
    """Every reference and hypothesis index appears exactly once, in increasing order."""
    ref_idx = [op.ref for op in ops if op.ref is not None]
    hyp_idx = [op.hyp for op in ops if op.hyp is not None]
    assert ref_idx == list(range(len(ref)))
    assert hyp_idx == list(range(len(hyp)))
    for op in ops:
        assert op.kind in {"equal", "sub", "del", "ins"}
        assert (op.kind == "ins") == (op.ref is None)
        assert (op.kind == "del") == (op.hyp is None)
        if op.kind == "equal":
            assert ref[op.ref] == hyp[op.hyp]
        if op.kind == "sub":
            assert ref[op.ref] != hyp[op.hyp]


# A 10-word reference with exactly one substitution (quick -> quack), one deletion (jumps)
# and one insertion (very). The deletion and the insertion are separated by three matching
# words, so no alignment of equal cost can trade them for a pair of substitutions: the
# S/D/I split (1, 1, 1) is the unique optimum, not a tie-break artefact.
REF10 = "the quick brown fox jumps over the lazy dog tonight".split()
HYP10 = "the quack brown fox over the lazy very dog tonight".split()


def test_one_of_each_error_on_ten_words() -> None:
    errors, ops = word_errors(REF10, HYP10)
    assert (errors.sub, errors.dele, errors.ins) == (1, 1, 1)
    assert errors.hits == 8
    assert errors.n_ref == 10
    assert errors.n_hyp == 10
    assert errors.wer == pytest.approx(0.3)
    _assert_total(REF10, HYP10, ops)
    assert [op.kind for op in ops] == [
        "equal", "sub", "equal", "equal", "del", "equal", "equal", "equal", "ins", "equal", "equal",
    ]
    assert ops[1] == Op("sub", 1, 1)
    assert ops[4] == Op("del", 4, None)
    assert ops[8] == Op("ins", None, 7)


def test_deletion_and_insertion_are_directional() -> None:
    # Missing "c" on the hypothesis side is a deletion; the reverse pairing is an insertion.
    errors, _ = word_errors(["a", "b", "c", "d"], ["a", "b", "d"])
    assert (errors.sub, errors.dele, errors.ins) == (0, 1, 0)
    assert errors.wer == pytest.approx(0.25)

    errors, _ = word_errors(["a", "b", "d"], ["a", "b", "c", "d"])
    assert (errors.sub, errors.dele, errors.ins) == (0, 0, 1)
    assert errors.wer == pytest.approx(1 / 3)


def test_identical_sequences_have_zero_error() -> None:
    tokens = "no change at all here".split()
    errors, ops = word_errors(tokens, list(tokens))
    assert errors.wer == 0.0
    assert (errors.hits, errors.sub, errors.dele, errors.ins) == (5, 0, 0, 0)
    assert all(op.kind == "equal" for op in ops)


def test_empty_hypothesis_is_all_deletions() -> None:
    errors, ops = word_errors(["a", "b", "c"], [])
    assert errors.wer == 1.0
    assert errors.dele == 3
    assert [op.kind for op in ops] == ["del"] * 3
    _assert_total(["a", "b", "c"], [], ops)


def test_empty_reference_cases() -> None:
    errors, ops = word_errors([], ["x", "y"])
    assert errors.wer is None
    assert errors.ins == 2
    _assert_total([], ["x", "y"], ops)

    errors, ops = word_errors([], [])
    assert errors.wer == 0.0
    assert ops == []


def test_tie_break_prefers_substitution_over_deletion_and_insertion() -> None:
    # "x a" against "a x" costs 2 either as (del x, eq a, ins x), (ins a, eq x, del a) or
    # (sub, sub); the documented rule takes the diagonal first, so two substitutions.
    ops = edit_ops(["x", "a"], ["a", "x"])
    assert [op.kind for op in ops] == ["sub", "sub"]

    # "a b" against "c": cost 2 as (sub a->c, del b) or (del a, sub b->c). Walking back
    # from the last cell, the diagonal is preferred over the vertical move, so the
    # substitution lands on the last reference token.
    ops = edit_ops(["a", "b"], ["c"])
    assert ops == [Op("del", 0, None), Op("sub", 1, 0)]


def test_edit_ops_is_total_on_random_inputs() -> None:
    rng = random.Random(0)
    for _ in range(200):
        ref = [rng.choice("abc") for _ in range(rng.randint(0, 12))]
        hyp = [rng.choice("abc") for _ in range(rng.randint(0, 12))]
        ops = edit_ops(ref, hyp)
        _assert_total(ref, hyp, ops)
        cost = sum(1 for op in ops if op.kind != "equal")
        assert cost == _naive_distance(ref, hyp)
        assert cost == levenshtein_distance(ref, hyp)


def test_distance_properties() -> None:
    # Symmetry of the total; identity; bounds. The S/D/I split itself is not mirrored when
    # the sides swap, because the tie-break prefers substitution in whichever direction it
    # runs, but the excess of deletions over insertions is fixed by the length difference.
    rng = random.Random(1)
    for _ in range(200):
        ref = [rng.choice("abcd") for _ in range(rng.randint(0, 10))]
        hyp = [rng.choice("abcd") for _ in range(rng.randint(0, 10))]
        fwd, _ = word_errors(ref, hyp)
        rev, _ = word_errors(hyp, ref)
        assert fwd.sub + fwd.dele + fwd.ins == rev.sub + rev.dele + rev.ins
        assert fwd.dele - fwd.ins == len(ref) - len(hyp)
        assert rev.dele - rev.ins == len(hyp) - len(ref)
        total = fwd.sub + fwd.dele + fwd.ins
        assert abs(len(ref) - len(hyp)) <= total <= max(len(ref), len(hyp))
        assert fwd.hits + fwd.sub + fwd.dele == len(ref)
        assert fwd.hits + fwd.sub + fwd.ins == len(hyp)
        same, _ = word_errors(ref, list(ref))
        assert same.wer == 0.0


def test_word_errors_on_longer_input() -> None:
    # A few hundred tokens with scattered errors, checked against the naive distance so the
    # vectorised row update is exercised on more than toy sizes.
    rng = random.Random(2)
    ref = [rng.choice(["alpha", "beta", "gamma", "delta", "zeta"]) for _ in range(300)]
    hyp = list(ref)
    for _ in range(30):
        k = rng.randrange(len(hyp))
        action = rng.choice(["sub", "del", "ins"])
        if action == "sub":
            hyp[k] = "zeta"
        elif action == "del":
            del hyp[k]
        else:
            hyp.insert(k, "eta")
    errors, ops = word_errors(ref, hyp)
    _assert_total(ref, hyp, ops)
    assert errors.sub + errors.dele + errors.ins == _naive_distance(ref, hyp)


# Chunked CER. Reference of twelve words; the hypothesis changes one letter of "seven" and
# drops "twelve". The first six words form an anchor (six equal ops, at least five); the
# four equal words after the substitution are too few to anchor, so the second chunk runs
# from "seven" to the end. Chunk distance: 1 (e -> a) + 7 (delete "twelve" and its space)
# = 8. Reference characters: 51 letters + 11 spaces = 62.
CER_REF = "one two three four five six seven eight nine ten eleven twelve".split()
CER_HYP = "one two three four five six sevan eight nine ten eleven".split()


def test_cer_chunked_matches_whole_string_distance_on_clean_anchors() -> None:
    _, ops = word_errors(CER_REF, CER_HYP)
    assert [op.kind for op in ops] == ["equal"] * 6 + ["sub"] + ["equal"] * 4 + ["del"]
    ref_chars = len(" ".join(CER_REF))
    assert ref_chars == 62
    whole = levenshtein_distance(list(" ".join(CER_REF)), list(" ".join(CER_HYP)))
    assert whole == 8
    assert cer_chunked(CER_REF, CER_HYP, ops) == pytest.approx(8 / 62)
    assert cer_chunked(CER_REF, CER_HYP, ops) == pytest.approx(whole / ref_chars)


def test_cer_chunked_single_chunk_equals_whole_string() -> None:
    # With an anchor run longer than any equal run, everything is one chunk.
    _, ops = word_errors(CER_REF, CER_HYP)
    whole = levenshtein_distance(list(" ".join(CER_REF)), list(" ".join(CER_HYP)))
    assert cer_chunked(CER_REF, CER_HYP, ops, anchor_run=100) == pytest.approx(whole / 62)


def test_cer_chunked_edge_cases() -> None:
    _, ops = word_errors(CER_REF, list(CER_REF))
    assert cer_chunked(CER_REF, list(CER_REF), ops) == 0.0
    _, ops = word_errors([], [])
    assert cer_chunked([], [], ops) == 0.0
    _, ops = word_errors([], ["x"])
    assert cer_chunked([], ["x"], ops) == math.inf
    with pytest.raises(ValueError):
        cer_chunked(CER_REF, CER_HYP, edit_ops(CER_REF, CER_HYP), anchor_run=0)


def test_cer_chunked_is_an_upper_bound_on_random_inputs() -> None:
    rng = random.Random(3)
    words = ["cat", "cart", "cast", "dog", "dot", "do", "bird"]
    for _ in range(100):
        ref = [rng.choice(words) for _ in range(rng.randint(1, 14))]
        hyp = [rng.choice(words) for _ in range(rng.randint(0, 14))]
        _, ops = word_errors(ref, hyp)
        ref_chars = len(" ".join(ref))
        whole = levenshtein_distance(list(" ".join(ref)), list(" ".join(hyp)))
        chunked = cer_chunked(ref, hyp, ops, anchor_run=2)
        assert chunked >= whole / ref_chars - 1e-12
        assert chunked >= 0.0
