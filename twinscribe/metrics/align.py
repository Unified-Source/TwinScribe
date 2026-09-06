"""Token alignment by unit-cost Levenshtein distance, word error counts and chunked CER.

The alignment keeps a full backtrace so that substitutions, deletions and insertions are
reported separately; deletion is the published engine's characteristic failure and a single
error rate hides it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Backpointer codes stored per cell of the dynamic-programming table.
_DIAG = 0  # from (i - 1, j - 1): equal or substitution
_UP = 1  # from (i - 1, j): deletion of reference token i - 1
_LEFT = 2  # from (i, j - 1): insertion of hypothesis token j - 1


@dataclass(frozen=True)
class Op:
    """One alignment step.

    `kind` is "equal", "sub", "del" or "ins". `ref` indexes the reference token list and is
    None for "ins"; `hyp` indexes the hypothesis token list and is None for "del".
    """

    kind: str
    ref: int | None
    hyp: int | None


@dataclass(frozen=True)
class WordErrors:
    """Counts from one alignment; `wer` is the usual ratio over the reference length."""

    n_ref: int
    n_hyp: int
    hits: int
    sub: int
    dele: int
    ins: int

    @property
    def wer(self) -> float | None:
        """(sub + dele + ins) / n_ref.

        0.0 when both sides are empty; None when the reference is empty but the hypothesis
        is not, because no finite rate describes that case.
        """
        if self.n_ref == 0:
            return None if self.ins > 0 else 0.0
        return (self.sub + self.dele + self.ins) / self.n_ref


def _intern(ref: Sequence[str], hyp: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Map both token sequences to integer ids under one shared table."""
    ids: dict[str, int] = {}
    ref_ids = np.fromiter((ids.setdefault(t, len(ids)) for t in ref), dtype=np.int64, count=len(ref))
    hyp_ids = np.fromiter((ids.setdefault(t, len(ids)) for t in hyp), dtype=np.int64, count=len(hyp))
    return ref_ids, hyp_ids


def _next_row(prev: np.ndarray, mismatch: np.ndarray, i: int, offsets: np.ndarray) -> np.ndarray:
    """Compute row i of the Levenshtein table from row i - 1.

    `mismatch[j - 1]` is 1 where reference token i - 1 differs from hypothesis token j - 1.
    The diagonal and vertical moves are vectorised directly. The horizontal (insertion)
    move depends on the cell to the left within the same row; because every insertion
    costs exactly 1, cur[j] = min over k <= j of (pre[k] + j - k), which is a running
    minimum of (pre[k] - k) plus j and is computed with one accumulate.
    """
    pre = np.empty(prev.shape[0], dtype=np.int32)
    pre[0] = i
    np.minimum(prev[:-1] + mismatch, prev[1:] + 1, out=pre[1:])
    return np.minimum.accumulate(pre - offsets) + offsets


def levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Unit-cost Levenshtein distance between two sequences, without a backtrace."""
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    a_ids, b_ids = _intern(a, b)
    offsets = np.arange(len(b) + 1, dtype=np.int32)
    prev = offsets.copy()
    for i in range(1, len(a) + 1):
        mismatch = (b_ids != a_ids[i - 1]).astype(np.int32)
        prev = _next_row(prev, mismatch, i, offsets)
    return int(prev[-1])


def edit_ops(ref: list[str], hyp: list[str]) -> list[Op]:
    """Align `hyp` to `ref` and return the alignment as a list of `Op`, in reference order.

    Unit-cost Levenshtein with a full backtrace; the table of backpointers is
    (len(ref) + 1) * (len(hyp) + 1) bytes, which is acceptable for the sizes scored here.

    Tie-break rule. The backtrace walks from cell (len(ref), len(hyp)) to (0, 0). At each
    cell, among the predecessor moves that achieve the cell's optimal cost, the diagonal
    move is taken first (an "equal" op when the tokens match, otherwise "sub"), then the
    vertical move ("del", consuming a reference token), then the horizontal move ("ins",
    consuming a hypothesis token). When tokens match the diagonal is always optimal, so a
    matching pair is never reported as a substitution. The rule changes only how ties are
    split between S, D and I, never the total edit count, and it is fixed so that two runs
    on the same input report the same split.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return [Op("ins", None, j) for j in range(m)]
    if m == 0:
        return [Op("del", i, None) for i in range(n)]

    ref_ids, hyp_ids = _intern(ref, hyp)
    back = np.empty((n + 1, m + 1), dtype=np.uint8)
    back[0, :] = _LEFT
    back[:, 0] = _UP

    offsets = np.arange(m + 1, dtype=np.int32)
    prev = offsets.copy()
    for i in range(1, n + 1):
        mismatch = (hyp_ids != ref_ids[i - 1]).astype(np.int32)
        diag = prev[:-1] + mismatch
        up = prev[1:] + 1
        cur = _next_row(prev, mismatch, i, offsets)
        row = cur[1:]
        # Preference order on ties: diagonal, then vertical, then horizontal.
        back[i, 1:] = np.where(row == diag, _DIAG, np.where(row == up, _UP, _LEFT))
        prev = cur

    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i, j]
        if move == _DIAG:
            kind = "equal" if ref_ids[i - 1] == hyp_ids[j - 1] else "sub"
            ops.append(Op(kind, i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == _UP:
            ops.append(Op("del", i - 1, None))
            i -= 1
        else:
            ops.append(Op("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def word_errors(ref: list[str], hyp: list[str]) -> tuple[WordErrors, list[Op]]:
    """Align and count; returns the counts and the alignment they were counted from."""
    ops = edit_ops(ref, hyp)
    hits = sum(1 for op in ops if op.kind == "equal")
    sub = sum(1 for op in ops if op.kind == "sub")
    dele = sum(1 for op in ops if op.kind == "del")
    ins = sum(1 for op in ops if op.kind == "ins")
    return WordErrors(len(ref), len(hyp), hits, sub, dele, ins), ops


def _anchor_mask(ops: Sequence[Op], anchor_run: int) -> list[bool]:
    """True for every op inside a run of at least `anchor_run` consecutive "equal" ops."""
    mask = [False] * len(ops)
    start = 0
    while start < len(ops):
        if ops[start].kind != "equal":
            start += 1
            continue
        end = start
        while end < len(ops) and ops[end].kind == "equal":
            end += 1
        if end - start >= anchor_run:
            for k in range(start, end):
                mask[k] = True
        start = end
    return mask


def cer_chunked(ref: list[str], hyp: list[str], ops: list[Op], anchor_run: int = 5) -> float:
    """Character error rate summed over the chunks between word-level anchors.

    Tokens are joined with single spaces on each side. Runs of at least `anchor_run`
    consecutive "equal" ops are anchors; every maximal stretch of ops between anchors (and
    before the first or after the last) is a chunk. For each chunk, the character
    Levenshtein distance between its reference text and its hypothesis text is computed,
    with each token carrying its following space so that the chunk texts and the anchor
    texts concatenate exactly to the full strings. The sum of the chunk distances is divided
    by the number of characters in the space-joined reference.

    The chunk sum is an upper bound on the whole-string character distance: the anchors are
    identical substrings on both sides, so editing each chunk independently and leaving the
    anchors untouched is one valid edit script for the whole string. It is exact when the
    anchors sit where a character-optimal alignment would put them; a run of five or more
    matching words is in practice always part of such an alignment, so on real transcripts
    the two agree.

    Returns 0.0 when both sides are empty and `math.inf` when the reference is empty but the
    hypothesis is not.
    """
    if anchor_run < 1:
        raise ValueError("anchor_run must be at least 1")
    n_ref_chars = len(" ".join(ref))
    if n_ref_chars == 0:
        return 0.0 if len(hyp) == 0 else math.inf

    anchored = _anchor_mask(ops, anchor_run)
    total = 0
    k = 0
    while k < len(ops):
        if anchored[k]:
            k += 1
            continue
        ref_chars: list[str] = []
        hyp_chars: list[str] = []
        while k < len(ops) and not anchored[k]:
            op = ops[k]
            if op.ref is not None:
                ref_chars.extend(ref[op.ref])
                ref_chars.append(" ")
            if op.hyp is not None:
                hyp_chars.extend(hyp[op.hyp])
                hyp_chars.append(" ")
            k += 1
        total += levenshtein_distance(ref_chars, hyp_chars)
    return total / n_ref_chars
