"""Per-speaker word recall from a token alignment and the reference's speaker labels.

The overall error rate can read as normal while one participant's words are produced by
nobody; attributing every substitution and deletion to the speaker of the reference token
makes that visible.
"""

from __future__ import annotations

from dataclasses import dataclass

from .align import edit_ops


@dataclass(frozen=True)
class SpeakerRecall:
    """Token counts for one reference speaker and the fraction recovered as "equal"."""

    speaker: str
    n_tokens: int
    recovered: int
    substituted: int
    deleted: int

    @property
    def recall(self) -> float | None:
        """recovered / n_tokens, or None for a speaker with no reference tokens."""
        if self.n_tokens == 0:
            return None
        return self.recovered / self.n_tokens


def per_speaker_recall(ref_tokens: list[str], ref_speakers: list[str], hyp_tokens: list[str]) -> dict[str, SpeakerRecall]:
    """Align with `edit_ops` and attribute each "equal", "sub" and "del" op to the speaker of
    its reference token.

    `ref_speakers[i]` is the speaker of `ref_tokens[i]`. Insertions carry no reference
    token and are not attributed. The result is keyed by speaker in order of first
    appearance in the reference.
    """
    if len(ref_tokens) != len(ref_speakers):
        raise ValueError(f"ref_tokens has {len(ref_tokens)} entries but ref_speakers has {len(ref_speakers)}")
    counts: dict[str, list[int]] = {}
    for speaker in ref_speakers:
        counts.setdefault(speaker, [0, 0, 0, 0])  # tokens, recovered, substituted, deleted
    for op in edit_ops(ref_tokens, hyp_tokens):
        if op.ref is None:
            continue
        row = counts[ref_speakers[op.ref]]
        row[0] += 1
        if op.kind == "equal":
            row[1] += 1
        elif op.kind == "sub":
            row[2] += 1
        elif op.kind == "del":
            row[3] += 1
    return {
        speaker: SpeakerRecall(speaker, n, recovered, substituted, deleted)
        for speaker, (n, recovered, substituted, deleted) in counts.items()
    }
