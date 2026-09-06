"""Scoring metrics for transcripts and speaker labels.

Word alignment with a full substitution, deletion and insertion split; chunked character
error rate; diarization error rate with its split beside Jaccard error rate; per-speaker
recall; and word timestamp offsets. Every function takes plain lists and returns records.
"""

from .align import Op, WordErrors, cer_chunked, edit_ops, levenshtein_distance, word_errors
from .diarization import MAX_SPEAKERS, DerResult, Seg, der, jer
from .speakers import SpeakerRecall, per_speaker_recall
from .timing import TimingResult, hypothesis_gaps, timestamp_offsets, words_in_gaps

__all__ = [
    "MAX_SPEAKERS",
    "DerResult",
    "Op",
    "Seg",
    "SpeakerRecall",
    "TimingResult",
    "WordErrors",
    "cer_chunked",
    "der",
    "edit_ops",
    "hypothesis_gaps",
    "jer",
    "levenshtein_distance",
    "per_speaker_recall",
    "timestamp_offsets",
    "word_errors",
    "words_in_gaps",
]
