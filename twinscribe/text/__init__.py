"""Text normalisation and tokenisation for scoring transcripts.

Two token views of the same text are used side by side: raw tokens, which keep the words as
written, and normalised tokens from the vendored Whisper English normaliser, which fold
spelling, numbers and contractions so that surface differences do not count as errors.
"""

from .normalize import NORMALISER_VERSION, normalize, tokenize_norm, tokenize_raw

__all__ = ["NORMALISER_VERSION", "normalize", "tokenize_norm", "tokenize_raw"]
