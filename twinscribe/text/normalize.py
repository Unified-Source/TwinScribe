"""English text normalisation and the two tokenisers used for word error rate.

`normalize` wraps the vendored Whisper English normaliser and is versioned so that a score
records which normaliser produced it. `tokenize_raw` keeps words as written, lower-cased and
stripped of punctuation, and `tokenize_norm` tokenises the normalised text.
"""

from __future__ import annotations

import unicodedata

from ._whisper_normalizer import EnglishTextNormalizer

# Bump when the vendored normaliser or the tokenisers change behaviour, so that scores
# computed under different rules are never compared as if they were the same.
NORMALISER_VERSION = "v1"

# The normaliser loads its spelling table from disk once; one shared instance is enough
# because it holds no per-call state.
_ENGLISH = EnglishTextNormalizer()

# Typographic apostrophe folded to the ASCII apostrophe before punctuation stripping, so that
# "don't" and "don’t" tokenise identically.
_APOSTROPHE_VARIANTS = {"’": "'", "‘": "'"}


def normalize(text: str) -> str:
    """Return the English-normalised form of `text`.

    Lower case, contractions expanded, fillers removed, spelled-out numbers converted to
    digits (a lone "one" stays a word, by the upstream rule), British spellings mapped to
    American, and all symbols and punctuation dropped. Surrounding whitespace is stripped;
    interior runs of whitespace are single spaces.
    """
    return _ENGLISH(text).strip()


def tokenize_raw(text: str) -> list[str]:
    """Split `text` into raw scoring tokens.

    Lower case; every punctuation character removed except the apostrophe; the underscore
    treated as a separator so that a spelled acronym written "C_D_" yields the tokens "c"
    and "d", the same tokens the normaliser produces for it; then split on whitespace.
    Symbols such as currency signs are not punctuation and are kept as written.
    """
    out: list[str] = []
    for ch in text.lower():
        ch = _APOSTROPHE_VARIANTS.get(ch, ch)
        if ch == "_":
            out.append(" ")
        elif ch == "'":
            out.append(ch)
        elif unicodedata.category(ch).startswith("P"):
            continue
        else:
            out.append(ch)
    return "".join(out).split()


def tokenize_norm(text: str) -> list[str]:
    """Normalise `text` with `normalize` and split on whitespace."""
    return normalize(text).split()
