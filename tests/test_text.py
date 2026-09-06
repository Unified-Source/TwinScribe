"""Tests for twinscribe.text: the vendored English normaliser and the two tokenisers."""

import pytest

from twinscribe.text import NORMALISER_VERSION, normalize, tokenize_norm, tokenize_raw

# Expected strings were produced by running the vendored normaliser at the upstream commit
# recorded in VENDORED.md and checked by hand against the upstream rules (numbers to digits,
# contractions expanded, fillers dropped, currency and percent folded onto the number,
# British spellings mapped, punctuation removed).
NORMALISE_CASES = [
    # spelled numbers
    ("She has twenty-three apples.", "she has 23 apples"),
    ("She has twenty three apples and one orange.", "she has 23 apples and one orange"),
    ("one hundred and one dalmatians", "101 dalmatians"),
    ("Nineteen sixties music", "1960s music"),
    ("three point one four", "3.14"),
    ("double oh seven", "007"),
    ("Two and a half hours", "2.5 hours"),
    # contractions and perfect tenses
    ("They won't go, you can't stay, and it's fine.", "they will not go you can not stay and it is fine"),
    ("She'd been there; you're right.", "she had been there you are right"),
    ("She's got two thousand five hundred and forty-two dollars.", "she has got $2542"),
    # fillers
    ("Um, they think, uh, you should, hmm, leave.", "they think you should leave"),
    # currency and percentages
    ("It cost twenty dollars and fifty cents.", "it cost $20.50"),
    ("It cost $20 million.", "it cost $20000000"),
    ("About ten percent of them agreed.", "about 10% of them agreed"),
    # spelling and titles
    ("The colour of the neighbourhood was grey.", "the color of the neighborhood was gray"),
    ("Mr Smith met Dr Jones.", "mister smith met doctor jones"),
    # bracketed annotations, punctuation, spacing
    ("[laughter] (inaudible) okay", "okay"),
    ("Hello, world.", "hello world"),
    ("C_D_", "c d"),
    ("  padded   spaces  ", "padded spaces"),
    ("", ""),
]


@pytest.mark.parametrize("text, expected", NORMALISE_CASES)
def test_normalize_fixtures(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_normaliser_version_tag() -> None:
    assert NORMALISER_VERSION == "v1"


def test_compound_number_becomes_digits_but_lone_one_stays_a_word() -> None:
    # The upstream rule writes "one" back as a word when it stands alone, so a transcript
    # that says "one" and a reference that writes "1" normalise to the same token, while
    # a compound number is always digits.
    assert normalize("twenty three") == "23"
    assert normalize("one") == "one"
    assert normalize("1") == "one"
    assert normalize("one one one") == "111"
    assert tokenize_norm("They want one, not twenty one") == ["they", "want", "one", "not", "21"]


def test_normalize_output_shape() -> None:
    # Property: output is lower case with single interior spaces and no surrounding space.
    for text, _ in NORMALISE_CASES:
        out = normalize(text)
        assert out == out.lower()
        assert out == out.strip()
        assert "  " not in out


RAW_CASES = [
    ("Hello, world!", ["hello", "world"]),
    ("C_D_ player", ["c", "d", "player"]),
    ("don't Don't DON'T", ["don't", "don't", "don't"]),
    ("well-known ... thing", ["wellknown", "thing"]),
    ('"quoted" \'single\' -- x', ["quoted", "'single'", "x"]),
    ("$20 (twenty) 10% [ok] _a_b_", ["$20", "twenty", "10", "ok", "a", "b"]),
    ("tabs\tand\nnewlines", ["tabs", "and", "newlines"]),
    ("", []),
]


@pytest.mark.parametrize("text, expected", RAW_CASES)
def test_tokenize_raw_fixtures(text: str, expected: list[str]) -> None:
    assert tokenize_raw(text) == expected


def test_tokenize_raw_folds_typographic_apostrophe() -> None:
    assert tokenize_raw("don’t") == ["don't"]


def test_tokenize_raw_underscore_acronym_matches_normaliser() -> None:
    # A reference that writes a spelled acronym as "C_D_" must yield the tokens the
    # normaliser produces for the same text, otherwise the raw and normalised rates
    # disagree on every spelled acronym.
    assert tokenize_raw("the C_D_ player") == ["the", "c", "d", "player"]
    assert tokenize_raw("the C_D_ player") == tokenize_norm("the C_D_ player")


def test_tokenize_norm_is_normalize_then_split() -> None:
    text = "They won't pay twenty dollars, um, for that."
    assert tokenize_norm(text) == normalize(text).split()
    assert tokenize_norm(text) == ["they", "will", "not", "pay", "$20", "for", "that"]


def test_tokenize_raw_output_shape() -> None:
    # Property: every raw token is lower case and non-empty, and contains no punctuation
    # other than the apostrophe.
    import unicodedata

    for text, _ in RAW_CASES + [(t, None) for t, _ in NORMALISE_CASES]:
        for token in tokenize_raw(text):
            assert token
            assert token == token.lower()
            for ch in token:
                assert ch == "'" or not unicodedata.category(ch).startswith("P")
