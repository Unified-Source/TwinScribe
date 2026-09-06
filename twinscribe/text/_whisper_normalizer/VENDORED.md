# Vendored: Whisper text normalisers

Source: the `whisper/normalizers/` directory of the upstream Whisper repository
(https://github.com/openai/whisper), MIT License, Copyright (c) 2022 OpenAI. The licence text
is reproduced verbatim in `LICENSE` in this folder.

Upstream commit: `86098128c0b4f24f0e2aa2994de830614b474227` (branch `main`, committed
2026-08-31). Files were fetched from that repository at that commit.

| File | Upstream path | Upstream SHA-256 | Status |
|---|---|---|---|
| `english.py` | `whisper/normalizers/english.py` | `d31a52685b5b42628ea4e3343c3a7925ce5b2f317e6c0ec4575677dd5c8b5d2c` | modified (see below) |
| `english.json` | `whisper/normalizers/english.json` | `6607f948be9824d2e1b2fa2223cd94c06c45afa4e05ea0e3d5e1f2bdffde2465` | verbatim |
| `basic.py` | `whisper/normalizers/basic.py` | `4742eaa040e0657fa1247a1361e0d856c62317a43326ea59a40c2e9edd8d2c38` | modified (see below) |
| `LICENSE` | `LICENSE` | `b5d65a59060e68c4ff940e1eddfa6f94b2d68fdf58ed7f4dd57721c997e35e9d` | verbatim |
| `__init__.py` | (none) | | written here; exports the two normaliser classes |
| `VENDORED.md` | (none) | | this record |

The upstream `whisper/normalizers/__init__.py` was not taken; the local `__init__.py` is a
two-line re-export with a docstring.

## Purpose of the changes

The upstream modules import two third-party packages, `more_itertools` and `regex`. This
copy is standard-library only, so the one helper used from `more_itertools` is inlined and
the one code path that needs `regex` is removed. Every other line, including comments and
the number-handling logic, is unchanged.

## Changes to `basic.py`

1. A module docstring was added at the top of the file.
2. `import regex` was removed.
3. `BasicTextNormalizer.__init__` lost its `split_letters` parameter and the
   `self.split_letters` attribute. The signature is now `__init__(self, remove_diacritics:
   bool = False)`.
4. In `BasicTextNormalizer.__call__` the branch

       if self.split_letters:
           s = " ".join(regex.findall(r"\X", s, regex.U))

   was removed. It split text into extended grapheme clusters using the third-party
   `regex` package. Nothing in this repository used the option.

## Changes to `english.py`

1. A module docstring was added at the top of the file.
2. `from more_itertools import windowed` was removed, and `Iterable` and `Tuple` were added
   to the `typing` import line.
3. A private function `_windowed(seq, n)` was added after the imports. It yields
   consecutive size-`n` windows of `seq` as tuples with step 1 and, for a sequence shorter
   than `n`, a single window padded on the right with `None`, which is the behaviour of
   `more_itertools.windowed` for the arguments used at the one call site.
4. In `EnglishNumberNormalizer.process_words`, the call
   `windowed([None] + words + [None], 3)` became `_windowed([None] + words + [None], 3)`.
5. In `EnglishSpellingNormalizer.__init__`, `self.mapping = json.load(open(mapping_path))`
   became a `with open(mapping_path, encoding="utf-8") as f:` block around `json.load(f)`,
   so the file handle is closed and the read does not depend on the process locale.

## Behaviour

For every input, `EnglishTextNormalizer` in this copy returns the same string as the
upstream class at the recorded commit. `BasicTextNormalizer` matches upstream for the
default `split_letters=False`.
