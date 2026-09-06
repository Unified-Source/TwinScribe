"""Vendored copy of the Whisper text normalisers (MIT, Copyright (c) 2022 OpenAI).

The upstream source, commit and every local change are recorded in VENDORED.md beside this
file; the upstream licence is in LICENSE. This package is standard-library only.
"""

from .basic import BasicTextNormalizer
from .english import EnglishTextNormalizer

__all__ = ["BasicTextNormalizer", "EnglishTextNormalizer"]
