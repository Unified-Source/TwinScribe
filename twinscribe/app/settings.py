"""Application settings kept as one JSON file under the application home, so that nothing is
written to the registry and a portable copy carries its settings with it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from twinscribe.paths import app_home
from twinscribe.runrecord import write_json_atomic

SETTINGS_SCHEMA = "twinscribe.settings.v1"
SETTINGS_FILE = "settings.json"
OUTPUT_BESIDE = "beside"
OUTPUT_FOLDER = "folder"
ACCELERATIONS: tuple[str, ...] = ("auto", "cpu", "cuda")
# The largest speaker count the window offers; zero means the count comes from clustering.
MAX_SPEAKERS = 8


@dataclass
class AppSettings:
    """Everything the window remembers between runs."""

    models_dir: str = ""
    output_mode: str = OUTPUT_BESIDE
    output_dir: str = ""
    quality: str = "standard"
    author: str = ""
    dark: bool = False
    threads: int = 0
    acceleration: str = "auto"
    speakers: int = 0
    follow: bool = True
    volume: float = 0.8
    rate: float = 1.0
    library: list[str] = field(default_factory=list)
    splitter: list[int] = field(default_factory=list)
    window_size: list[int] = field(default_factory=list)

    @property
    def output_dir_or_none(self) -> Path | None:
        """The output folder when the mode asks for one, else None (beside each recording)."""
        if self.output_mode == OUTPUT_FOLDER and self.output_dir:
            return Path(self.output_dir)
        return None

    @property
    def threads_or_none(self) -> int | None:
        return self.threads if self.threads > 0 else None

    @property
    def speakers_or_none(self) -> int | None:
        """The speaker count when one is set; None (the default) clusters by threshold."""
        return self.speakers if self.speakers > 0 else None


def settings_path() -> Path:
    """Where the settings file lives."""
    return app_home() / SETTINGS_FILE


def load_settings(path: str | os.PathLike[str] | None = None) -> AppSettings:
    """Read the settings; defaults for anything absent, unreadable or of the wrong type."""
    target = Path(path) if path is not None else settings_path()
    settings = AppSettings()
    if not target.is_file():
        return settings
    try:
        with open(target, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError):
        return settings
    if not isinstance(doc, dict):
        return settings
    for item in fields(AppSettings):
        value: Any = doc.get(item.name)
        if value is None:
            continue
        current = getattr(settings, item.name)
        if isinstance(current, bool):
            if isinstance(value, bool):
                setattr(settings, item.name, value)
        elif isinstance(current, int):
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(settings, item.name, value)
        elif isinstance(current, float):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                setattr(settings, item.name, float(value))
        elif isinstance(current, str):
            if isinstance(value, str):
                setattr(settings, item.name, value)
        elif isinstance(current, list):
            if isinstance(value, list):
                setattr(settings, item.name, list(value))
    if settings.output_mode not in (OUTPUT_BESIDE, OUTPUT_FOLDER):
        settings.output_mode = OUTPUT_BESIDE
    if settings.acceleration not in ACCELERATIONS:
        settings.acceleration = "auto"
    settings.speakers = min(MAX_SPEAKERS, max(0, settings.speakers))
    settings.volume = min(1.0, max(0.0, settings.volume))
    settings.rate = min(2.0, max(0.5, settings.rate))
    return settings


def save_settings(settings: AppSettings, path: str | os.PathLike[str] | None = None) -> Path:
    """Write the settings atomically and return the path."""
    target = Path(path) if path is not None else settings_path()
    doc = {"schema": SETTINGS_SCHEMA} | asdict(settings)
    write_json_atomic(doc, target)
    return target
