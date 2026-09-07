"""Folders the application writes to: one home for settings, decoded work files and batch
records, resolved from the environment first so that a portable copy can keep everything
beside itself.
"""

from __future__ import annotations

import os
from pathlib import Path

HOME_ENV = "TWINSCRIBE_HOME"
APP_FOLDER = "twinscribe"


def app_home() -> Path:
    """Folder for settings, decoded work files and batch records; created when absent.

    The environment variable named by HOME_ENV wins, so a portable copy can point at a folder
    beside itself. Otherwise the per-user application data folder is used, and the user's home
    folder as a last resort.
    """
    override = os.environ.get(HOME_ENV)
    if override:
        base = Path(override)
    else:
        local = os.environ.get("LOCALAPPDATA") if os.name == "nt" else os.environ.get("XDG_DATA_HOME")
        base = Path(local) / APP_FOLDER if local else Path.home() / f".{APP_FOLDER}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def work_dir() -> Path:
    """Folder for decoded audio while a file is being processed."""
    folder = app_home() / "work"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def runs_dir() -> Path:
    """Folder for batch records."""
    folder = app_home() / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder
