"""Folders the application writes to: one home for settings, decoded work files and batch
records, resolved from the environment first so that a portable copy can keep everything
beside itself.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HOME_ENV = "TWINSCRIBE_HOME"
APP_FOLDER = "twinscribe"


def default_app_home(platform_name: str | None = None, environ: dict[str, str] | None = None) -> Path:
    """The per-user application data folder of each platform, without creating it.

    Windows: the local application data folder. macOS: Application Support under the
    library. Elsewhere: the XDG data home, else .local/share. A hidden folder under the home
    directory is the last resort everywhere.
    """
    name = platform_name if platform_name is not None else sys.platform
    env = environ if environ is not None else dict(os.environ)
    home = Path(env.get("HOME") or env.get("USERPROFILE") or Path.home())
    if name.startswith("win"):
        local = env.get("LOCALAPPDATA")
        if local:
            return Path(local) / APP_FOLDER
    elif name.startswith("darwin"):
        return home / "Library" / "Application Support" / APP_FOLDER
    else:
        xdg = env.get("XDG_DATA_HOME")
        if xdg:
            return Path(xdg) / APP_FOLDER
        return home / ".local" / "share" / APP_FOLDER
    return home / f".{APP_FOLDER}"


def frozen_sibling(name: str) -> Path | None:
    """`<folder of the executable>/<name>` when this is a frozen build and that folder exists,
    else None; a folder unpacked anywhere then carries its own models and home."""
    if not getattr(sys, "frozen", False):
        return None
    candidate = Path(sys.executable).resolve().parent / name
    return candidate if candidate.is_dir() else None


def app_home() -> Path:
    """Folder for settings, decoded work files and batch records; created when absent.

    The environment variable named by HOME_ENV wins, so a portable copy can point at a folder
    beside itself; a frozen build with a home folder beside its executable uses that.
    Otherwise the platform's per-user application data folder is used.
    """
    override = os.environ.get(HOME_ENV)
    if override:
        base = Path(override)
    else:
        base = frozen_sibling("home") or default_app_home()
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
