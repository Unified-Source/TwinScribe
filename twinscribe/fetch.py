"""Fetching the catalogue models into a store, on an explicit action only.

This is the one place in the package that reaches the network, and it runs only when a
person asks for it: the first-run dialog's Download button, or the fetch-models command. The
engines never do; they load from the store. Every file fetched is recorded in the lock with
its digest, size, source, revision and licence, and a file already present with a digest
matching the lock is not fetched again. A download goes to a temporary name and is renamed
only when complete, so a cancelled or failed fetch leaves no half file behind. An archive is
downloaded once and its wanted members are written in a single streaming pass.
"""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twinscribe.models import (
    BACKEND_CT2,
    CATALOGUE,
    KEY_AUDIO_TAGGER,
    MODELS_ENV,
    ModelSet,
    ModelSpec,
    Source,
    find_models,
    lock_entry,
    read_lock,
    sha256_file,
    write_lock,
)
from twinscribe.paths import app_home
from twinscribe.profiles import PROFILES, profile_for

CHUNK = 1 << 20
USER_AGENT = "twinscribe-fetch/0.1"
TIMEOUT_S = 60

STAGE_DOWNLOAD = "download"
STAGE_EXTRACT = "extract"
STAGE_DONE = "done"


class Cancelled(Exception):
    """Raised inside a fetch when the cancel callback says to stop."""


@dataclass(frozen=True)
class Progress:
    """Where a fetch stands.

    `stage` is "download" while bytes of `file` (a model file, or an archive) arrive, with the
    bytes so far and the total when the source says it; "extract" while `file` is being written
    out of an archive, with no byte counts; "done" once `file` is in place and counted in
    `files_done`. `files_done` and `files_total` count the files to fetch.
    """

    key: str
    file: str
    done_bytes: int
    total_bytes: int
    files_done: int
    files_total: int
    stage: str = STAGE_DOWNLOAD

    @property
    def fraction(self) -> float | None:
        """The file's fraction, or None when its size is not known or no bytes are moving."""
        if self.stage != STAGE_DOWNLOAD or self.total_bytes <= 0:
            return None
        return min(1.0, self.done_bytes / self.total_bytes)


ProgressFn = Callable[[Progress], None]
CancelFn = Callable[[], bool]
LogFn = Callable[[str], None]


def download(
    url: str,
    target: Path,
    progress: Callable[[int, int], None] | None = None,
    cancel: CancelFn | None = None,
) -> str | None:
    """Stream a URL to `target` through a temporary name; return an upstream revision header.

    `progress` receives (bytes so far, total bytes or 0). `cancel` is asked between chunks;
    a True answer raises Cancelled after the partial file is removed.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response, open(partial, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            revision = response.headers.get("X-Repo-Commit") or response.headers.get("ETag")
            done = 0
            while True:
                if cancel is not None and cancel():
                    raise Cancelled(url)
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
    except BaseException:
        try:
            partial.unlink()
        except OSError:
            pass
        raise
    os.replace(partial, target)
    return revision.strip('"') if revision else None


def extract_members(
    archive: Path,
    targets: dict[str, Path],
    written: Callable[[str], None] | None = None,
    cancel: CancelFn | None = None,
) -> None:
    """Copy the wanted members out of a tar archive in one streaming pass, matching on the
    file name inside any top folder; `written` is told each member's name once it is in place.

    The archive is read once from start to end, which matters for compressed archives, where
    seeking back means decompressing from the start again.
    """
    wanted = dict(targets)
    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            if cancel is not None and cancel():
                raise Cancelled(archive.name)
            name = Path(member.name).name
            if not member.isfile() or name not in wanted:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            target = wanted.pop(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            try:
                with extracted, open(partial, "wb") as handle:
                    shutil.copyfileobj(extracted, handle)
            except BaseException:
                try:
                    partial.unlink()
                except OSError:
                    pass
                raise
            os.replace(partial, target)
            if written is not None:
                written(name)
            if not wanted:
                return
    if wanted:
        raise FileNotFoundError(f"{', '.join(sorted(wanted))} not found inside {archive.name}")


def extract_member(archive: Path, member_name: str, target: Path) -> None:
    """Copy one member out of a tar archive, matching on the file name inside any top folder."""
    extract_members(archive, {member_name: target})


def present_and_pinned(target: Path, key: str, lock: dict[str, dict[str, Any]]) -> bool:
    entry = lock.get(key)
    return target.is_file() and entry is not None and sha256_file(target) == entry.get("sha256")


def missing_sources(spec: ModelSpec, root: Path, lock: dict[str, dict[str, Any]]) -> list[Source]:
    """The sources of a model whose files are absent or do not match the lock."""
    folder = root / spec.key
    return [source for source in spec.sources if not present_and_pinned(folder / source.target, f"{spec.key}/{source.target}", lock)]


def fetch_specs(
    specs: Sequence[ModelSpec],
    root: str | os.PathLike[str],
    progress: ProgressFn | None = None,
    cancel: CancelFn | None = None,
    log: LogFn | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch every file of the given models that the store lacks; returns the lock entries.

    A plain source is downloaded to its place. An archive is downloaded once into a temporary
    folder under the root, reported under its own name, and every wanted member is written in
    one pass and reported as it lands. The lock is written after every file, so an interrupted
    fetch keeps what it finished.
    """
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    lock = read_lock(base)
    say = log if log is not None else (lambda _text: None)
    wanted = [(spec, source) for spec in specs for source in missing_sources(spec, base, lock)]
    total = len(wanted)
    state = {"done": 0}
    downloads = Path(tempfile.mkdtemp(prefix="twinscribe-fetch-", dir=str(base)))

    def report(stage: str, key: str, file: str, done_bytes: int = 0, total_bytes: int = 0) -> None:
        if progress is not None:
            progress(Progress(key, file, done_bytes, total_bytes, state["done"], total, stage))

    def placed(spec: ModelSpec, source: Source, target: Path, revision: str | None) -> None:
        key = f"{spec.key}/{source.target}"
        lock[key] = lock_entry(spec, source, target, revision)
        write_lock(base, lock)
        state["done"] += 1
        say(f"  {source.target}: {lock[key]['bytes'] / 1e6:.1f} MB, sha256 {lock[key]['sha256'][:16]}")
        report(STAGE_DONE, spec.key, source.target, lock[key]["bytes"], lock[key]["bytes"])

    finished: set[tuple[str, str]] = set()
    try:
        for spec, source in wanted:
            if (spec.key, source.target) in finished:
                continue
            if cancel is not None and cancel():
                raise Cancelled(f"{spec.key}/{source.target}")
            folder = base / spec.key
            folder.mkdir(parents=True, exist_ok=True)
            if source.archive is None:
                say(f"{spec.title}: downloading {source.target}")
                revision = download(
                    source.url, folder / source.target,
                    lambda done_bytes, total_bytes, _k=spec.key, _f=source.target: report(STAGE_DOWNLOAD, _k, _f, done_bytes, total_bytes),
                    cancel,
                )
                placed(spec, source, folder / source.target, revision)
                continue
            # Every wanted member of this archive for this model, written in one pass.
            missing_here = {s.target for owner, s in wanted if owner is spec}
            members = [s for s in spec.sources
                       if s.archive == source.archive and s.target in missing_here and (spec.key, s.target) not in finished]
            say(f"{spec.title}: downloading {source.archive}")
            archive_path = downloads / source.archive
            revision = download(
                source.url, archive_path,
                lambda done_bytes, total_bytes, _k=spec.key, _a=source.archive: report(STAGE_DOWNLOAD, _k, _a, done_bytes, total_bytes),
                cancel,
            )
            say(f"{spec.title}: extracting {len(members)} file(s) from {source.archive}")
            report(STAGE_EXTRACT, spec.key, source.archive)
            by_name = {m.target: m for m in members}

            def written(name: str, _spec: ModelSpec = spec, _by_name: dict[str, Source] = by_name, _rev: str | None = revision) -> None:
                member = _by_name[name]
                placed(_spec, member, base / _spec.key / member.target, _rev)
                finished.add((_spec.key, member.target))

            extract_members(archive_path, {m.target: folder / m.target for m in members}, written, cancel)
            try:
                archive_path.unlink()
            except OSError:
                pass
    finally:
        shutil.rmtree(downloads, ignore_errors=True)
    return lock


def specs_for_level(name: str) -> tuple[ModelSpec, ...]:
    """The models a quality level needs with its CTranslate2 detector, plus the audio tagger
    (small, optional, used by the scene pass when present), in catalogue order."""
    profile = profile_for(name)
    keys = set(profile.required_keys) | {profile.detector, KEY_AUDIO_TAGGER}
    return tuple(spec for spec in CATALOGUE if spec.key in keys and (spec.backend in ("", BACKEND_CT2)))


def missing_for_levels(names: Iterable[str], models: ModelSet) -> list[ModelSpec]:
    """The catalogue entries the store lacks for the given levels, in catalogue order."""
    keys: set[str] = set()
    for name in names:
        keys.update(spec.key for spec in specs_for_level(name))
    return [spec for spec in CATALOGUE if spec.key in keys and not models.has(spec.key)]


def level_names() -> tuple[str, ...]:
    return tuple(profile.name for profile in PROFILES)


def _writable(folder: Path) -> bool:
    """Whether files can be made in the folder; a folder created for the probe is removed again."""
    created = not folder.exists()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write-probe"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError:
        return False
    finally:
        if created:
            try:
                folder.rmdir()
            except OSError:
                pass
    return True


def proposed_root(models: ModelSet | None = None, explicit: bool = False) -> Path:
    """Where a first fetch should go.

    The store in use, when it already holds anything, or when it was named explicitly (the
    environment variable, or the settings when `explicit` says so) and can be written; else a
    models folder beside a frozen build's executable when that can be made, so an unpacked
    folder carries its own models; else the store in use when it can be written; else a
    models folder under the application home.
    """
    current = models if models is not None else find_models(None)
    named = explicit or bool(os.environ.get(MODELS_ENV))
    if current.present or current.incomplete or (named and _writable(current.root)):
        return current.root
    if getattr(sys, "frozen", False):
        beside = Path(sys.executable).resolve().parent / "models"
        if _writable(beside):
            return beside
    if _writable(current.root):
        return current.root
    return app_home() / "models"


__all__ = [
    "Cancelled",
    "Progress",
    "STAGE_DONE",
    "STAGE_DOWNLOAD",
    "STAGE_EXTRACT",
    "download",
    "extract_member",
    "extract_members",
    "fetch_specs",
    "level_names",
    "missing_for_levels",
    "missing_sources",
    "proposed_root",
    "specs_for_level",
]
