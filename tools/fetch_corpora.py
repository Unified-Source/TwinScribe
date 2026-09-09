"""Fetch the bench corpora into a local store and pin their digests.

This tool is outside the package on purpose: with the model fetch tool it is the only place
that reaches the network. For every catalogue corpus (or those named with --only) it downloads
each file, or each archive and the members the corpus names or selects, places them under the
corpus folder, and records digest, size, URL, revision and licence in corpora.lock.json. A file
already present with a digest matching the lock is skipped. --verify digests the store against
the lock and fetches nothing.

Usage:
    python tools/fetch_corpora.py --root corpora [--only KEY ...]
    python tools/fetch_corpora.py --root corpora --verify
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from twinscribe.bench.corpora import (  # noqa: E402
    CATALOGUE,
    STATUS_VERIFIED,
    CorpusFile,
    CorpusSpec,
    lock_entry,
    read_lock,
    select_members,
    spec_for,
    verify_store,
    write_lock,
)
from twinscribe.models import sha256_file  # noqa: E402

CHUNK = 1 << 20
USER_AGENT = "twinscribe-fetch/0.1"


def download(url: str, target: Path) -> str | None:
    """Stream a URL to target with a progress line; return an upstream revision header if any."""
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, open(target, "wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        revision = response.headers.get("X-Repo-Commit") or response.headers.get("ETag")
        done = 0
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            else:
                print(f"\r    {done / 1e6:8.1f} MB", end="", flush=True)
        print()
    return revision.strip('"') if revision else None


def archive_names(archive: Path) -> list[str]:
    """Member names of a zip or tar archive."""
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            return [name for name in handle.namelist() if not name.endswith("/")]
    with tarfile.open(archive, "r:*") as handle:
        return [member.name for member in handle.getmembers() if member.isfile()]


def extract_members(archive: Path, wanted: dict[str, Path]) -> None:
    """Copy the wanted members (member name to target path) out of a zip or tar archive."""
    missing = set(wanted)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for name in handle.namelist():
                if name in wanted:
                    wanted[name].parent.mkdir(parents=True, exist_ok=True)
                    with handle.open(name) as source, open(wanted[name], "wb") as target:
                        shutil.copyfileobj(source, target)
                    missing.discard(name)
    else:
        with tarfile.open(archive, "r:*") as handle:
            for member in handle:
                if member.isfile() and member.name in wanted:
                    extracted = handle.extractfile(member)
                    if extracted is None:
                        continue
                    wanted[member.name].parent.mkdir(parents=True, exist_ok=True)
                    with extracted, open(wanted[member.name], "wb") as target:
                        shutil.copyfileobj(extracted, target)
                    missing.discard(member.name)
    if missing:
        raise FileNotFoundError(f"{', '.join(sorted(missing))} not found inside {archive.name}")


def _present(target: Path, key: str, lock: dict) -> bool:
    entry = lock.get(key)
    return target.is_file() and entry is not None and sha256_file(target) == entry.get("sha256")


def fetch_spec(spec: CorpusSpec, root: Path, lock: dict, downloads: Path) -> None:
    folder = root / spec.key
    folder.mkdir(parents=True, exist_ok=True)
    print(f"{spec.key}  ({spec.title}; {spec.licence}; {spec.revision})")
    archives: dict[str, tuple[Path, str | None]] = {}

    def archive_for(url: str, name: str) -> tuple[Path, str | None]:
        if name not in archives:
            path = downloads / name
            print(f"  downloading {name}")
            archives[name] = (path, download(url, path))
        return archives[name]

    files: list[CorpusFile] = list(spec.files)
    if spec.selection is not None and spec.archive_url is not None:
        archive_name = spec.archive_url.rsplit("/", 1)[-1]
        archive_path, _ = archive_for(spec.archive_url, archive_name)
        for member, target in select_members(spec, archive_names(archive_path)):
            files.append(CorpusFile(url=spec.archive_url, target=target, archive=archive_name, member=member))

    pending: dict[str, dict[str, Path]] = {}
    for file in files:
        target = folder / file.target
        key = f"{spec.key}/{file.target}"
        if _present(target, key, lock):
            continue
        if file.archive is not None:
            archive_for(file.url, file.archive)
            pending.setdefault(file.archive, {})[file.member or file.target] = target
        else:
            print(f"  downloading {file.target}")
            revision = download(file.url, target)
            lock[key] = lock_entry(spec, file, target, revision)
            print(f"  {file.target}: {lock[key]['bytes'] / 1e6:.2f} MB, sha256 {lock[key]['sha256'][:16]}")
            write_lock(root, lock)
    for archive_name, wanted in pending.items():
        archive_path, revision = archives[archive_name]
        print(f"  extracting {len(wanted)} member(s) of {archive_name}")
        extract_members(archive_path, wanted)
        for file in files:
            if file.archive == archive_name and (file.member or file.target) in wanted:
                key = f"{spec.key}/{file.target}"
                lock[key] = lock_entry(spec, file, folder / file.target, revision)
        write_lock(root, lock)
    present = sum(1 for file in files if (folder / file.target).is_file())
    print(f"  {present} of {len(files)} files present")
    for archive_path, _ in archives.values():
        try:
            archive_path.unlink()
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the bench corpora and pin their digests.")
    parser.add_argument("--root", type=Path, required=True, help="the corpora root folder")
    parser.add_argument("--only", nargs="*", default=None, help="corpus keys to fetch (default: all)")
    parser.add_argument("--verify", action="store_true", help="digest the store against the lock; fetch nothing")
    args = parser.parse_args(argv)
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)

    if args.verify:
        checks = verify_store(root)
        for check in checks:
            print(f"{check.status:<9} {check.key}/{check.file}  {check.detail}")
        bad = [c for c in checks if c.status != STATUS_VERIFIED]
        print(f"{len(checks) - len(bad)} verified, {len(bad)} not")
        return 0 if not bad else 1

    specs = CATALOGUE if not args.only else tuple(spec_for(key) for key in args.only)
    lock = read_lock(root)
    downloads = Path(tempfile.mkdtemp(prefix="twinscribe-fetch-", dir=str(root)))
    try:
        for spec in specs:
            fetch_spec(spec, root, lock, downloads)
    finally:
        shutil.rmtree(downloads, ignore_errors=True)
    print()
    print("Attribution required by the licences of the corpora fetched:")
    for spec in specs:
        print(f"  {spec.title}: {spec.credit} ({spec.licence})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
