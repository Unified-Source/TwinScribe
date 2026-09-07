"""Fetch the catalogue models into a local store and pin their digests.

This tool is outside the package on purpose: it is the one place that reaches the network.
For every catalogue entry (or those named with --only) it downloads each source, extracts the
named member when the source is an archive, places the file in the model's folder under the
root, and records digest, size, URL, revision and licence in models.lock.json. A file that is
already present with a digest matching the lock is skipped. --verify digests the store against
the lock and fetches nothing.

Usage:
    python tools/fetch_models.py --root models [--only KEY ...]
    python tools/fetch_models.py --root models --verify
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from twinscribe.models import (  # noqa: E402
    CATALOGUE,
    STATUS_VERIFIED,
    ModelSpec,
    Source,
    lock_entry,
    read_lock,
    sha256_file,
    spec_for,
    verify_store,
    write_lock,
)

CHUNK = 1 << 20
USER_AGENT = "twinscribe-fetch/0.1"


def download(url: str, target: Path) -> str | None:
    """Stream a URL to target with a progress line; return the upstream revision header if any."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as handle:
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


def extract_member(archive: Path, member_name: str, target: Path) -> None:
    """Copy one member out of a tar archive, matching on the file name inside any top folder."""
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            if member.isfile() and Path(member.name).name == member_name:
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                with extracted, open(target, "wb") as handle:
                    shutil.copyfileobj(extracted, handle)
                return
    raise FileNotFoundError(f"{member_name} not found inside {archive.name}")


def fetch_spec(spec: ModelSpec, root: Path, lock: dict, downloads: Path) -> None:
    folder = root / spec.key
    folder.mkdir(parents=True, exist_ok=True)
    print(f"{spec.key}  ({spec.title}; {spec.licence})")
    archives: dict[str, tuple[Path, str | None]] = {}
    for source in spec.sources:
        target = folder / source.target
        key = f"{spec.key}/{source.target}"
        entry = lock.get(key)
        if target.is_file() and entry is not None and sha256_file(target) == entry.get("sha256"):
            print(f"  {source.target}: present, digest matches the lock")
            continue
        revision: str | None
        if source.archive is not None:
            if source.archive not in archives:
                archive_path = downloads / source.archive
                print(f"  downloading {source.archive}")
                revision = download(source.url, archive_path)
                archives[source.archive] = (archive_path, revision)
            archive_path, revision = archives[source.archive]
            extract_member(archive_path, source.target, target)
        else:
            print(f"  downloading {source.target}")
            revision = download(source.url, target)
        lock[key] = lock_entry(spec, source, target, revision)
        print(f"  {source.target}: {lock[key]['bytes'] / 1e6:.1f} MB, sha256 {lock[key]['sha256'][:16]}")
        write_lock(root, lock)
    for archive_path, _ in archives.values():
        try:
            archive_path.unlink()
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the catalogue models and pin their digests.")
    parser.add_argument("--root", type=Path, required=True, help="the models root folder")
    parser.add_argument("--only", nargs="*", default=None, help="catalogue keys to fetch (default: all)")
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
    print("Attribution required by the licences of the models fetched:")
    for spec in specs:
        print(f"  {spec.title}: {spec.credit} ({spec.licence})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
