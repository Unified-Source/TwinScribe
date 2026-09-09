"""Fetch the catalogue models into a local store and pin their digests.

A thin front for `twinscribe.fetch`, kept so that a checkout fetches the same way it always
did. For every catalogue entry (or those named with --only) it downloads each source,
extracts the named member when the source is an archive, places the file in the model's
folder under the root, and records digest, size, URL, revision and licence in
models.lock.json. A file already present with a digest matching the lock is skipped. --verify
digests the store against the lock and fetches nothing.

Usage:
    python tools/fetch_models.py --root models [--only KEY ...]
    python tools/fetch_models.py --root models --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from twinscribe.fetch import STAGE_DONE, STAGE_EXTRACT, Cancelled, Progress, fetch_specs  # noqa: E402
from twinscribe.models import CATALOGUE, STATUS_VERIFIED, spec_for, verify_store  # noqa: E402


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

    specs = list(CATALOGUE) if not args.only else [spec_for(key) for key in args.only]
    last: dict[str, int] = {}

    def show(progress: Progress) -> None:
        if progress.stage == STAGE_DONE:
            return
        percent = int(100 * progress.fraction) if progress.fraction is not None else -1
        if last.get(progress.file) != percent:
            last[progress.file] = percent
            size = "extracting" if progress.stage == STAGE_EXTRACT else (f"{progress.done_bytes / 1e6:.1f} MB" if percent < 0 else f"{percent:3d}%")
            print(f"\r    {progress.key}/{progress.file} {size}", end="", flush=True)

    try:
        fetch_specs(specs, root, progress=show, log=lambda text: print("\n" + text, end=""))
    except Cancelled:
        print("\ncancelled")
        return 1
    print()
    print("Attribution required by the licences of the models fetched:")
    for spec in specs:
        print(f"  {spec.title}: {spec.credit} ({spec.licence})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
