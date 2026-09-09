"""Build the executables: one folder with twinscribe.exe (the command line) and
twinscribe-app.exe (the window), every library beside them, from tools/twinscribe.spec.

Outside the package on purpose. PyInstaller is installed into the current interpreter when it
is missing (the one network access of this tool). A models folder given with --models is
copied in beside the executables, where a frozen build looks for it; --zip packs the result.

Usage:
    python tools/build_exe.py [--dist dist] [--models models] [--zip] [--no-icon]
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC = REPO_ROOT / "tools" / "twinscribe.spec"
ICON = REPO_ROOT / "assets" / "twinscribe.ico"


def ensure_pyinstaller() -> None:
    if importlib.util.find_spec("PyInstaller") is None:
        print("installing PyInstaller into this interpreter")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller>=6.10"], check=True)


def folder_size(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the twinscribe executables.")
    parser.add_argument("--dist", type=Path, default=REPO_ROOT / "dist", help="output folder (default: dist)")
    parser.add_argument("--work", type=Path, default=REPO_ROOT / "build" / "pyinstaller", help="PyInstaller work folder")
    parser.add_argument("--models", type=Path, default=None, help="a fetched models folder to copy beside the executables")
    parser.add_argument("--zip", action="store_true", help="pack the folder as <dist>/twinscribe-win64.zip")
    parser.add_argument("--no-icon", action="store_true", help="do not render the icon first")
    args = parser.parse_args(argv)

    ensure_pyinstaller()
    if not args.no_icon:
        subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "make_icon.py")], check=True, cwd=str(REPO_ROOT))
    started = time.perf_counter()
    command = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--distpath", str(args.dist), "--workpath", str(args.work), str(SPEC),
    ]
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))
    folder = args.dist / "twinscribe"
    for name in ("twinscribe.exe", "twinscribe-app.exe"):
        if not (folder / name).is_file():
            print(f"missing: {folder / name}", file=sys.stderr)
            return 1
    if args.models is not None:
        target = folder / "models"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(args.models, target)
        print(f"models copied: {folder_size(target) / 1e9:.2f} GB")
    for name in ("NOTICE", "LICENSE", "README.md"):
        source = REPO_ROOT / name
        if source.is_file():
            shutil.copyfile(source, folder / name)
    print(f"built {folder} in {time.perf_counter() - started:.0f} s, {folder_size(folder) / 1e9:.2f} GB")
    if args.zip:
        archive = shutil.make_archive(str(args.dist / "twinscribe-win64"), "zip", root_dir=str(args.dist), base_dir="twinscribe")
        print(f"packed {archive} ({Path(archive).stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
