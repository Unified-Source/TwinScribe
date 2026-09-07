"""Assemble a portable folder that runs twinscribe with nothing installed.

Outside the package on purpose: together with fetch_models.py this is the only code that
reaches the network. The layout is described in docs/Portable_Layout.md. --dry-run prints the
plan and touches nothing.

Usage:
    python tools/build_portable.py --out twinscribe-portable --models models --ffmpeg path/to/ffmpeg.exe
    python tools/build_portable.py --out twinscribe-portable --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PYTHON_VERSION = "3.11.9"
PYTHON_ZIP_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
PLATFORM = "win_amd64"
PYTHON_TAG = "3.11"

# The wheel set. PySide6 needs the abi3 tag stated explicitly for pip download.
REQUIREMENTS: tuple[str, ...] = (
    "numpy>=1.26",
    "PySide6>=6.8",
    "faster-whisper>=1.2,<2",
    "sherpa-onnx>=1.13,<2",
)

LAUNCHER_CLI = """@echo off
setlocal
set "TWINSCRIBE_HOME=%~dp0home"
set "TWINSCRIBE_MODELS=%~dp0models"
set "PATH=%~dp0bin;%PATH%"
"%~dp0python\\python.exe" -m twinscribe %*
endlocal
"""

LAUNCHER_APP = """@echo off
setlocal
set "TWINSCRIBE_HOME=%~dp0home"
set "TWINSCRIBE_MODELS=%~dp0models"
set "PATH=%~dp0bin;%PATH%"
start "" "%~dp0python\\pythonw.exe" -m twinscribe.app %*
endlocal
"""

# Forward slashes are accepted in ._pth entries on Windows.
PTH_LINES = ("python311.zip", ".", "../Lib/site-packages", "..", "import site")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "twinscribe-build/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, open(target, "wb") as handle:
        shutil.copyfileobj(response, handle)


def plan(args: argparse.Namespace) -> list[str]:
    out: Path = args.out
    steps = [
        f"1. download {PYTHON_ZIP_URL} and unzip into {out / 'python'}; write python311._pth",
        f"2. pip download {' '.join(REQUIREMENTS)} for {PLATFORM} / Python {PYTHON_TAG} (abi3 stated) into {out / 'wheels'}",
        f"3. pip install --no-deps --no-index --target {out / 'Lib' / 'site-packages'} every wheel",
        f"4. copy the twinscribe package from {REPO_ROOT / 'twinscribe'}",
        f"5. copy the models from {args.models} (with models.lock.json)" if args.models else "5. no models folder given; the app will report none",
        f"6. copy ffmpeg from {args.ffmpeg} into {out / 'bin'}" if args.ffmpeg else "6. no ffmpeg given; decoding will need one on the search path",
        f"7. write twinscribe.cmd, twinscribe-app.cmd, NOTICE and RECORD.txt under {out}",
    ]
    return steps


def build(args: argparse.Namespace) -> int:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    record: list[str] = []

    python_dir = out / "python"
    python_dir.mkdir(exist_ok=True)
    zip_path = out / f"python-{PYTHON_VERSION}-embed-amd64.zip"
    print(f"downloading {PYTHON_ZIP_URL}")
    download(PYTHON_ZIP_URL, zip_path)
    record.append(f"{zip_path.name}  {sha256_file(zip_path)}  {PYTHON_ZIP_URL}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(python_dir)
    zip_path.unlink()
    pth = next(python_dir.glob("python*._pth"), python_dir / "python311._pth")
    pth.write_text("\n".join(PTH_LINES) + "\n", encoding="ascii")

    wheels = out / "wheels"
    wheels.mkdir(exist_ok=True)
    print("downloading the wheel set")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "download",
            "--dest", str(wheels),
            "--platform", PLATFORM,
            "--python-version", PYTHON_TAG,
            "--implementation", "cp",
            "--abi", f"cp{PYTHON_TAG.replace('.', '')}",
            "--abi", "abi3",
            "--abi", "none",
            "--only-binary=:all:",
            *REQUIREMENTS,
        ],
        check=True,
    )
    site = out / "Lib" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    wheel_files = sorted(wheels.glob("*.whl"))
    print(f"unpacking {len(wheel_files)} wheels")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--target", str(site), *map(str, wheel_files)],
        check=True,
    )
    for wheel in wheel_files:
        record.append(f"wheels/{wheel.name}  {sha256_file(wheel)}")

    package_target = out / "twinscribe"
    if package_target.exists():
        shutil.rmtree(package_target)
    shutil.copytree(REPO_ROOT / "twinscribe", package_target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    if args.models:
        models_target = out / "models"
        if models_target.exists():
            shutil.rmtree(models_target)
        shutil.copytree(args.models, models_target)
    if args.ffmpeg:
        (out / "bin").mkdir(exist_ok=True)
        shutil.copy2(args.ffmpeg, out / "bin" / Path(args.ffmpeg).name)
        record.append(f"bin/{Path(args.ffmpeg).name}  {sha256_file(Path(args.ffmpeg))}  copied from the path given")
    (out / "home").mkdir(exist_ok=True)
    (out / "twinscribe.cmd").write_text(LAUNCHER_CLI, encoding="ascii", newline="\r\n")
    (out / "twinscribe-app.cmd").write_text(LAUNCHER_APP, encoding="ascii", newline="\r\n")
    shutil.copy2(REPO_ROOT / "NOTICE", out / "NOTICE")
    shutil.copy2(REPO_ROOT / "LICENSE", out / "LICENSE")
    (out / "RECORD.txt").write_text("\n".join(record) + "\n", encoding="utf-8")
    print(f"portable folder assembled at {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble a portable twinscribe folder.")
    parser.add_argument("--out", type=Path, required=True, help="the folder to assemble")
    parser.add_argument("--models", type=Path, default=None, help="a fetched models folder to copy in")
    parser.add_argument("--ffmpeg", type=Path, default=None, help="an ffmpeg executable to copy in")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and touch nothing")
    args = parser.parse_args(argv)
    if args.dry_run:
        for step in plan(args):
            print(step)
        return 0
    if args.models is not None and not args.models.is_dir():
        print(f"models folder not found: {args.models}", file=sys.stderr)
        return 2
    if args.ffmpeg is not None and not args.ffmpeg.is_file():
        print(f"ffmpeg not found: {args.ffmpeg}", file=sys.stderr)
        return 2
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
