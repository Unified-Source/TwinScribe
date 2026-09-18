"""Assemble a portable folder that runs twinscribe with nothing installed, and pack it into
the parts a release can carry.

Outside the package on purpose: together with fetch_models.py this is the only code that
reaches the network. The layout is described in docs/Portable_Layout.md. --dry-run prints the
plan and touches nothing.

Usage:
    python tools/build_portable.py --out twinscribe-portable --models models --ffmpeg path/to/ffmpeg.exe
    python tools/build_portable.py --out twinscribe-portable --gpu --bundle-ffmpeg --wheels wheels --models models --level standard --pack
    python tools/build_portable.py --out twinscribe-portable --pack-only
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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PYTHON_VERSION = "3.11.9"
PYTHON_TAG = "3.11"

# Windows architectures the embeddable interpreter is published for, with the wheel platform
# tag of each. CTranslate2 publishes no wheel for arm64, so that build carries the ONNX
# detector only.
ARCHITECTURES: dict[str, str] = {"amd64": "win_amd64", "arm64": "win_arm64"}

# The wheel set. PySide6 needs the abi3 tag stated explicitly for pip download.
REQUIREMENTS_COMMON: tuple[str, ...] = (
    "numpy>=1.26",
    "PySide6>=6.8",
    "sherpa-onnx>=1.13,<2",
)
REQUIREMENTS_CT2: tuple[str, ...] = ("faster-whisper>=1.2,<2",)
REQUIREMENTS_CUDA: tuple[str, ...] = ("nvidia-cublas-cu12", "nvidia-cudnn-cu12")
REQUIREMENTS_FFMPEG: tuple[str, ...] = ("imageio-ffmpeg>=0.5",)

DEFAULT_LEVEL = "standard"

# The parts. A release carries files of at most two gibibytes, and the folder with the models
# of a level is larger than that, so it is published as parts: the first unpacks to the folder
# named here and holds everything but the checker's model and the CUDA runtime packages; the
# second holds the checker's model; the third, optional, the CUDA packages. The launchers
# unpack any part placed beside them on the first start.
TOP_FOLDER = "twinscribe-portable"
PART_PREFIX = "twinscribe-portable-win64"
PART_NAMES: dict[str, str] = {
    "part1": f"{PART_PREFIX}-part1.zip",
    "part2": f"{PART_PREFIX}-part2.zip",
    "cuda": f"{PART_PREFIX}-cuda.zip",
}
ASSET_LIMIT = 2 * 1024 ** 3


def python_zip_url(arch: str) -> str:
    return f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-{arch}.zip"


def requirements_for(arch: str, gpu: bool, bundled_ffmpeg: bool) -> tuple[str, ...]:
    """The wheel set for an architecture: the ONNX-only set on arm64, CUDA packages on request."""
    wheels = list(REQUIREMENTS_COMMON)
    if arch == "amd64":
        wheels.extend(REQUIREMENTS_CT2)
        if gpu:
            wheels.extend(REQUIREMENTS_CUDA)
    if bundled_ffmpeg:
        wheels.extend(REQUIREMENTS_FFMPEG)
    return tuple(wheels)


# The launchers set the three folders beside themselves, unpack any part placed beside them
# (once: a part unpacked is renamed so that it is not unpacked again), then start the program.
# A failed unpack stops the start with the reason on screen, since a folder with half a model
# in it would otherwise open and report the model absent.
LAUNCHER_HEAD = """@echo off
setlocal
set "HERE=%~dp0"
set "TWINSCRIBE_HOME=%HERE%home"
set "TWINSCRIBE_MODELS=%HERE%models"
set "PATH=%HERE%bin;%PATH%"
for %%Z in ("%HERE%twinscribe-portable-*.zip") do call :unpack "%%~fZ"
if defined UNPACK_FAILED exit /b 1
"""

LAUNCHER_TAIL = """endlocal
exit /b

:unpack
echo Unpacking %~nx1 into this folder; this happens once and may take a few minutes.
"%HERE%python\\python.exe" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" %1 "%HERE%."
if errorlevel 1 (
  echo %~nx1 could not be unpacked; the download may be incomplete. Download it again and put it beside this launcher.
  set "UNPACK_FAILED=1"
  pause
  exit /b 1
)
ren %1 "%~nx1.unpacked"
exit /b 0
"""

LAUNCHER_CLI = LAUNCHER_HEAD + '"%HERE%python\\python.exe" -m twinscribe %*\n' + LAUNCHER_TAIL
LAUNCHER_APP = LAUNCHER_HEAD + 'start "" "%HERE%python\\pythonw.exe" -m twinscribe.app %*\n' + LAUNCHER_TAIL

# Forward slashes are accepted in ._pth entries on Windows.
PTH_LINES = ("python311.zip", ".", "../Lib/site-packages", "..", "import site")


@dataclass(frozen=True)
class Part:
    """One archive of a packed folder: its file, its size, its digest and its member count."""

    name: str
    path: Path
    bytes: int
    sha256: str
    files: int


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


def level_keys(level: str, arch: str) -> tuple[str, ...]:
    """The catalogue keys a quality level needs on an architecture: with the CTranslate2
    detector where that library has a wheel, with the ONNX detector elsewhere."""
    from twinscribe.fetch import specs_for_level
    from twinscribe.models import BACKEND_CT2, BACKEND_ONNX

    backends = {BACKEND_CT2: arch == "amd64", BACKEND_ONNX: True}
    return tuple(spec.key for spec in specs_for_level(level, backends))


def detector_key(level: str, arch: str) -> str:
    """The level's detector on the architecture: the model the second part carries."""
    from twinscribe.profiles import profile_for

    keys = level_keys(level, arch)
    return next(key for key in profile_for(level).detectors if key in keys)


def copy_models(source: Path, target: Path, keys: Sequence[str] | None) -> tuple[str, ...]:
    """Copy model folders out of a store with a lock holding their entries only: the folders
    named by `keys`, or every folder when none are named. Returns the keys copied; a key
    without a folder in the source is an error rather than a silent gap."""
    from twinscribe.models import read_lock, write_lock

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    chosen = tuple(keys) if keys is not None else tuple(sorted(p.name for p in source.iterdir() if p.is_dir()))
    for key in chosen:
        folder = source / key
        if not folder.is_dir():
            raise FileNotFoundError(f"{key} has no folder under {source}")
        shutil.copytree(folder, target / key)
    lock = read_lock(source)
    kept = {entry: value for entry, value in lock.items() if entry.split("/", 1)[0] in chosen}
    write_lock(target, kept)
    return chosen


def classify(relative: str, checker: str) -> str | None:
    """Which part a file of the assembled folder goes to, by its path relative to the folder
    with forward slashes: "cuda" for the CUDA runtime packages, "part2" for the checker's
    model, None for what is not shipped (the wheel set, byte-code caches, the contents of the
    home folder, parts lying in the folder and their unpacked markers), "part1" for the rest."""
    pieces = relative.split("/")
    if pieces[0] in ("wheels", "home") or "__pycache__" in pieces or relative.endswith((".pyc", ".unpacked")):
        return None
    if len(pieces) == 1 and relative.startswith(f"{TOP_FOLDER}-") and relative.endswith(".zip"):
        return None
    if pieces[:2] == ["Lib", "site-packages"] and len(pieces) > 3 and (pieces[2] == "nvidia" or pieces[2].startswith("nvidia_")):
        return "cuda"
    if pieces[:2] == ["models", checker]:
        return "part2"
    return "part1"


def pack(folder: Path, out: Path, checker: str, limit: int = ASSET_LIMIT) -> list[Part]:
    """Write the parts of an assembled folder into `out`.

    The first part holds the folder under TOP_FOLDER with everything but the checker's model
    and the CUDA packages, and an empty home folder, so that unpacking it gives a folder that
    starts. The second part holds the checker's model and the third the CUDA packages, both
    laid out relative to the folder, which is where the launchers unpack them; the third is
    not written when the folder has no CUDA packages. Model files are stored without
    compression, since weights hardly compress and unpacking them is then a copy. A part
    larger than `limit` is an error, because a release cannot carry it.
    """
    out.mkdir(parents=True, exist_ok=True)
    members: dict[str, list[str]] = {key: [] for key in PART_NAMES}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        relative = path.relative_to(folder).as_posix()
        part = classify(relative, checker)
        if part is not None:
            members[part].append(relative)
    if not members["part2"]:
        raise ValueError(f"nothing for {PART_NAMES['part2']}: {folder / 'models' / checker} holds no files")
    written: list[Part] = []
    for key, name in PART_NAMES.items():
        if key == "cuda" and not members[key]:
            continue
        target = out / name
        prefix = f"{TOP_FOLDER}/" if key == "part1" else ""
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            if key == "part1":
                archive.writestr(zipfile.ZipInfo(f"{TOP_FOLDER}/home/"), b"")
            for relative in members[key]:
                compression = zipfile.ZIP_STORED if relative.startswith("models/") else zipfile.ZIP_DEFLATED
                archive.write(folder / relative, prefix + relative, compress_type=compression)
        size = target.stat().st_size
        if size > limit:
            raise ValueError(f"{name} is {size} bytes, over the {limit} bytes a release asset may carry")
        written.append(Part(name, target, size, sha256_file(target), len(members[key])))
    return written


def describe_parts(parts: Sequence[Part]) -> list[str]:
    """One line per part: name, size in gigabytes, digest and member count."""
    return [f"{part.name}  {part.bytes / 1e9:.2f} GB  sha256 {part.sha256}  {part.files} files" for part in parts]


def plan(args: argparse.Namespace) -> list[str]:
    out: Path = args.out
    wheels = requirements_for(args.arch, args.gpu, args.bundle_ffmpeg)
    if args.wheels:
        wheel_step = f"2. take the wheel set already in {args.wheels}"
    else:
        wheel_step = f"2. pip download {' '.join(wheels)} for {ARCHITECTURES[args.arch]} / Python {PYTHON_TAG} (abi3 stated) into {out / 'wheels'}"
    if args.models and args.level:
        models_step = f"5. copy the models of the {args.level} level from {args.models}, with a lock holding their entries"
    elif args.models:
        models_step = f"5. copy every model from {args.models} (with models.lock.json)"
    else:
        models_step = "5. no models folder given; the app will report none"
    steps = [
        f"1. download {python_zip_url(args.arch)} and unzip into {out / 'python'}; write python311._pth",
        wheel_step,
        f"3. pip install --no-deps --no-index --target {out / 'Lib' / 'site-packages'} every wheel",
        f"4. copy the twinscribe package from {REPO_ROOT / 'twinscribe'}",
        models_step,
        f"6. copy ffmpeg from {args.ffmpeg} into {out / 'bin'}" if args.ffmpeg else (
            "6. ffmpeg comes from the imageio-ffmpeg package" if args.bundle_ffmpeg else "6. no ffmpeg given; decoding will need one on the search path"),
        f"7. write twinscribe.cmd, twinscribe-app.cmd, NOTICE and RECORD.txt under {out}",
    ]
    if args.pack:
        level = args.level or DEFAULT_LEVEL
        steps.append(f"8. pack the folder into {', '.join(PART_NAMES.values())} beside it, the second part carrying {detector_key(level, args.arch)}")
    if args.arch == "arm64":
        steps.append("note: CTranslate2 has no wheel for arm64; this build carries the ONNX detector only")
    if args.gpu and args.arch != "amd64":
        steps.append("note: the CUDA packages apply to amd64 only and are skipped")
    return steps


def build(args: argparse.Namespace) -> int:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    record: list[str] = []
    url = python_zip_url(args.arch)

    python_dir = out / "python"
    python_dir.mkdir(exist_ok=True)
    zip_path = out / url.rsplit("/", 1)[-1]
    print(f"downloading {url}")
    download(url, zip_path)
    record.append(f"{zip_path.name}  {sha256_file(zip_path)}  {url}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(python_dir)
    zip_path.unlink()
    pth = next(python_dir.glob("python*._pth"), python_dir / "python311._pth")
    pth.write_text("\n".join(PTH_LINES) + "\n", encoding="ascii")

    if args.wheels is not None:
        wheels = args.wheels
        print(f"taking the wheel set in {wheels}")
    else:
        wheels = out / "wheels"
        wheels.mkdir(exist_ok=True)
        print("downloading the wheel set")
        subprocess.run(
            [
                sys.executable, "-m", "pip", "download",
                "--dest", str(wheels),
                "--platform", ARCHITECTURES[args.arch],
                "--python-version", PYTHON_TAG,
                "--implementation", "cp",
                "--abi", f"cp{PYTHON_TAG.replace('.', '')}",
                "--abi", "abi3",
                "--abi", "none",
                "--only-binary=:all:",
                *requirements_for(args.arch, args.gpu, args.bundle_ffmpeg),
            ],
            check=True,
        )
    site = out / "Lib" / "site-packages"
    if site.exists():
        shutil.rmtree(site)
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
        keys = level_keys(args.level, args.arch) if args.level else None
        copied = copy_models(args.models, out / "models", keys)
        print(f"models copied: {', '.join(copied)}")
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


def pack_folder(args: argparse.Namespace) -> int:
    out: Path = args.out
    if not out.is_dir():
        print(f"folder not found: {out}", file=sys.stderr)
        return 2
    checker = detector_key(args.level or DEFAULT_LEVEL, args.arch)
    print(f"packing {out} into parts beside it; the second part carries {checker}")
    parts = pack(out, out.parent, checker)
    for line in describe_parts(parts):
        print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble a portable twinscribe folder for Windows, and pack it into the parts a release can carry.")
    parser.add_argument("--out", type=Path, required=True, help="the folder to assemble")
    parser.add_argument("--arch", default="amd64", choices=sorted(ARCHITECTURES), help="the Windows architecture")
    parser.add_argument("--gpu", action="store_true", help="add the CUDA runtime packages (amd64 only)")
    parser.add_argument("--bundle-ffmpeg", action="store_true", help="add the imageio-ffmpeg package as the decoder")
    parser.add_argument("--wheels", type=Path, default=None, help="a folder holding the wheel set already, downloaded earlier and recorded")
    parser.add_argument("--models", type=Path, default=None, help="a fetched models folder to copy in")
    parser.add_argument("--level", default=None, help="copy only the models this quality level needs, with a lock to match")
    parser.add_argument("--ffmpeg", type=Path, default=None, help="an ffmpeg executable to copy in")
    parser.add_argument("--pack", action="store_true", help="after assembling, write the parts beside the folder")
    parser.add_argument("--pack-only", action="store_true", help="write the parts of a folder assembled earlier; build nothing")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and touch nothing")
    args = parser.parse_args(argv)
    if args.dry_run:
        for step in plan(args):
            print(step)
        return 0
    if args.pack_only:
        return pack_folder(args)
    if args.models is not None and not args.models.is_dir():
        print(f"models folder not found: {args.models}", file=sys.stderr)
        return 2
    if args.wheels is not None and not any(args.wheels.glob("*.whl")):
        print(f"no wheels found in: {args.wheels}", file=sys.stderr)
        return 2
    if args.ffmpeg is not None and not args.ffmpeg.is_file():
        print(f"ffmpeg not found: {args.ffmpeg}", file=sys.stderr)
        return 2
    status = build(args)
    if status == 0 and args.pack:
        status = pack_folder(args)
    return status


if __name__ == "__main__":
    sys.exit(main())
