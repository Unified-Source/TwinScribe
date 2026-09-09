# -*- mode: python ; coding: utf-8 -*-
# PyInstaller specification: one folder holding two executables, the console command line
# (twinscribe.exe) and the windowed application (twinscribe-app.exe), sharing every library.
# Run through tools/build_exe.py, which resolves the paths below relative to the repository.

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

repo = Path(SPECPATH).resolve().parent

datas = [(str(repo / "twinscribe" / "text" / "_whisper_normalizer"), "twinscribe/text/_whisper_normalizer")]
binaries = []
hiddenimports = collect_submodules("twinscribe")
# Packages that carry shared libraries or data files beside their Python code.
for package in ("sherpa_onnx", "ctranslate2", "faster_whisper", "onnxruntime", "imageio_ffmpeg", "av", "tokenizers", "huggingface_hub", "nvidia"):
    try:
        d, b, h = collect_all(package)
    except Exception:  # noqa: BLE001 - an absent optional package is left out
        continue
    datas += d
    binaries += b
    hiddenimports += h

icon = repo / "assets" / "twinscribe.ico"
icon_arg = str(icon) if icon.is_file() else None

a = Analysis(
    [str(repo / "tools" / "pyinstaller_entry.py")],
    pathex=[str(repo)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "notebook", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

cli = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="twinscribe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=icon_arg,
)
windowed = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="twinscribe-app",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=icon_arg,
)
coll = COLLECT(
    cli,
    windowed,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="twinscribe",
)
