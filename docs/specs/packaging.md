# Specification: icon and packaging

Read `CONVENTIONS.md` first, then `../Portable_Layout.md` and `../Platforms.md`.

Deliver `tools/build_exe.py`, `tools/twinscribe.spec`, `tools/pyinstaller_entry.py`, the
frozen-build fallbacks in `twinscribe/models.py` and `twinscribe/paths.py` with `tests/test_paths.py`,
and the installation section of the README. The icon (`twinscribe/app/app_icon.py`,
`tools/make_icon.py`, `assets/twinscribe.ico`, both windows) landed with the README walkthrough.

## 1. The icon

The mark is specified in `icon.md`. `app_icon()` gives it at every standard size from 16 to
256 pixels; both windows set it. `tools/make_icon.py` renders the same painting to
`assets/twinscribe.ico` (every size, PNG-compressed entries) for the executables, to
`docs/images/icon.png`, and beside the wordmark to `docs/images/logo.png` and `logo-dark.png`
for the documentation; the .ico is tracked as binary.

## 2. The executables

`tools/build_exe.py` drives PyInstaller from `tools/twinscribe.spec` to one folder,
`dist/twinscribe/`, holding `twinscribe.exe` (the console command line) and
`twinscribe-app.exe` (the window, no console), sharing every library: PySide6, CTranslate2 and
faster-whisper with the CUDA runtime packages, sherpa-onnx, onnxruntime, the decoder carried by
imageio-ffmpeg, and this package with its data files. The entry script is the command line,
which opens the window when given no command. `--models` copies a fetched models folder beside
the executables; `--zip` packs the folder. PyInstaller is installed into the interpreter when
missing; this tool is outside the package and, with the fetch tools, the only code that
reaches the network.

A frozen build looks for `models` and `home` beside its executable before the usual places,
so a folder unpacked anywhere runs as a portable copy; without a `home` folder beside it the
settings go to the per-user application data folder as before.

## 3. The portable folder

`tools/build_portable.py` as documented in `Portable_Layout.md`: the embeddable interpreter,
the wheel set, the package, the models, the decoder, the launchers and `RECORD.txt`. This
specification adds nothing to it beyond being run and recorded.

## 4. Installation paths, in the README

Three ways, in order of least to most effort for the person installing: the portable folder
(unzip, double-click `twinscribe-app.cmd`), the executables folder (unzip, double-click
`twinscribe-app.exe`), and from source with the project's extras. Each names where the models
go and how to fetch them.

## 5. Tests

The icon at every size with a transparent corner, a filled centre and a bright trace; the
.ico writer's header and directory arithmetic on small images; the frozen fallbacks with a
fake `sys.frozen` and executable path.
