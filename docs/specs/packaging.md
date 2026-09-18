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

## 3. The portable folder, and its parts

`tools/build_portable.py` as documented in `Portable_Layout.md`: the embeddable interpreter,
the wheel set (downloaded, or taken from a folder given so that a build repeats the versions a
machine was checked with), the package, the models of one quality level with a lock holding
their entries only, the decoder, the launchers and `RECORD.txt`.

A release carries files of at most two gibibytes and the folder with its models is larger, so
the tool also packs the folder into parts, each under that limit: the first unpacks to the
folder itself, with everything but the checker's model and the CUDA runtime packages, and an
empty `home`; the second holds the checker's model; the third, wanted only for an NVIDIA
device, the CUDA packages. The second and third are laid out relative to the folder, and the
launchers unpack any part placed beside them before starting the program, once, renaming a
part unpacked so that it is not unpacked again and stopping the start when an unpack fails.
A person unzips the first part, puts the others inside the folder it gives, and double-clicks
the launcher. Model files are stored in the parts without compression. A part over the limit,
or a folder without the checker's model, fails the pack; the wheel set, byte-code caches and
the contents of `home` are never packed.

## 4. Installation paths, in the README

Three ways, in order of least to most effort for the person installing: the portable parts
from the Releases page (unzip the first, put the second and, for an NVIDIA device, the third
inside the folder, double-click `twinscribe-app.cmd`; nothing is fetched), the executables
folder (unzip, double-click `twinscribe-app.exe`; the models are fetched on first start), and
from source with the project's extras. Each names where the models go and how they get there.

## 5. Tests

The icon at every size with a transparent corner, a filled centre and a bright trace; the
.ico writer's header and directory arithmetic on small images; the frozen fallbacks with a
fake `sys.frozen` and executable path; the launchers' contract (the three folders beside
them, the parts unpacked once, before the start); a store cut down to a level with a lock to
match; and the split of an assembled folder into parts, with nothing in two parts, nothing
shipped that should not be, the model files stored, and a part too large or a folder without
the checker's model refused.
