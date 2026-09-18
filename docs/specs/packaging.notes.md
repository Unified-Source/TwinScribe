# Notes: icon and packaging

Companion to `packaging.md`. Records what was built, what was checked, and what was left.

## Built and checked

- The executables folder, from `tools/twinscribe.spec` through PyInstaller 6 into one folder
  with `twinscribe.exe` (console) and `twinscribe-app.exe` (windowed) sharing `_internal`:
  about 2.4 GB with the CUDA runtime packages, 1.4 GB zipped; 4.7 GB with the Standard
  level's models beside the executables. PyInstaller warned that `onnxruntime.quantization`
  could not be collected because the optional `onnx` package is absent; nothing here uses it.
- Checked from the built folder on a desktop with an NVIDIA device: `--version`; `check` with
  a `models` folder beside the executable (found through the frozen fallback, the Standard
  level offered, the plan choosing the device); the window opened on a recording and rendered
  a screenshot; a `home` folder beside the executable received the settings; and a 232 s MP3
  transcribed through the executable in 47 s with all six outputs, which exercises the bundled
  decoder carried by imageio-ffmpeg.
- The portable folder, assembled by `tools/build_portable.py --gpu --bundle-ffmpeg` with the
  Standard level's models: the embeddable interpreter, 34 wheels unpacked into
  `Lib/site-packages` (3.0 GB), the models (2.3 GB), the launchers, `RECORD.txt` with 35
  digests. The wheel set (1.7 GB) stays in `wheels/` as the tool records it; a copy handed
  to a person can drop that folder. Checked through its own launcher: `--version`, `check`
  (the Standard level offered, the device seen), the window opened on a recording through
  `twinscribe.cmd app --shot`, and the settings written under the folder's own `home`.
- The parts, 2026-09-18, from the same wheel set and the Standard level's models: the first
  1.17 GB (6,468 files), the second 1.62 GB (the checker's five files, stored), the third
  1.36 GB (the CUDA packages, 55 files); each well under the two-gibibyte limit. Checked on
  the desktop with an NVIDIA device, through the launchers only: the assembled folder gives
  its version and its check; the first part unzipped as a person would (54 s through the
  platform's own extraction) gives a folder whose check names the checker absent and no
  level; with the second and third parts put inside, the first start unpacked both and ran
  the check in 24 s, the parts renamed `.unpacked`, all six models present, the levels
  Standard and Laptop, the detector planned for the device; the second start took no time
  on unpacking; `check --verify` found every file of the six models verified; a 654 s chapter
  transcribed through `twinscribe.cmd run` in 127 s with ten review marks and all six outputs
  beside it, the batch record and the history under the copy's own `home`; and a copy with
  the first two parts alone started and checked clean. That copy still planned the detector
  for the device, because the desktop has the CUDA toolkit installed system-wide and the
  libraries load from there; the processor-only path of a machine without them was not
  exercised on this desktop.
- The C++ runtime, found before publishing: a scan of the 629 libraries and extension modules
  in the assembled folder for their imports showed `msvcp140.dll` wanted by 204 of them,
  CTranslate2, onnxruntime, sherpa-onnx and the media libraries among them, with the folder's
  only copies inside the PySide6 and shiboken6 folders, which the loader searches only once
  PySide6 has been imported; the embeddable interpreter ships `vcruntime140.dll` and
  `vcruntime140_1.dll` alone. On the desktop, where the runtime is installed system-wide, the
  engines loaded from the system copy and every check passed; on a machine that never had it
  installed they would not have loaded, and installing it needs administrator rights. The
  build now copies the whole set out of the PySide6 wheel (version 14.44.35211.0, newer than
  the interpreter's 14.38.33126.1, which it replaces) into the interpreter's folder, which the
  loader searches for every library's dependencies, and records each file with its version,
  read from the file's own version resource so that an older copy is never installed over a
  newer one. Checked by listing the runtime modules a process of the folder has loaded after
  importing the engines: every one from the folder's `python` folder, none from the system.
- `check --verify` now digests the models the store holds, whole or in part, rather than
  every catalogue model: a store with one level's models reported the other five models'
  files as missing and returned a failure although all of its own files verified, which a
  portable copy made a certainty.
- The `check` command now reports the decoder as the pipeline resolves it (the search path, a
  bin folder beside the package, or the bundled executable) rather than the search path alone,
  because the earlier line said "not found" while decoding worked from the bundled copy.

## Choices

- Two executables from one analysis rather than two builds: the entry script is the command
  line, which opens the window when it is given no command, so the windowed executable is the
  same program without a console.
- A frozen build looks beside its executable for `models` and `home` before the usual places,
  after the environment variables, so that an unpacked folder behaves like the portable one
  without launchers; without those folders it behaves like an installed program.
- The models are not inside the executables' zip: 2.3 GB for the Standard level alone, and
  the fetch tool with its lock is the way to get them with their digests and licences recorded.
- The portable folder is published as parts rather than trimmed to fit one file. A release
  carries files of at most two gibibytes; the Standard level's models alone are past that, so
  no single file could hold the folder, and a smaller checker would have been a different
  product from the one the bench measured. The cut follows what a machine needs: the folder
  with the published engine and the small models in the first part, the checker's model in
  the second, the CUDA runtime packages in a third that only a machine with an NVIDIA device
  wants; the first two parts are the download for the laptops the tool is for.
- The launchers unpack the parts rather than the person unzipping three archives into one
  folder. Unzipping the first part gives a folder that starts; the others go inside it as they
  are, and the first start unpacks them, so no path has to be typed into an extraction dialog.
  A part unpacked is renamed, not deleted: the file is the person's download, and a stale
  copy beside the launcher costs a rename to unpack again.
- Model files are stored in the parts without compression: weights hardly compress, and an
  unpack of a stored member is a copy, so the second part unpacks in seconds.
- The wheel set is taken from the folder of an earlier build rather than downloaded again, so
  that the parts carry the versions the executables and the bench were checked with, and the
  record lists the same digests.

## Not done, and why

- No installer (a setup program that writes to Program Files and the Start menu): the design's
  machines have no administrator rights, and an unpacked folder is what runs there.
- No signing of the executables; unsigned binaries draw a warning from Windows on first run.
  Signing needs a certificate and a decision that is not the tool's.
- Linux and macOS builds are the source path with the same extras, as before.
- The arm64 portable folder (the ONNX detector only) has not been assembled or packed; the
  tool plans it, and the pack would carry `whisper-turbo-onnx` in its second part.
