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
- The models are not inside the zip: 2.3 GB for the Standard level alone, and the fetch tool
  with its lock is the way to get them with their digests and licences recorded.

## Not done, and why

- No installer (a setup program that writes to Program Files and the Start menu): the design's
  machines have no administrator rights, and an unpacked folder is what runs there.
- No signing of the executables; unsigned binaries draw a warning from Windows on first run.
  Signing needs a certificate and a decision that is not the tool's.
- Linux and macOS builds are the source path with the same extras, as before.
