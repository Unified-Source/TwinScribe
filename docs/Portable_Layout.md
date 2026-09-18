# Running without an installation

TwinScribe is meant to run on an ordinary laptop with no administrator rights and no network:
one folder that holds an interpreter, the libraries, the models and ffmpeg, started from a
launcher beside them. Nothing is installed and nothing is written outside the folder except,
by choice, the outputs beside the recordings.

## The folder

```
twinscribe-portable/
  python/                  the embeddable Windows interpreter, unzipped
  python/python311._pth    adds Lib\site-packages and the parent folder to the import path
  Lib/site-packages/       every wheel, unpacked: numpy, PySide6, faster-whisper, CTranslate2,
                           sherpa-onnx, onnxruntime and their dependencies
  twinscribe/              this package
  models/                  one folder per model, plus models.lock.json with the digests
  bin/ffmpeg.exe           the decoder, from a build whose origin is recorded in RECORD.txt
  home/                    settings, decoded work files, batch records (TWINSCRIBE_HOME)
  twinscribe.cmd           command line: twinscribe.cmd run <folder> ...
  twinscribe-app.cmd       the window, without a console
  RECORD.txt               where every part came from, its version and its digest
  NOTICE                   the attributions
```

The launchers set `TWINSCRIBE_HOME`, `TWINSCRIBE_MODELS` and the search path to the folders
beside them, so the same folder works from any drive or share, and two copies never share state.
`twinscribe.cmd` is the command line (`twinscribe.cmd run <folder>` transcribes with progress
in the console; no arguments opens the window); `twinscribe-app.cmd` opens the window without
a console, and that window writes anything it would have printed, and any unhandled error, to
`home\twinscribe.log`. The checkout carries the same two launchers for the project's own
virtual environment.

## The parts on the Releases page

A release carries files of at most two gibibytes, and the folder with the models of the
Standard level inside is larger than that, so it is published as parts, each under the limit:

- `twinscribe-portable-win64-part1.zip` unzips to the folder above with everything but the
  checker's model and the CUDA runtime packages: the interpreter, the libraries, the decoder,
  the package, the published engine, the voice detector, the speaker models and the audio
  tagger, an empty `home`, the launchers and the records.
- `twinscribe-portable-win64-part2.zip` holds the checker's model, the second engine, laid out
  relative to the folder (`models\whisper-large-v3-turbo-ct2\...`).
- `twinscribe-portable-win64-cuda.zip` holds the CUDA runtime packages (cuBLAS and cuDNN, laid
  out under `Lib\site-packages\`), wanted only on a machine with an NVIDIA device; without it
  every engine runs on the processor and the plan says so.

The launchers unpack any `twinscribe-portable-*.zip` placed beside them before starting the
program, once: a person unzips the first part, puts the second (and the third, for an NVIDIA
device) inside the folder it gives, next to `twinscribe-app.cmd`, and double-clicks the
launcher. The first start unpacks the parts into the folder, which takes a few minutes, then
opens the window; a part unpacked is renamed with `.unpacked` at the end so that it is not
unpacked again, and can be deleted. A part whose unpack fails, as a download cut short does,
stops the start with the reason on screen. Unzipping the second and third parts into the
folder by hand comes to the same thing. In the parts, model files are stored without
compression, since weights hardly compress and unpacking them is then a copy.

## Assembling it

`tools/build_portable.py` assembles the folder on a connected machine:

1. downloads the pinned embeddable interpreter zip and records its digest;
2. downloads the wheel set for the target platform with `pip download` into `wheels/` (the
   `--platform`, `--python-version`, `--only-binary` and `--abi abi3` arguments PySide6 needs
   are set for it), or takes a wheel set downloaded earlier when `--wheels` names its folder,
   so that a build repeats the versions a machine was checked with;
3. unpacks the wheels into `Lib/site-packages` with `pip install --no-deps --target`;
4. copies this package, the models folder given with `--models` (with its lock; `--level`
   cuts it down to the models that quality level needs, with a lock holding their entries
   only), the ffmpeg executable given with `--ffmpeg`, the launchers and `NOTICE`;
5. writes `RECORD.txt` with a digest for every file it placed;
6. with `--pack`, writes the parts beside the folder and prints each one's size and digest;
   `--pack-only` does that for a folder assembled earlier. The wheel set, byte-code caches
   and the contents of `home` are never packed; a part over the limit, or a folder without
   the checker's model, fails the pack.

`--arch amd64` (the default) or `--arch arm64` chooses the Windows architecture; the arm64
build carries the ONNX detector only, because CTranslate2 publishes no wheel for it. `--gpu`
adds the CUDA runtime packages (cuBLAS and cuDNN) so that CTranslate2 can use an NVIDIA
device with nothing installed; `--bundle-ffmpeg` adds the imageio-ffmpeg package as the
decoder instead of a copied executable. `--dry-run` prints the plan without downloading
anything. The tool is not part of the package. Inside the package, the fetch of the models
(`twinscribe/fetch.py`, behind the `fetch-models` command, the window's Get models dialog and
`tools/fetch_models.py`) is the only code that reaches the network, and only when asked.

On Linux and macOS the equivalent is a virtual environment with the same extras
(`pip install .[engines,app]`, plus `cuda` or `ffmpeg` as wanted) and the models folder
beside it; the launchers are one-line shell scripts setting the same three variables.

## The models

`python -m twinscribe fetch-models --root models` fetches what the Standard level lacks (other
levels with `--level`, named models with `--only`); `tools/fetch_models.py --root models` fetches
every catalogue model (or `--only <key>` for some). Both extract archive members, record the digest,
size, URL, revision and licence of every file in `models/models.lock.json`, and skip files
already present whose digest matches the lock. `tools/fetch_models.py --root models --verify`
digests the store against the lock without fetching; `twinscribe check --verify` does the same
from the package.

Attribution: the catalogue records the licence and credit of every model; the Parakeet and
TitaNet models (CC BY 4.0) require attribution, which the NOTICE file carries.

## What was and was not exercised

The amd64 folder has been assembled with the CUDA packages, the bundled decoder and the
Standard level's models, started through its own launchers, and packed into the parts, which
were unpacked again through the launcher and checked; `docs/specs/packaging.notes.md` records
each check. The arm64 build has not been assembled; its wheel set is the tool's plan only.
