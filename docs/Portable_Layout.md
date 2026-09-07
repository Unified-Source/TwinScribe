# Running without an installation

twinscribe is meant to run on an ordinary laptop with no administrator rights and no network:
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

## Assembling it

`tools/build_portable.py` assembles the folder on a connected machine:

1. downloads the pinned embeddable interpreter zip and records its digest;
2. downloads the wheel set for the target platform with `pip download` into `wheels/` (the
   `--platform`, `--python-version`, `--only-binary` and `--abi abi3` arguments PySide6 needs
   are set for it);
3. unpacks the wheels into `Lib/site-packages` with `pip install --no-deps --target`;
4. copies this package, the models folder given with `--models` (with its lock), the ffmpeg
   executable given with `--ffmpeg`, the launchers and `NOTICE`;
5. writes `RECORD.txt` with a digest for every file it placed.

`--arch amd64` (the default) or `--arch arm64` chooses the Windows architecture; the arm64
build carries the ONNX detector only, because CTranslate2 publishes no wheel for it. `--gpu`
adds the CUDA runtime packages (cuBLAS and cuDNN) so that CTranslate2 can use an NVIDIA
device with nothing installed; `--bundle-ffmpeg` adds the imageio-ffmpeg package as the
decoder instead of a copied executable. `--dry-run` prints the plan without downloading
anything. The tool is not part of the package and is the one place that reaches the network,
together with `tools/fetch_models.py`.

On Linux and macOS the equivalent is a virtual environment with the same extras
(`pip install .[engines,app]`, plus `cuda` or `ffmpeg` as wanted) and the models folder
beside it; the launchers are one-line shell scripts setting the same three variables.

## The models

`tools/fetch_models.py --root models` fetches every catalogue model from the sources the
catalogue names (or `--only <key>` for some), extracts archive members, records the digest,
size, URL, revision and licence of every file in `models/models.lock.json`, and skips files
already present whose digest matches the lock. `tools/fetch_models.py --root models --verify`
digests the store against the lock without fetching; `twinscribe check --verify` does the same
from the package.

Attribution: the catalogue records the licence and credit of every model; the Parakeet and
TitaNet models (CC BY 4.0) require attribution, which the NOTICE file carries.

## What was and was not exercised

The fetch tool has fetched the standard-profile set (the published engine, the turbo
detector, the voice detector and the two speaker models) into a store beside the package and
verified every file against the lock; the two other detectors remain to be fetched where they
are wanted. The build tool ran in `--dry-run` form only. No portable folder has been
assembled or started yet; that belongs with the lab hardware step, on the platform the
laptops actually run.
