# Conventions for every module in twinscribe

These apply to all code, tests and documentation in this repository. They are short because
they are absolute.

## Provenance

- Every line is written fresh from the specifications in this folder and from
  `docs/Design_and_Findings_2026-09-06.md`. Nothing is copied or adapted from any other
  codebase, and no other codebase is consulted while writing.
- Third-party code enters only by vendoring from its upstream source with its licence file
  beside it, and is recorded in `NOTICE`.
- No file may name any organisation, person, machine, project or document outside this
  repository. No absolute paths. No hostnames. No internal identifiers of any kind.
- Nothing in the repository describes how or with what tooling it was written.

## Language and layout

- Python 3.11 or later. Type hints on every public function. Dataclasses for records.
- Standard library first; `numpy` is allowed everywhere; the engine libraries
  (`faster-whisper`, `sherpa-onnx`) are imported lazily inside the functions that need them,
  so the package imports cleanly without them and tests can skip.
- `PySide6` is imported only under `twinscribe/app/`.
- No network access at runtime anywhere in the package, with one exception: `twinscribe/fetch.py`
  fetches the catalogue models, on an explicit action only (the window's Get models dialog, the
  `fetch-models` command), and records every file in the lock. The engines never reach the
  network: model loading is by local directory path, and the environment variables that keep
  Hugging Face tooling offline are set before any engine import.
- Deterministic where possible; every timing is recorded beside what was timed.

## Writing

- Comments and docstrings are technical and impersonal. Say what the code does and why the
  non-obvious choice was made. No first person, no narrative, no history.
- ASCII hyphen only. Never an en dash or an em dash, in code, comments, docs or strings.
- British spelling in prose (normalise, colour); library and API names as they are.
- Every module opens with a docstring stating its purpose in one to three sentences.

## Tests

- `pytest`, under `tests/`, runnable with `.venv/Scripts/python -m pytest`.
- Every metric has fixtures with hand-derived expected values written in the test, and at
  least one property test where the maths allows (symmetry, identity, bounds).
- Tests that need an engine library or a model skip with a clear reason when it is absent.
- Tests write only under `tmp_path`.

## What a worker delivers

- The modules named in its specification, the tests, and a short `docs/specs/<name>.notes.md`
  recording any deviation from the specification and why. Nothing else is created or
  modified. Workers do not run `git`.
