# Notes: fetching the models on first run

Companion to `first_run_models.md`. Records what was built, what was checked, and what was left.

## Built and checked

- `twinscribe/fetch.py`, the one module in the package that reaches the network, and only
  when asked; `tools/fetch_models.py` now calls it, with the same `--root`, `--only` and
  `--verify` as before. The catalogue carries a download size per model, taken from the lock
  of a fetched store and rounded: the Standard level is 2.4 GB, Careful 3.9 GB.
- The `fetch-models` command, run against the real sources for the two smallest models into
  an empty folder: the voice detector as a plain file from the sherpa-onnx releases (0.6 MB,
  entity tag recorded as the revision) and the audio tagger as two members of a tar.bz2
  archive downloaded once (10.5 MB; the Hugging Face sources answer with the repository
  commit, recorded the same way). The lock then verified through `tools/fetch_models.py
  --verify` with matching digests. A second run reported nothing to fetch.
- The models dialog and the window's offer, on the real display: a window with an empty
  store shows "Get models" beside a disabled Transcribe, the quality box reads "No models"
  with the tool tip naming the folder and the two ways out, and the dialog lists the six
  models of the Standard level with role, size, licence and source host, the folder, and the
  total. The dialog's own table follows the level check boxes and the folder as typed.
- The suite: 431 tests pass, 10 skip for want of a model or a library, on the fetch module
  against `file:` sources, the dialog with the fetch replaced by a stand-in that writes the
  store (done, cancelled, failed), the window's button and post-fetch refresh, and the command.

## Choices

- Explicit action only. Nothing fetches on its own: the window offers the dialog once on
  first start when no level is complete and a detector library is installed, and the
  Download button starts the transfer. Screenshot runs and windows built directly (tests, the
  review harness) get no offer.
- Where a first fetch goes. A store that already holds anything, or one named explicitly (the
  environment variable, the settings) and writable, is kept; otherwise a frozen build fetches
  into a `models` folder beside its executables, so that an unpacked folder carries its own
  models and is removed with them; otherwise the usual store when writable, else a folder
  under the application home. The folder is editable in the dialog before Download.
- The audio tagger is part of every level's fetch although no level requires it: 11 MB, and
  the scene pass uses it when present.
- Resumable by construction rather than by range requests: the lock is written after every
  file, a download goes to a `.part` name and is renamed only when complete, so a cancelled
  or failed fetch keeps the files it finished and fetches the rest next time. A partial file
  is never kept.
- The size in the catalogue is the number the person sees before agreeing to a transfer, so
  it comes from measured bytes rather than a guess, and it is rounded so that a rebuilt
  upstream file does not make it wrong by a digit.

## Not done, and why

- No mirror or fallback source: the sources are the ones the catalogue names, and a fetch
  that fails says so and can be run again.
- No range requests to continue a file that was cut off mid-way; a cut-off file is fetched
  again from the start. The largest single file is about 1.6 GB.
- The offer on first start is not tested through `main()`, which runs the event loop; the
  condition it uses (`needs_models`) and the dialog it opens are tested through the window.
