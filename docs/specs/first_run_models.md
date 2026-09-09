# Specification: fetching the models on first run

Read `CONVENTIONS.md` first, then `packaging.md`.

Deliver `twinscribe/fetch.py` with `tests/test_fetch.py`, `twinscribe/app/models_dialog.py`
with `tests/test_app_models_dialog.py`, the `fetch-models` command in `twinscribe/cli.py`,
`tools/fetch_models.py` as a front for the module, the sizes in the catalogue, the amended
network rule in `CONVENTIONS.md`, and the README's installation section.

## 1. Why

The built folders carry every library and the decoder but not the models: 2.4 GB for the
Standard level, more for the others, and the design records their sources, digests and
licences in a lock rather than in the repository. A person who downloads the executables
therefore had nothing to transcribe with until they had run a tool from a source checkout.
The window now offers the fetch itself, and the command line has the same.

## 2. The rule

`CONVENTIONS.md` said no network access at runtime anywhere in the package. The amended rule
keeps that for everything except one module, `twinscribe/fetch.py`, which reaches the sources
the catalogue names on an explicit action only: the Download button of the models dialog, or
the `fetch-models` command. Nothing runs it on its own; the engines still load from a local
folder with the offline variables set. The exception is stated in the conventions, the README
and `Portable_Layout.md`, so a reader who relied on the old rule finds the new one.

## 3. The module

- `download(url, target, progress, cancel)` streams a URL to `<target>.part` in 1 MB chunks
  with a fixed user agent and a 60 s timeout, asks `cancel` between chunks, and renames the
  file into place only when complete; any failure or cancellation removes the partial file.
  It returns the upstream revision when a header carries one (the Hugging Face commit, else
  the entity tag).
- `extract_members(archive, targets, written, cancel)` copies the wanted members out of a tar
  archive by file name, whatever top folder the archive uses, in one streaming pass from start
  to end, so a compressed archive is decompressed once however many members are wanted; each
  member goes to a temporary name and is renamed when complete, and `written` is told its name.
- `missing_sources(spec, root, lock)` names the files of a model that are absent or whose
  digest no longer matches the lock; `fetch_specs(specs, root, progress, cancel, log)`
  fetches those and only those, downloading each archive once into a temporary folder under
  the root and writing every wanted member of it in one pass, writing the lock after every file
  so that an interrupted fetch keeps what it finished, and removing the temporary folder at the
  end. It returns the lock.
- `Progress` carries the model key, the file, the bytes so far and the total (or zero when
  the source does not say), the count of files done out of the files to fetch, and a stage:
  `download` while bytes of a file or an archive arrive (the archive under its own name),
  `extract` while members are written out of an archive, `done` once a file is in place and
  counted. The fraction is defined only while downloading.
- `specs_for_level(name)` is the level's required models with its CTranslate2 detector and
  the audio tagger, in catalogue order; `missing_for_levels(names, store)` is what the store
  lacks of them; `proposed_root(store, explicit)` is where a first fetch goes: the store in
  use when it holds anything or was named explicitly (the environment variable, the
  settings) and can be written, else a `models` folder beside a frozen build's executable,
  else the store in use when writable, else one under the application home.
- The catalogue gains `size_mb` per model, the download size as the lock records it,
  rounded, so the dialog and the command can say what a choice costs before it starts.

## 4. The dialog

`ModelsDialog(models, parent, fetch, explicit)`: a sentence saying what is fetched and from
where; one check box per quality level (Standard on by default); a table of the models the
chosen levels lack with role, size, licence and source host, the credit as a tool tip; the
folder, editable and browsable; a summary line with the count and the total; Download,
Cancel while running, Later. Download runs `fetch_specs` in a `QThread` that forwards
progress and the outcome as signals: the bar shows the overall fraction across files (busy
while extracting or
when a size is unknown), the status line names the file. Done, cancelled and failed all end
with the outcome in the status line, the controls enabled again, the table recomputed, and
`fetched(root)` emitted so the window reads the store; a cancelled or failed fetch keeps what
it finished and Download continues from there. Closing the dialog while a fetch runs cancels
it and waits.

## 5. The window and the command line

- A "Get models" button in the top bar, visible only while no level is complete although a
  detector library is installed; the quality box and the Transcribe button say the same in
  their tool tips, and Transcribe without a level opens the dialog instead of a message.
- `open_models()` connects `fetched` to a handler that remembers the folder in the settings
  when it is not the store already in use, reads the store again, refreshes the levels, saves
  the settings and reports the levels in the status bar.
- `main()` opens the dialog once the window is shown when no level is complete and a
  detector library is installed, and not on a screenshot run. Windows built directly, as the
  tests and the review harness build them, get no offer.
- `twinscribe fetch-models [--root FOLDER] [--level ...] [--only KEY ...]` fetches what the
  levels lack (Standard by default) or the named models, with a line per file on the
  console, then prints the store, the levels and the attributions the licences require.
  `tools/fetch_models.py` keeps its interface (`--root`, `--only`, `--verify`) over the module.

## 6. Tests

`tests/test_fetch.py`, against `file:` sources in a temporary folder: a plain file and two
members of a tar.bz2 archive placed and recorded in the lock with matching digests, progress
and log lines seen, no temporary folder or partial file left; a second fetch skips what is
present and pinned and fetches again what was changed; cancellation before the first chunk
leaves no file and an empty lock; a failed download leaves no partial; the level plans and
their sizes; what the store lacks as files appear; the proposed root for a store, a frozen
build and an explicitly named folder.

`tests/test_app_models_dialog.py`, offscreen with a stand-in fetch that writes the store: the
table follows the levels and the folder; a fetch in the thread ends with the store complete,
`fetched` emitted and the outcome shown; cancellation and failure are reported and leave the
dialog usable; the window shows Get models while no level is complete, reads the store again
after `fetched`, remembers a new folder and not the one in use, and offers nothing without a
detector library. `tests/test_cli.py` runs the command with the module's fetch replaced.
