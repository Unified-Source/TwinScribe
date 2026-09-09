"""Command line: run the pipeline over recordings and folders, check the machine and the
model store, render outputs again from a transcript document, open the verification screen,
or open the application window.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from twinscribe import __version__
from twinscribe.audio import find_ffmpeg
from twinscribe.hardware import (
    BACKEND_CT2,
    BACKEND_ONNX,
    PREFERENCES,
    current_plan,
    describe_machine,
    probe_libraries,
    probe_machine,
    probe_nvidia_gpus,
)
from twinscribe.models import STATUS_VERIFIED, find_models, verify_store
from twinscribe.outputs import render_all
from twinscribe.outputs.export import DEFAULT_FORMATS, FORMATS, export_outputs
from twinscribe.outputs.transcript_doc import load_document
from twinscribe.pipeline import Progress, discover_media, output_paths, run_batch
from twinscribe.profiles import DEFAULT_PROFILE, PROFILES, ModelsMissing, available_profiles, select


def build_parser() -> argparse.ArgumentParser:
    """The command line with its five commands."""
    parser = argparse.ArgumentParser(prog="twinscribe", description="Offline transcription with speaker labels.")
    parser.add_argument("--version", action="version", version=f"TwinScribe {__version__}")
    commands = parser.add_subparsers(dest="command")

    app = commands.add_parser("app", help="open the application window")
    app.add_argument("paths", nargs="*", type=Path, help="recordings or folders to add to the library")
    app.add_argument("--models", type=Path, default=None, help="models root folder")
    app.add_argument("--dark", action="store_true", help="use the dark palette")
    app.add_argument("--shot", type=Path, default=None, metavar="OUT_PNG", help="render, save a screenshot, exit")

    run = commands.add_parser("run", help="transcribe recordings and folders")
    run.add_argument("paths", nargs="+", type=Path, help="recordings or folders")
    run.add_argument("--models", type=Path, default=None, help="models root folder")
    run.add_argument("--out", type=Path, default=None, help="output folder (default: beside each recording)")
    run.add_argument("--quality", default=DEFAULT_PROFILE, choices=[p.name for p in PROFILES], help="quality level")
    run.add_argument("--threads", type=int, default=None, help="threads per engine (default: min of 8 and the cores)")
    run.add_argument("--device", default="auto", choices=list(PREFERENCES),
                     help="auto uses a CUDA device when one is usable; cpu keeps every engine on the processor")
    run.add_argument("--author", default="", help="author written into the Word document properties")
    run.add_argument(
        "--speakers", type=int, default=None, metavar="N",
        help="the number of speakers, when it is known; by default the count comes from clustering by "
             "threshold, so a speaker the models cannot separate is missing from the labels rather than "
             "hidden inside another",
    )
    run.add_argument("--keep-audio", action="store_true", help="keep the decoded 16 kHz work file")
    run.add_argument("--no-recurse", action="store_true", help="do not descend into sub-folders")

    check = commands.add_parser("check", help="report the machine, ffmpeg, engine libraries, models, levels and the plan")
    check.add_argument("--models", type=Path, default=None, help="models root folder")
    check.add_argument("--verify", action="store_true", help="digest every model file against the lock")
    check.add_argument("--device", default="auto", choices=list(PREFERENCES), help="the preference the plan is made for")

    export = commands.add_parser("export", help="render text, Word and subtitles again from transcript documents")
    export.add_argument("documents", nargs="+", type=Path, help="*.transcript.json files")
    export.add_argument("--author", default="", help="author written into the Word document properties")
    export.add_argument("--out", type=Path, default=None, help="folder to write into (default: beside each document)")
    export.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS), choices=[key for key, _, _ in FORMATS],
                        help="what to write (default: text docx srt)")

    fetch = commands.add_parser("fetch-models", help="download the models a quality level needs into the store and pin their digests")
    fetch.add_argument("--root", type=Path, default=None, help="models root folder (default: the store in use, or a folder beside a frozen build)")
    fetch.add_argument("--level", nargs="+", default=["standard"], choices=[p.name for p in PROFILES], help="quality levels to complete (default: standard)")
    fetch.add_argument("--only", nargs="*", default=None, help="catalogue keys to fetch instead of levels")

    verify = commands.add_parser("verify", help="open the verification screen for a review set")
    verify.add_argument("review_set", type=Path)
    verify.add_argument("--dark", action="store_true")
    return parser


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _span(seconds: float) -> str:
    whole = max(0, int(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def _remaining(eta_s: float | None) -> str:
    if eta_s is None:
        return ""
    if eta_s < 10.0:
        return ", almost done"
    if eta_s < 90.0:
        return f", about {int(round(eta_s / 10.0)) * 10} s left"
    return f", roughly {int(round(eta_s / 60.0))} min left"


class LiveLine:
    """One status line that updates in place on a terminal, or prints on stage changes and
    every ten per cent elsewhere, so a run always shows movement."""

    SPINNER = "|/-\\"

    def __init__(self, stream) -> None:
        self._stream = stream
        self._tty = bool(getattr(stream, "isatty", lambda: False)())
        self._spin = 0
        self._last_key: tuple[str, str, int] | None = None
        self._open = False
        self._width = 0

    def update(self, prefix: str, report: Progress) -> None:
        percent = int(round(100 * report.fraction))
        text = f"{prefix}: {report.message} {percent:3d}%  {_span(report.elapsed_s)} elapsed{_remaining(report.eta_s)}"
        if self._tty:
            self._spin = (self._spin + 1) % len(self.SPINNER)
            line = f"{self.SPINNER[self._spin]} {text}"
            padding = " " * max(0, self._width - len(line))
            self._stream.write("\r" + line + padding)
            self._stream.flush()
            self._width = len(line)
            self._open = True
            return
        key = (prefix, report.message, percent // 10)
        if key != self._last_key:
            self._stream.write(text + "\n")
            self._stream.flush()
            self._last_key = key

    def finish(self) -> None:
        if self._open:
            self._stream.write("\n")
            self._stream.flush()
            self._open = False
            self._width = 0


def command_check(args: argparse.Namespace) -> int:
    """Print what the machine offers and what the store holds."""
    # The decoder as the pipeline resolves it: the search path, a bin folder beside the package,
    # or the executable a bundled imageio-ffmpeg carries.
    try:
        ffmpeg: str | None = find_ffmpeg()
    except FileNotFoundError:
        ffmpeg = None
    print(f"TwinScribe {__version__}")
    print(f"ffmpeg:          {ffmpeg or 'not found: not on the search path and no bundled decoder'}")
    for module, extra in (("faster_whisper", "engines"), ("sherpa_onnx", "engines"), ("PySide6", "app")):
        state = "installed" if _module_present(module) else f"not installed (extra '{extra}')"
        print(f"{module + ':':<17}{state}")
    models = find_models(args.models)
    print(f"models root:     {models.root}")
    print(models.describe())
    libraries = probe_libraries()
    backends = {BACKEND_CT2: libraries.whisper_ct2, BACKEND_ONNX: libraries.sherpa_onnx}
    levels = available_profiles(models, backends)
    if levels:
        print("quality levels:  " + ", ".join(p.name for p in levels))
    else:
        print("quality levels:  none; no level has all of its models present for the installed libraries")
    print()
    for line in describe_machine(probe_machine(), probe_nvidia_gpus(), libraries):
        print(line)
    print("plan (" + args.device + "):")
    for line in current_plan(args.device).describe():
        print("  " + line)
    status = 0
    if args.verify:
        checks = verify_store(models.root)
        for check in checks:
            print(f"  {check.status:<9} {check.key}/{check.file}  {check.detail}")
        if any(c.status != STATUS_VERIFIED for c in checks):
            status = 1
    return status


def command_run(args: argparse.Namespace) -> int:
    """Transcribe every recording found under the given paths."""
    sources = discover_media(args.paths, recursive=not args.no_recurse)
    if not sources:
        print("no recordings found under the given paths", file=sys.stderr)
        return 2
    if args.speakers is not None and args.speakers < 1:
        print("--speakers must be at least 1 when given", file=sys.stderr)
        return 2
    models = find_models(args.models)
    plan = current_plan(args.device, args.threads)
    try:
        selection = select(args.quality, models, plan.backends)
    except ModelsMissing as exc:
        print(str(exc), file=sys.stderr)
        return 2
    profile = selection.profile
    for line in plan.describe():
        print(line)
    print(f"detector model: {selection.detector} ({selection.backend})")
    print(f"speakers: {'fixed at ' + str(args.speakers) if args.speakers else 'by clustering threshold'}")
    live = LiveLine(sys.stdout)

    def report(index: int, total: int, p: Progress) -> None:
        live.update(f"[{index + 1}/{total}] {sources[index].name}", p)

    def outcome(index: int, _outcome) -> None:
        live.finish()

    result = run_batch(
        sources,
        profile,
        models,
        out_dir=args.out,
        threads=args.threads,
        author=args.author,
        keep_audio=args.keep_audio,
        progress=report,
        plan=plan,
        preference=args.device,
        on_outcome=outcome,
        speakers=args.speakers,
    )
    live.finish()
    for outcome in result.outcomes:
        if outcome.ok and outcome.result is not None:
            print(
                f"done   {outcome.source}  {outcome.result.duration_s:.0f} s of audio, "
                f"{outcome.result.marks} review marks, {outcome.elapsed_s:.0f} s"
            )
        elif outcome.cancelled:
            print(f"skip   {outcome.source}  cancelled")
        else:
            print(f"FAILED {outcome.source}  {outcome.error_class}: {outcome.error}", file=sys.stderr)
    print(f"batch record: {result.record_path}")
    return 0 if not result.failures else 1


def command_fetch_models(args: argparse.Namespace) -> int:
    """Fetch what the chosen levels lack (or the named models) and report the store afterwards."""
    from twinscribe.fetch import STAGE_DONE, STAGE_EXTRACT, Cancelled, Progress, fetch_specs, missing_for_levels, proposed_root

    models = find_models(args.root)
    root = Path(args.root) if args.root is not None else proposed_root(models)
    if args.only:
        from twinscribe.models import spec_for

        specs = [spec_for(key) for key in args.only]
    else:
        specs = missing_for_levels(args.level, find_models(root))
    if not specs:
        print(f"nothing to fetch: {', '.join(args.level)} complete under {root}")
        return 0
    total_mb = sum(spec.size_mb for spec in specs)
    size = f"{total_mb / 1000:.1f} GB" if total_mb >= 1000 else f"{total_mb} MB"
    print(f"fetching {len(specs)} model(s), about {size}, into {root}")
    last: dict[str, int] = {}

    def show(progress: Progress) -> None:
        if progress.stage == STAGE_DONE:
            return
        percent = int(100 * progress.fraction) if progress.fraction is not None else -1
        if last.get(progress.file) != percent:
            last[progress.file] = percent
            size = "extracting" if progress.stage == STAGE_EXTRACT else (f"{progress.done_bytes / 1e6:.0f} MB" if percent < 0 else f"{percent:3d}%")
            print(f"\r  {min(progress.files_done + 1, progress.files_total)}/{progress.files_total} {progress.key}/{progress.file} {size}", end="", flush=True)

    try:
        fetch_specs(specs, root, progress=show, log=lambda text: print("\n" + text, end=""))
    except Cancelled:
        print("\ncancelled")
        return 1
    print()
    store = find_models(root)
    print(store.describe())
    print("quality levels:  " + (", ".join(p.name for p in available_profiles(store)) or "none"))
    print("Attribution required by the licences of the models fetched:")
    for spec in specs:
        print(f"  {spec.title}: {spec.credit} ({spec.licence})")
    return 0


def command_export(args: argparse.Namespace) -> int:
    """Write text, Word and subtitles again beside each transcript document."""
    status = 0
    for path in args.documents:
        try:
            doc = load_document(path)
        except (OSError, ValueError) as exc:
            print(f"cannot read {path}: {exc}", file=sys.stderr)
            status = 1
            continue
        source_name = str(doc.get("source", {}).get("name", path.stem))
        paths = output_paths(path.parent / source_name)
        stem = paths.transcript.name[: -len(".transcript.json")]
        sources = {"transcript": path, "review": paths.review, "run": paths.run}
        written = export_outputs(doc, args.out if args.out is not None else path.parent, stem, args.formats, author=args.author, sources=sources)
        print("rendered " + ", ".join(p.name for p in written))
    return status


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit status."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command is None or args.command == "app":
        from twinscribe.app.main import main as app_main

        app_argv: list[str] = []
        if args.command == "app":
            app_argv += [str(p) for p in args.paths]
            if args.models is not None:
                app_argv += ["--models", str(args.models)]
            if args.dark:
                app_argv.append("--dark")
            if args.shot is not None:
                app_argv += ["--shot", str(args.shot)]
        return app_main(app_argv)
    if args.command == "run":
        return command_run(args)
    if args.command == "check":
        return command_check(args)
    if args.command == "export":
        return command_export(args)
    if args.command == "fetch-models":
        return command_fetch_models(args)
    if args.command == "verify":
        from twinscribe.app.verify import main as verify_main

        verify_argv = [str(args.review_set)] + (["--dark"] if args.dark else [])
        return verify_main(verify_argv)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
