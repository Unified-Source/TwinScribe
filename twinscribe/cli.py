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
from twinscribe.models import STATUS_VERIFIED, find_models, verify_store
from twinscribe.outputs import render_all
from twinscribe.outputs.transcript_doc import load_document
from twinscribe.pipeline import Progress, discover_media, output_paths, run_batch
from twinscribe.profiles import DEFAULT_PROFILE, PROFILES, ModelsMissing, available_profiles, choose_profile


def build_parser() -> argparse.ArgumentParser:
    """The command line with its five commands."""
    parser = argparse.ArgumentParser(prog="twinscribe", description="Offline transcription with speaker labels.")
    parser.add_argument("--version", action="version", version=f"twinscribe {__version__}")
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
    run.add_argument("--author", default="", help="author written into the Word document properties")
    run.add_argument("--keep-audio", action="store_true", help="keep the decoded 16 kHz work file")
    run.add_argument("--no-recurse", action="store_true", help="do not descend into sub-folders")

    check = commands.add_parser("check", help="report ffmpeg, engine libraries, models and quality levels")
    check.add_argument("--models", type=Path, default=None, help="models root folder")
    check.add_argument("--verify", action="store_true", help="digest every model file against the lock")

    export = commands.add_parser("export", help="render text, Word and subtitles again from transcript documents")
    export.add_argument("documents", nargs="+", type=Path, help="*.transcript.json files")
    export.add_argument("--author", default="", help="author written into the Word document properties")

    verify = commands.add_parser("verify", help="open the verification screen for a review set")
    verify.add_argument("review_set", type=Path)
    verify.add_argument("--dark", action="store_true")
    return parser


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def command_check(args: argparse.Namespace) -> int:
    """Print what the machine offers and what the store holds."""
    ffmpeg = shutil.which("ffmpeg")
    print(f"twinscribe {__version__}")
    print(f"ffmpeg:          {ffmpeg or 'not found on the search path'}")
    for module, extra in (("faster_whisper", "engines"), ("sherpa_onnx", "engines"), ("PySide6", "app")):
        state = "installed" if _module_present(module) else f"not installed (extra '{extra}')"
        print(f"{module + ':':<17}{state}")
    models = find_models(args.models)
    print(f"models root:     {models.root}")
    print(models.describe())
    levels = available_profiles(models)
    if levels:
        print("quality levels:  " + ", ".join(p.name for p in levels))
    else:
        print("quality levels:  none; no level has all of its models present")
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
    models = find_models(args.models)
    try:
        profile = choose_profile(args.quality, models)
    except ModelsMissing as exc:
        print(str(exc), file=sys.stderr)
        return 2
    last_line = {"text": ""}

    def report(index: int, total: int, p: Progress) -> None:
        text = f"[{index + 1}/{total}] {sources[index].name}: {p.message} {100.0 * p.fraction:5.1f}%"
        if text != last_line["text"]:
            print(text, flush=True)
            last_line["text"] = text

    result = run_batch(
        sources,
        profile,
        models,
        out_dir=args.out,
        threads=args.threads,
        author=args.author,
        keep_audio=args.keep_audio,
        progress=report,
    )
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
        render_all(doc, paths.text, paths.docx, paths.subtitles, author=args.author)
        print(f"rendered {paths.text.name}, {paths.docx.name}, {paths.subtitles.name}")
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
    if args.command == "verify":
        from twinscribe.app.verify import main as verify_main

        verify_argv = [str(args.review_set)] + (["--dark"] if args.dark else [])
        return verify_main(verify_argv)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
