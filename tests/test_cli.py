"""Tests for the command line: parsing, the environment check, running a batch, export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._fixtures import make_document, make_engines, make_models, make_plan_for
from twinscribe import __version__, audio, cli, pipeline
from twinscribe.models import MODELS_ENV
from twinscribe.outputs.transcript_doc import write_document


def test_parser_commands(tmp_path: Path) -> None:
    parser = cli.build_parser()
    run = parser.parse_args(["run", str(tmp_path), "--quality", "quick", "--threads", "2", "--out", "o", "--no-recurse", "--device", "cuda"])
    assert run.command == "run" and run.quality == "quick" and run.threads == 2 and run.out == Path("o")
    assert run.no_recurse is True and run.device == "cuda"
    assert parser.parse_args(["run", "x"]).device == "auto"
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "x", "--device", "npu"])
    check = parser.parse_args(["check", "--verify"])
    assert check.command == "check" and check.verify is True
    export = parser.parse_args(["export", "a.transcript.json", "--author", "Me"])
    assert export.command == "export" and export.author == "Me"
    app = parser.parse_args(["app", "x.mp3", "--dark", "--shot", "s.png"])
    assert app.command == "app" and app.dark and app.shot == Path("s.png")
    with pytest.raises(SystemExit):
        parser.parse_args(["run", ".", "--quality", "ultra"])


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_check_reports_models_and_levels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv(MODELS_ENV, str(tmp_path / "models"))
    assert cli.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "ffmpeg:" in out and "models root:" in out and "quality levels:  none" in out
    assert "Machine:" in out and "plan (auto):" in out
    make_models(tmp_path / "models")
    monkeypatch.setattr(cli, "probe_libraries", lambda: make_plan_for().backends and __import__("twinscribe.hardware", fromlist=["x"]).Libraries(True, True, True, True, {}))
    assert cli.main(["check", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "quality levels:  quick, standard, careful" in out and "plan (cpu):" in out
    assert cli.main(["check", "--verify"]) == 1          # present but unpinned files
    assert "unpinned" in capsys.readouterr().out


def test_run_without_recordings_or_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv(MODELS_ENV, str(tmp_path / "models"))
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["run", str(empty)]) == 2
    assert "no recordings found" in capsys.readouterr().err
    audio.synthetic_wav(empty / "a.wav", 1.0)
    assert cli.main(["run", str(empty)]) == 2
    assert "needs models that are not present" in capsys.readouterr().err


def test_run_records_engine_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv(MODELS_ENV, str(tmp_path / "models"))
    monkeypatch.setenv("TWINSCRIBE_HOME", str(tmp_path / "home"))
    make_models(tmp_path / "models")
    folder = tmp_path / "media"
    folder.mkdir()
    source = audio.synthetic_wav(folder / "a.wav", 1.0)
    # Engines that fail stand in for the real ones, so the recorded failure is what the test checks
    # whether or not the engine libraries are installed.
    monkeypatch.setattr(pipeline, "default_engines", lambda: make_engines(fail_publisher=True))
    monkeypatch.setattr(pipeline, "current_plan", lambda *args, **kwargs: make_plan_for())
    monkeypatch.setattr(cli, "current_plan", lambda *args, **kwargs: make_plan_for())
    assert cli.main(["run", str(folder), "--device", "cpu"]) == 1
    captured = capsys.readouterr()
    assert "FAILED" in captured.err and "publisher exploded" in captured.err and "batch record:" in captured.out
    record = json.loads((folder / "a.run.json").read_text(encoding="utf-8"))
    assert record["failures"] and record["failures"][0]["path"] == str(source)
    assert list((tmp_path / "home" / "runs").glob("batch_*.json"))


def test_live_line_prints_on_stage_changes_without_a_terminal() -> None:
    import io

    from twinscribe.cli import LiveLine
    from twinscribe.pipeline import Progress

    stream = io.StringIO()
    line = LiveLine(stream)
    line.update("[1/1] a.wav", Progress("publisher", 0.10, "Transcribing (published engine)", 3.0, None))
    line.update("[1/1] a.wav", Progress("publisher", 0.12, "Transcribing (published engine)", 4.0, 30.0))
    line.update("[1/1] a.wav", Progress("publisher", 0.25, "Transcribing (published engine)", 8.0, 100.0))
    line.finish()
    out = stream.getvalue().splitlines()
    assert len(out) == 2
    assert out[0].endswith("0:03 elapsed") and "roughly 2 min left" in out[1]


def test_live_line_updates_in_place_on_a_terminal() -> None:
    import io

    from twinscribe.cli import LiveLine
    from twinscribe.pipeline import Progress

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = Terminal()
    line = LiveLine(stream)
    line.update("[1/2] a.wav", Progress("digest", 0.0, "Reading the file", 0.0, None))
    line.update("[1/2] a.wav", Progress("detector", 0.5, "Checking (second engine)", 30.0, 30.0))
    line.finish()
    text = stream.getvalue()
    assert text.count("\r") == 2 and text.endswith("\n") and "about 30 s left" in text


def test_export_renders_beside_the_document(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    doc = make_document(source_name="meeting.mp3")
    path = tmp_path / "meeting.transcript.json"
    write_document(doc, path)
    assert cli.main(["export", str(path), "--author", "Ann"]) == 0
    assert "rendered meeting.txt, meeting.docx, meeting.srt" in capsys.readouterr().out
    assert (tmp_path / "meeting.txt").is_file() and (tmp_path / "meeting.docx").is_file() and (tmp_path / "meeting.srt").is_file()
    assert cli.main(["export", str(tmp_path / "absent.json")]) == 1
