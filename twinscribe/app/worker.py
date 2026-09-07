"""Background thread that runs the batch pipeline and reports to the window through signals,
so the window stays responsive and playback continues while recordings are transcribed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from twinscribe.hardware import Plan
from twinscribe.models import ModelSet
from twinscribe.pipeline import BatchResult, Engines, Outcome, Progress, run_batch
from twinscribe.profiles import Profile


class PipelineWorker(QThread):
    """Runs one batch; `rows` maps each source back to its library row.

    Signals carry the library row: `progress(row, report)` with the Progress record,
    `partial(row, segment)` with each segment the published engine produces, `file_done(row,
    result)` with the FileResult, `file_failed(row, error_class, message)`, and
    `finished_all(result)` with the BatchResult once every recording has been processed or the
    batch was cancelled.
    """

    progress = Signal(int, object)
    partial = Signal(int, object)
    file_done = Signal(int, object)
    file_failed = Signal(int, str, str)
    finished_all = Signal(object)

    def __init__(
        self,
        rows: Sequence[int],
        sources: Sequence[Path],
        profile: Profile,
        models: ModelSet,
        out_dir: Path | None = None,
        threads: int | None = None,
        author: str = "",
        engines: Engines | None = None,
        record_dir: Path | None = None,
        parent: QObject | None = None,
        plan: Plan | None = None,
    ) -> None:
        super().__init__(parent)
        if len(rows) != len(sources):
            raise ValueError("rows and sources must have the same length")
        self._plan = plan
        self._rows = list(rows)
        self._sources = [Path(s) for s in sources]
        self._profile = profile
        self._models = models
        self._out_dir = out_dir
        self._threads = threads
        self._author = author
        self._engines = engines
        self._record_dir = record_dir
        self._cancel = False

    def cancel(self) -> None:
        """Ask the batch to stop at the next progress check."""
        self._cancel = True

    def cancelled(self) -> bool:
        return self._cancel

    def run(self) -> None:  # noqa: D401 (Qt virtual)
        def on_progress(index: int, _total: int, report: Progress) -> None:
            self.progress.emit(self._rows[index], report)

        def on_partial(index: int, role: str, segment) -> None:
            if role == "publisher":
                self.partial.emit(self._rows[index], segment)

        def on_outcome(index: int, outcome: Outcome) -> None:
            row = self._rows[index]
            if outcome.ok and outcome.result is not None:
                self.file_done.emit(row, outcome.result)
            elif not outcome.cancelled:
                self.file_failed.emit(row, outcome.error_class or "Error", outcome.error or "failed")

        result: BatchResult
        try:
            result = run_batch(
                self._sources,
                self._profile,
                self._models,
                out_dir=self._out_dir,
                threads=self._threads,
                author=self._author,
                progress=on_progress,
                cancel=self.cancelled,
                engines=self._engines,
                record_dir=self._record_dir,
                on_outcome=on_outcome,
                plan=self._plan,
                on_partial=on_partial,
            )
        except Exception as exc:  # noqa: BLE001 - a thread must not die silently
            result = BatchResult()
            for row in self._rows:
                self.file_failed.emit(row, type(exc).__name__, str(exc))
        self.finished_all.emit(result)
