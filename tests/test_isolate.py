"""Tests for running an engine call in a child process: the call runs in another process,
its arguments and result travel intact, an exception comes back as itself, and the pipeline's
default diarizer is the child-process one."""

from __future__ import annotations

import os

import pytest

from tests._fixtures import child_echo, child_fail
from twinscribe.engines.isolate import run_in_child_process


def test_call_runs_in_another_process_with_its_arguments() -> None:
    pid, values = run_in_child_process(child_echo, {"a": 1}, times=3)
    assert pid != os.getpid() and pid > 0
    assert values == [{"a": 1}] * 3


def test_exception_in_the_child_is_raised_here() -> None:
    with pytest.raises(ValueError, match="no models here"):
        run_in_child_process(child_fail, "no models here")


def test_pipeline_labels_speakers_in_a_child() -> None:
    from twinscribe.engines import diarize
    from twinscribe.pipeline import default_engines

    assert default_engines().diarizer is diarize.diarize_in_child
