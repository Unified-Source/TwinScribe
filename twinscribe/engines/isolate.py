"""Running an engine call in a child process.

The speaker library holds the interpreter lock for the whole of a labelling call, which can
run for minutes on a processor; a thread in the same process then gets no time, and a window
whose thread that is stops answering. A call made here runs in a freshly started process and
the caller's thread waits on a pipe, so the rest of the program keeps running. The function
and its arguments travel by pickling, so the function must be importable by name and the
arguments and the result must pickle; an exception in the child is raised again here.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def run_in_child_process(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Call `func(*args, **kwargs)` in a child process started afresh and return its result."""
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=1) as pool:
        return pool.apply(func, args, kwargs)
