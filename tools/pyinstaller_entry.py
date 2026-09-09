"""Entry point of the frozen executables: the command line, which opens the window when it is
given no command, so one set of files serves both the console and the windowed executable."""

from __future__ import annotations

import multiprocessing
import sys

from twinscribe.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
