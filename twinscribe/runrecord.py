"""Run record: which engines ran with which settings on which input, how long each stage
took, whether the machine was busy, and every file that failed and why.

Also holds the machine facts that qualify a timing and the atomic JSON writer that the
review-set document shares.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import socket
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from twinscribe.load import LoadVerdict

RUNRECORD_SCHEMA = "twinscribe.runrecord.v1"

_DIGEST_CHUNK_BYTES = 1 << 20
_BYTES_PER_GB = 1024**3


def utc_now() -> str:
    """Current time in UTC as an ISO 8601 string with second resolution."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def input_digest(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a file's bytes as lower-case hex.

    Intentionally standalone: the record module keeps no dependency on the audio module, so a
    record can be written for an input the audio module refused to open. The value is
    identical to the audio module's digest of the same file, and a caller that already holds
    that digest passes it in rather than hashing twice.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_DIGEST_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _processor_name() -> str | None:
    name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or ""
    if not name and sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="ascii", errors="replace") as handle:
                for line in handle:
                    key, _, value = line.partition(":")
                    if key.strip() in ("model name", "Model", "Hardware"):
                        name = value.strip()
                        break
        except OSError:
            name = ""
    return name or None


def _memory_gb() -> float | None:
    total: int | None = None
    if sys.platform == "win32":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total = int(status.ullTotalPhys)
    elif hasattr(os, "sysconf"):
        try:
            total = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError, AttributeError):
            total = None
    if total is None or total <= 0:
        return None
    return round(total / _BYTES_PER_GB, 1)


def machine_facts(include_hostname: bool = False) -> dict[str, Any]:
    """Facts that qualify a timing: processor, cores, memory, operating system, Python.

    The hostname is included only when explicitly requested, so records written by default
    identify the class of machine and not the machine.
    """
    facts: dict[str, Any] = {}
    probes = {
        "processor": _processor_name,
        "logical_cores": os.cpu_count,
        "memory_gb": _memory_gb,
        "os": lambda: f"{platform.system()} {platform.release()} ({platform.version()})".strip(),
        "architecture": platform.machine,
        "python": platform.python_version,
    }
    for key, probe in probes.items():
        try:
            facts[key] = probe()
        except Exception:
            facts[key] = None
    if include_hostname:
        try:
            facts["hostname"] = socket.gethostname()
        except Exception:
            facts["hostname"] = None
    return facts


@dataclass(frozen=True)
class Failure:
    """One input that did not complete: its path, the exception class name and message."""

    path: str
    error_class: str
    message: str

    @classmethod
    def from_exception(cls, path: str, exc: BaseException) -> "Failure":
        return cls(path=str(path), error_class=type(exc).__name__, message=str(exc))


def _as_failure(item: Any) -> Failure:
    if isinstance(item, Failure):
        return item
    if isinstance(item, dict):
        return Failure(str(item["path"]), str(item["error_class"]), str(item["message"]))
    path, error_class, message = item
    return Failure(str(path), str(error_class), str(message))


@dataclass(frozen=True)
class RunRecord:
    """Everything needed to reproduce or discount one run over one input.

    engines: role -> {"engine", "model", "preset"}; the roles are "publisher" and "detector".
    versions: library and model versions as reported by the engines.
    settings: the effective preset parameters and review thresholds.
    Timings are in seconds; real_time_factor is transcribe_s over audio_s (load excluded).
    """

    engines: dict[str, dict[str, str]]
    versions: dict[str, str]
    settings: dict[str, Any]
    input_path: str
    input_sha256: str
    audio_s: float
    load_s: float
    transcribe_s: float
    started_utc: str
    ended_utc: str
    load_verdict: LoadVerdict | None = None
    failures: tuple[Failure, ...] = ()
    machine: dict[str, Any] = field(default_factory=machine_facts)

    def __post_init__(self) -> None:
        # Accept plain (path, error_class, message) tuples and normalise them to Failure.
        object.__setattr__(self, "failures", tuple(_as_failure(item) for item in self.failures))

    @property
    def real_time_factor(self) -> float | None:
        if self.audio_s <= 0.0:
            return None
        return self.transcribe_s / self.audio_s

    def to_dict(self) -> dict[str, Any]:
        verdict = None if self.load_verdict is None else dataclasses.asdict(self.load_verdict)
        return {
            "schema": RUNRECORD_SCHEMA,
            "engines": {role: dict(facts) for role, facts in self.engines.items()},
            "versions": dict(self.versions),
            "settings": dict(self.settings),
            "input_path": self.input_path,
            "input_sha256": self.input_sha256,
            "audio_s": float(self.audio_s),
            "load_s": float(self.load_s),
            "transcribe_s": float(self.transcribe_s),
            "real_time_factor": self.real_time_factor,
            "started_utc": self.started_utc,
            "ended_utc": self.ended_utc,
            "load_verdict": verdict,
            "failures": [dataclasses.asdict(failure) for failure in self.failures],
            "machine": dict(self.machine),
        }

    def write(self, path: str | os.PathLike[str]) -> None:
        write_json_atomic(self.to_dict(), path)


def _json_default(value: Any) -> Any:
    """Coerce values json cannot serialise: numpy scalars, dataclasses, paths, tuples, sets."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"value of type {type(value).__name__} is not JSON serialisable")


def write_json_atomic(doc: Any, path: str | os.PathLike[str]) -> None:
    """Write doc as UTF-8 JSON through a temporary file in the same folder, then rename.

    A reader never sees a partial file: either the previous content or the new content.
    """
    target = os.fspath(path)
    folder = os.path.dirname(os.path.abspath(target))
    os.makedirs(folder, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(prefix=".", suffix=".json.tmp", dir=folder)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(doc, handle, indent=2, ensure_ascii=False, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    except BaseException:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
