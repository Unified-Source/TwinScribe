"""Load probe: was the machine busy with something else while a timing was taken?

Two snapshots of the operating system's CPU counters bracket a timed cell. The difference
gives the average number of cores that other processes kept busy in between, which decides
whether the timing is comparable with timings taken on an idle machine. Every probe returns
None when the platform or the counters are unavailable; probes never raise.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

# Windows FILETIME counters are in 100-nanosecond units.
_FILETIME_UNITS_PER_SECOND = 1e7


@dataclass(frozen=True)
class Snapshot:
    """CPU counters at one instant.

    wall_s: monotonic wall-clock seconds.
    total_cpu_s: CPU seconds consumed by all processes on all cores since boot.
    self_cpu_s: CPU seconds consumed by this process since it started.
    """

    wall_s: float
    total_cpu_s: float
    self_cpu_s: float


@dataclass(frozen=True)
class LoadVerdict:
    """Contention verdict for one timed cell, recorded beside the timing it qualifies."""

    other_load_cores: float | None
    busy: bool | None
    on_mains_power: bool | None
    threshold_cores: float


def _filetime_seconds(filetime: object) -> float:
    high = int(getattr(filetime, "dwHighDateTime"))
    low = int(getattr(filetime, "dwLowDateTime"))
    return ((high << 32) | low) / _FILETIME_UNITS_PER_SECOND


def _snapshot_windows() -> Snapshot | None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetSystemTimes.argtypes = [ctypes.POINTER(wintypes.FILETIME)] * 3
    kernel32.GetSystemTimes.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel32.GetProcessTimes.restype = wintypes.BOOL

    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    created, exited, p_kernel, p_user = (wintypes.FILETIME() for _ in range(4))

    wall = time.monotonic()
    if not kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    handle = kernel32.GetCurrentProcess()
    if not kernel32.GetProcessTimes(
        handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(p_kernel), ctypes.byref(p_user)
    ):
        return None

    # The system kernel counter includes idle time; subtracting idle leaves busy kernel time.
    total = (_filetime_seconds(kernel) - _filetime_seconds(idle)) + _filetime_seconds(user)
    own = _filetime_seconds(p_kernel) + _filetime_seconds(p_user)
    return Snapshot(wall_s=wall, total_cpu_s=total, self_cpu_s=own)


def _snapshot_linux() -> Snapshot | None:
    ticks_per_second = float(os.sysconf("SC_CLK_TCK"))
    wall = time.monotonic()
    with open("/proc/stat", encoding="ascii") as handle:
        first = handle.readline().split()
    if not first or first[0] != "cpu":
        return None
    values = [int(part) for part in first[1:]]
    # Fields: user nice system idle iowait irq softirq steal (guest times are already inside
    # user and nice). Idle and iowait are the two idle states.
    busy_indices = (0, 1, 2, 5, 6, 7)
    busy_ticks = sum(values[index] for index in busy_indices if index < len(values))

    with open("/proc/self/stat", encoding="ascii") as handle:
        text = handle.read()
    # The command name sits in parentheses and may contain spaces; fields are counted from
    # the character after the closing parenthesis. utime and stime are fields 14 and 15 of
    # the whole line, hence indices 11 and 12 after the state field.
    after_name = text[text.rindex(")") + 2 :].split()
    own_ticks = int(after_name[11]) + int(after_name[12])

    return Snapshot(
        wall_s=wall,
        total_cpu_s=busy_ticks / ticks_per_second,
        self_cpu_s=own_ticks / ticks_per_second,
    )


def _snapshot_darwin() -> Snapshot | None:
    """macOS: the host's CPU tick counters through the Mach host statistics call."""
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libSystem.dylib", use_errno=True)
    host_cpu_load_info = 3
    cpu_state_max = 4
    counts = (ctypes.c_uint32 * cpu_state_max)()
    count = ctypes.c_uint32(cpu_state_max)
    libc.mach_host_self.restype = ctypes.c_uint32
    libc.host_statistics.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
    libc.host_statistics.restype = ctypes.c_int
    wall = time.monotonic()
    if libc.host_statistics(libc.mach_host_self(), host_cpu_load_info, counts, ctypes.byref(count)) != 0:
        return None
    ticks_per_second = float(os.sysconf("SC_CLK_TCK")) if hasattr(os, "sysconf") else 100.0
    # States: user, system, idle, nice; idle is the one excluded.
    busy_ticks = int(counts[0]) + int(counts[1]) + int(counts[3])
    times = os.times()
    return Snapshot(wall_s=wall, total_cpu_s=busy_ticks / ticks_per_second, self_cpu_s=times.user + times.system)


def snapshot() -> Snapshot | None:
    """Read the CPU counters now; None where the platform offers no supported counters."""
    try:
        if sys.platform == "win32":
            return _snapshot_windows()
        if sys.platform.startswith("linux"):
            return _snapshot_linux()
        if sys.platform == "darwin":
            return _snapshot_darwin()
        return None
    except Exception:
        return None


def other_load(before: Snapshot | None, after: Snapshot | None) -> float | None:
    """Average number of cores other processes kept busy between two snapshots.

    Computed as (total CPU delta - own CPU delta) / wall delta, clamped at zero. None when
    either snapshot is missing or the wall delta is not positive.
    """
    if before is None or after is None:
        return None
    wall = after.wall_s - before.wall_s
    if wall <= 0.0:
        return None
    other_cpu = (after.total_cpu_s - before.total_cpu_s) - (after.self_cpu_s - before.self_cpu_s)
    return max(0.0, other_cpu / wall)


def busy(before: Snapshot | None, after: Snapshot | None, threshold_cores: float = 0.5) -> bool | None:
    """True when other processes exceeded threshold_cores on average; None when unknown."""
    load = other_load(before, after)
    if load is None:
        return None
    return load > threshold_cores


def _on_mains_windows() -> bool | None:
    import ctypes

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetSystemPowerStatus.argtypes = [ctypes.POINTER(SystemPowerStatus)]
    kernel32.GetSystemPowerStatus.restype = ctypes.c_int
    status = SystemPowerStatus()
    if not kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    # ACLineStatus: 0 offline, 1 online, 255 unknown.
    if status.ACLineStatus == 1:
        return True
    if status.ACLineStatus == 0:
        return False
    return None


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="ascii", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _on_mains_linux() -> bool | None:
    root = "/sys/class/power_supply"
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return None
    mains_states: list[bool] = []
    battery_states: list[str] = []
    for entry in entries:
        base = os.path.join(root, entry)
        kind = _read_text(os.path.join(base, "type"))
        if kind == "Mains":
            online = _read_text(os.path.join(base, "online"))
            if online in ("0", "1"):
                mains_states.append(online == "1")
        elif kind == "Battery":
            status = _read_text(os.path.join(base, "status"))
            if status:
                battery_states.append(status)
    if mains_states:
        return any(mains_states)
    # Without a mains supply entry, a battery that is not discharging implies external power.
    if battery_states:
        if all(state == "Discharging" for state in battery_states):
            return False
        if any(state in ("Charging", "Full", "Not charging") for state in battery_states):
            return True
    return None


def _on_mains_darwin() -> bool | None:
    """macOS: the power management tool names the present supply."""
    import subprocess

    completed = subprocess.run(
        ["pmset", "-g", "batt"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=5, check=False,
    )
    if completed.returncode != 0:
        return None
    text = completed.stdout.decode("utf-8", errors="replace")
    if "AC Power" in text:
        return True
    if "Battery Power" in text:
        return False
    return None


def on_mains_power() -> bool | None:
    """True on external power, False on battery, None where it cannot be determined."""
    try:
        if sys.platform == "win32":
            return _on_mains_windows()
        if sys.platform.startswith("linux"):
            return _on_mains_linux()
        if sys.platform == "darwin":
            return _on_mains_darwin()
        return None
    except Exception:
        return None


def verdict(before: Snapshot | None, after: Snapshot | None, threshold_cores: float = 0.5) -> LoadVerdict:
    """Combine the other-load figure, the busy flag and the mains flag for one timed cell."""
    return LoadVerdict(
        other_load_cores=other_load(before, after),
        busy=busy(before, after, threshold_cores),
        on_mains_power=on_mains_power(),
        threshold_cores=threshold_cores,
    )
