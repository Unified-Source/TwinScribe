"""Tests for the load probe: snapshot arithmetic on hand-built values and probe shapes."""

from __future__ import annotations

import time

import pytest

import twinscribe.load as load
from twinscribe.load import LoadVerdict, Snapshot, busy, on_mains_power, other_load, snapshot, verdict

BEFORE = Snapshot(wall_s=100.0, total_cpu_s=1000.0, self_cpu_s=10.0)
# Ten seconds of wall time, 25 s of CPU consumed system-wide of which 5 s by this process:
# other processes kept 20 / 10 = 2.0 cores busy on average.
AFTER = Snapshot(wall_s=110.0, total_cpu_s=1025.0, self_cpu_s=15.0)


# ----------------------------------------------------------------- arithmetic


def test_other_load_on_hand_built_snapshots():
    assert other_load(BEFORE, AFTER) == pytest.approx(2.0)


def test_busy_threshold_is_strictly_greater():
    assert busy(BEFORE, AFTER) is True
    assert busy(BEFORE, AFTER, threshold_cores=1.99) is True
    assert busy(BEFORE, AFTER, threshold_cores=2.0) is False
    assert busy(BEFORE, AFTER, threshold_cores=2.5) is False


def test_idle_machine_is_not_busy():
    before = Snapshot(wall_s=0.0, total_cpu_s=0.0, self_cpu_s=0.0)
    after = Snapshot(wall_s=10.0, total_cpu_s=3.0, self_cpu_s=3.0)
    assert other_load(before, after) == 0.0
    assert busy(before, after) is False


def test_missing_snapshots_give_none():
    assert other_load(None, AFTER) is None
    assert other_load(BEFORE, None) is None
    assert other_load(None, None) is None
    assert busy(None, AFTER) is None
    assert busy(BEFORE, None) is None


def test_zero_or_negative_wall_delta_gives_none():
    assert other_load(BEFORE, BEFORE) is None
    assert other_load(AFTER, BEFORE) is None
    assert busy(AFTER, BEFORE) is None


def test_negative_other_load_is_clamped_to_zero():
    # Counter granularity can make the process delta exceed the system delta slightly.
    before = Snapshot(wall_s=0.0, total_cpu_s=0.0, self_cpu_s=0.0)
    after = Snapshot(wall_s=1.0, total_cpu_s=2.0, self_cpu_s=3.0)
    assert other_load(before, after) == 0.0
    assert busy(before, after) is False


@pytest.mark.parametrize(
    ("total_delta", "self_delta", "wall"),
    [(25.0, 5.0, 10.0), (0.5, 0.5, 2.0), (16.0, 0.0, 4.0), (1.0, 0.25, 0.5)],
)
def test_other_load_bounds(total_delta: float, self_delta: float, wall: float):
    before = Snapshot(wall_s=5.0, total_cpu_s=50.0, self_cpu_s=1.0)
    after = Snapshot(wall_s=5.0 + wall, total_cpu_s=50.0 + total_delta, self_cpu_s=1.0 + self_delta)
    figure = other_load(before, after)
    assert figure is not None
    assert 0.0 <= figure <= total_delta / wall
    assert figure == pytest.approx((total_delta - self_delta) / wall)


def test_verdict_on_hand_built_snapshots():
    result = verdict(BEFORE, AFTER, threshold_cores=0.5)
    assert isinstance(result, LoadVerdict)
    assert result.other_load_cores == pytest.approx(2.0)
    assert result.busy is True
    assert result.threshold_cores == 0.5
    assert result.on_mains_power in (True, False, None)
    unknown = verdict(None, AFTER)
    assert unknown.other_load_cores is None
    assert unknown.busy is None


# --------------------------------------------------------------------- probes


def test_snapshot_probe_shape():
    first = snapshot()
    assert first is None or isinstance(first, Snapshot)
    if first is not None:
        assert first.wall_s >= 0.0
        assert first.total_cpu_s >= 0.0
        assert first.self_cpu_s >= 0.0


def test_two_snapshots_are_monotonic_and_measurable():
    first = snapshot()
    if first is None:
        pytest.skip("no supported CPU counters on this platform")
    deadline = time.monotonic() + 0.05
    while time.monotonic() < deadline:
        pass
    second = snapshot()
    assert second is not None
    assert second.wall_s > first.wall_s
    assert second.total_cpu_s >= first.total_cpu_s
    assert second.self_cpu_s >= first.self_cpu_s
    figure = other_load(first, second)
    assert figure is not None and figure >= 0.0
    assert busy(first, second) in (True, False)


def test_on_mains_power_shape():
    assert on_mains_power() in (True, False, None)


def test_probes_return_none_on_an_unsupported_platform(monkeypatch):
    monkeypatch.setattr(load.sys, "platform", "unsupported-os")
    assert snapshot() is None
    assert on_mains_power() is None


def test_probes_never_raise(monkeypatch):
    def failing(*args, **kwargs):
        raise OSError("counter unavailable")

    monkeypatch.setattr(load, "_snapshot_windows", failing)
    monkeypatch.setattr(load, "_snapshot_linux", failing)
    monkeypatch.setattr(load, "_on_mains_windows", failing)
    monkeypatch.setattr(load, "_on_mains_linux", failing)
    assert snapshot() is None
    assert on_mains_power() is None
    result = verdict(snapshot(), snapshot())
    assert result == LoadVerdict(other_load_cores=None, busy=None, on_mains_power=None, threshold_cores=0.5)
