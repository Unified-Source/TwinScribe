"""Tests for the audio tagger's region handling: the cut of a long region into pieces the
model accepts, the merge of the pieces' events, and the region loop against a fake tagger that
records what it was given. Expected values are hand-derived."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from twinscribe.audio import SAMPLE_RATE, synthetic_wav
from twinscribe.engines import tagging
from twinscribe.scenes import Event


def test_pieces_cut_a_span_into_equal_parts_no_longer_than_the_limit() -> None:
    limit = 30 * SAMPLE_RATE
    # A span that fits is one piece, whatever its offset.
    assert tagging.pieces(1000, 1000 + limit, limit) == [(1000, 1000 + limit)]
    # 45 s is two pieces of 22.5 s; 61 s is three pieces, the last one ending exactly at hi.
    assert tagging.pieces(0, 45 * SAMPLE_RATE, limit) == [(0, 360000), (360000, 720000)]
    three = tagging.pieces(160, 160 + 61 * SAMPLE_RATE, limit)
    assert len(three) == 3 and three[0][0] == 160 and three[-1][1] == 160 + 61 * SAMPLE_RATE
    assert all(end - start <= limit for start, end in three)
    assert all(three[index][1] == three[index + 1][0] for index in range(2))
    # An empty span is no piece; a bad limit is refused.
    assert tagging.pieces(5, 5, limit) == []
    with pytest.raises(ValueError):
        tagging.pieces(0, 10, 0)


def test_merge_events_keeps_each_class_at_its_highest_probability() -> None:
    merged = tagging.merge_events(
        [
            (Event("Speech", 0.2), Event("Music", 0.7)),
            (Event("Speech", 0.9), Event("Silence", 0.5)),
            (),
        ],
        top_k=8,
    )
    assert merged == (Event("Speech", 0.9), Event("Music", 0.7), Event("Silence", 0.5))
    # top_k keeps the strongest; ties fall to the name.
    assert tagging.merge_events([(Event("B", 0.5), Event("A", 0.5), Event("C", 0.6))], top_k=2) == (
        Event("C", 0.6),
        Event("A", 0.5),
    )
    assert tagging.merge_events([(), ()], top_k=3) == ()


def test_tag_regions_feeds_the_model_pieces_it_accepts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = synthetic_wav(tmp_path / "long.wav", seconds=70.0)
    model_dir = tmp_path / "tagger"
    model_dir.mkdir()
    (model_dir / tagging.MODEL_FILE).write_bytes(b"")
    (model_dir / tagging.LABELS_FILE).write_text("index,mid,display_name\n", encoding="ascii")

    lengths: list[int] = []

    class FakeStream:
        def __init__(self) -> None:
            self.samples = None

        def accept_waveform(self, rate: int, samples) -> None:
            assert rate == SAMPLE_RATE
            self.samples = samples

    class FakeTagger:
        def create_stream(self) -> FakeStream:
            return FakeStream()

        def compute(self, stream: FakeStream) -> list:
            lengths.append(len(stream.samples))
            # The model's own limits, as measured: under 0.16 s and over 30 s it fails.
            assert 0.16 * SAMPLE_RATE <= len(stream.samples) <= 30 * SAMPLE_RATE
            strength = 0.9 if len(stream.samples) > 20 * SAMPLE_RATE else 0.4
            return [SimpleNamespace(name="Speech", prob=strength), SimpleNamespace(name="Music", prob=0.1)]

    monkeypatch.setattr(tagging, "import_sherpa_onnx", lambda: SimpleNamespace(version="0"))
    monkeypatch.setattr(tagging, "_build_tagger", lambda *args, **kwargs: FakeTagger())
    monkeypatch.setattr(tagging, "library_versions", lambda module: {"sherpa_onnx": "0"})

    result = tagging.tag_regions(wav, model_dir, [(0.0, 70.0), (1.0, 1.1), (2.0, 12.0)], threads=1)

    # 70 s becomes three pieces of 23.3 s; the 0.1 s region is not tagged; 10 s is one piece.
    assert [round(n / SAMPLE_RATE, 1) for n in lengths] == [23.3, 23.3, 23.3, 10.0]
    assert result.events[1] == ()
    # The long region's events are merged: Speech at the strongest piece, Music kept.
    assert result.events[0] == (Event("Speech", 0.9), Event("Music", 0.1))
    assert result.events[2] == (Event("Speech", 0.4), Event("Music", 0.1))
    assert result.regions == ((0.0, 70.0), (1.0, 1.1), (2.0, 12.0))
