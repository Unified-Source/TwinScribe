"""Tests for twinscribe.audio on a synthetic sine WAV; the ffmpeg tests skip when it is absent."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import wave

import numpy as np
import pytest

from twinscribe import audio

TONE_HZ = 440.0


def _write_wav(path, rate: int, channels: int, width: int, frames: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames * channels * width))


def test_synthetic_wav_round_trip(tmp_path):
    path = audio.synthetic_wav(tmp_path / "tone.wav", seconds=1.0, tone_hz=TONE_HZ)
    samples = audio.read_wav_mono16k(path)
    assert samples.dtype == np.float32
    assert samples.shape == (audio.SAMPLE_RATE,)
    assert abs(float(np.max(np.abs(samples))) - 0.5) < 1e-3
    # One second at 16 kHz puts the tone in FFT bin number 440 exactly.
    spectrum = np.abs(np.fft.rfft(samples))
    assert int(np.argmax(spectrum)) == int(TONE_HZ)


def test_synthetic_wav_is_deterministic(tmp_path):
    first = audio.synthetic_wav(tmp_path / "a.wav", seconds=0.25)
    second = audio.synthetic_wav(tmp_path / "b.wav", seconds=0.25)
    assert audio.sha256_of(first) == audio.sha256_of(second)


def test_synthetic_wav_rejects_negative_seconds(tmp_path):
    with pytest.raises(ValueError):
        audio.synthetic_wav(tmp_path / "bad.wav", seconds=-1.0)


def test_duration_matches_seconds(tmp_path):
    path = audio.synthetic_wav(tmp_path / "tone.wav", seconds=2.5)
    assert audio.duration_s(path) == pytest.approx(2.5)


def test_empty_wav_reads_as_empty_array(tmp_path):
    path = audio.synthetic_wav(tmp_path / "empty.wav", seconds=0.0)
    assert audio.read_wav_mono16k(path).shape == (0,)
    assert audio.duration_s(path) == 0.0


@pytest.mark.parametrize(
    ("rate", "channels", "width", "fragment"),
    [
        (8000, 1, 2, "8000 Hz"),
        (16000, 2, 2, "2 channels"),
        (16000, 1, 1, "8-bit"),
        (44100, 2, 1, "44100 Hz"),
    ],
)
def test_read_rejects_wrong_format(tmp_path, rate, channels, width, fragment):
    path = tmp_path / "wrong.wav"
    _write_wav(path, rate, channels, width, frames=100)
    with pytest.raises(audio.AudioFormatError) as excinfo:
        audio.read_wav_mono16k(path)
    assert fragment in str(excinfo.value)
    with pytest.raises(audio.AudioFormatError):
        audio.duration_s(path)


def test_read_rejects_non_wav(tmp_path):
    path = tmp_path / "text.wav"
    path.write_bytes(b"this is not a wav file")
    with pytest.raises(audio.AudioFormatError):
        audio.read_wav_mono16k(path)


def test_sha256_known_value(tmp_path):
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")
    assert audio.sha256_of(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_matches_hashlib_on_multi_chunk_file(tmp_path):
    path = tmp_path / "big.bin"
    payload = bytes(range(256)) * 8192
    path.write_bytes(payload)
    assert audio.sha256_of(path) == hashlib.sha256(payload).hexdigest()


def test_find_ffmpeg_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        audio.find_ffmpeg(tmp_path / "no-such-ffmpeg.exe")


def test_decode_rejects_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        audio.decode_to_wav(tmp_path / "missing.wav", tmp_path / "out.wav")


def test_decode_rejects_non_positive_duration(tmp_path):
    src = audio.synthetic_wav(tmp_path / "tone.wav", seconds=0.5)
    with pytest.raises(ValueError):
        audio.decode_to_wav(src, tmp_path / "out.wav", duration_s=0.0)


ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not on the search path"
)


@ffmpeg_required
def test_decode_wav_to_wav_preserves_samples(tmp_path):
    src = audio.synthetic_wav(tmp_path / "tone.wav", seconds=2.0)
    dst = tmp_path / "decoded.wav"
    decoded = audio.decode_to_wav(src, dst)
    assert decoded == pytest.approx(2.0, abs=1e-3)
    original = audio.read_wav_mono16k(src)
    copy = audio.read_wav_mono16k(dst)
    assert copy.shape == original.shape
    assert float(np.max(np.abs(copy - original))) <= 2.0 / 32768.0


@ffmpeg_required
def test_decode_applies_duration_cut(tmp_path):
    src = audio.synthetic_wav(tmp_path / "tone.wav", seconds=3.0)
    dst = tmp_path / "cut.wav"
    decoded = audio.decode_to_wav(src, dst, duration_s=1.0)
    assert decoded == pytest.approx(1.0, abs=2e-3)
    assert audio.duration_s(dst) == decoded


@ffmpeg_required
def test_decode_container_to_wav(tmp_path):
    src = audio.synthetic_wav(tmp_path / "tone.wav", seconds=1.5)
    flac = tmp_path / "tone.flac"
    encode = subprocess.run(
        [audio.find_ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
         "-i", str(src), "-c:a", "flac", str(flac)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if encode.returncode != 0:
        pytest.skip("this ffmpeg build cannot encode FLAC for the container test")
    dst = tmp_path / "from_flac.wav"
    decoded = audio.decode_to_wav(src=flac, dst=dst)
    assert decoded == pytest.approx(1.5, abs=1e-3)
    original = audio.read_wav_mono16k(src)
    copy = audio.read_wav_mono16k(dst)
    assert copy.shape == original.shape
    assert float(np.max(np.abs(copy - original))) <= 2.0 / 32768.0


@ffmpeg_required
def test_decode_reports_ffmpeg_failure(tmp_path):
    bogus = tmp_path / "garbage.mp3"
    bogus.write_bytes(b"\x00" * 64)
    with pytest.raises(RuntimeError) as excinfo:
        audio.decode_to_wav(bogus, tmp_path / "out.wav")
    assert "ffmpeg exited" in str(excinfo.value)
