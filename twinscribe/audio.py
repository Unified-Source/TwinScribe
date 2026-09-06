"""Audio helpers: 16 kHz mono WAV reading, decoding through ffmpeg, durations, digests and a
synthetic tone writer.

Every engine consumes 16 kHz mono 16-bit PCM and nothing else is done to the signal: no
noise reduction, no level normalisation, no silence stripping. Decoding is delegated to an
external ffmpeg executable invoked as an argument list, never through a shell.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2

_PCM_FULL_SCALE = 32768.0
_DIGEST_CHUNK_BYTES = 1 << 20

PathLike = str | os.PathLike[str]


class AudioFormatError(ValueError):
    """Raised when a WAV file is not 16 kHz, mono, 16-bit PCM."""


def _open_checked(path: PathLike) -> wave.Wave_read:
    """Open a WAV file and verify its format, raising AudioFormatError on any mismatch."""
    try:
        handle = wave.open(str(path), "rb")
    except wave.Error as exc:
        raise AudioFormatError(f"{path}: not a plain PCM WAV file ({exc})") from exc
    problems = []
    if handle.getnchannels() != CHANNELS:
        problems.append(f"{handle.getnchannels()} channels (expected {CHANNELS})")
    if handle.getframerate() != SAMPLE_RATE:
        problems.append(f"{handle.getframerate()} Hz (expected {SAMPLE_RATE})")
    if handle.getsampwidth() != SAMPLE_WIDTH_BYTES:
        problems.append(
            f"{8 * handle.getsampwidth()}-bit samples (expected {8 * SAMPLE_WIDTH_BYTES}-bit)"
        )
    if problems:
        handle.close()
        raise AudioFormatError(f"{path}: " + "; ".join(problems))
    return handle


def read_wav_mono16k(path: PathLike) -> np.ndarray:
    """Read a 16 kHz mono 16-bit PCM WAV file into a float32 array scaled to [-1, 1).

    Anything else is rejected with AudioFormatError. The standard-library reader is used so
    that the function has no dependency beyond numpy.
    """
    with _open_checked(path) as handle:
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    samples /= _PCM_FULL_SCALE
    return samples


def _wav_duration(path: PathLike) -> float:
    """Duration in seconds of a checked WAV file, from its frame count."""
    with _open_checked(path) as handle:
        return handle.getnframes() / float(handle.getframerate())


def duration_s(path: PathLike) -> float:
    """Duration in seconds of a 16 kHz mono 16-bit WAV file."""
    return _wav_duration(path)


def find_ffmpeg(ffmpeg: PathLike | None = None) -> str:
    """Resolve the ffmpeg executable: the argument first, then the search path.

    Raises FileNotFoundError with a clear message when neither yields an executable.
    """
    if ffmpeg is not None:
        candidate = str(ffmpeg)
        if Path(candidate).is_file():
            return candidate
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
        raise FileNotFoundError(f"ffmpeg not found at {candidate!r}")
    resolved = shutil.which("ffmpeg")
    if resolved is None:
        raise FileNotFoundError("ffmpeg not found on the search path; pass its location explicitly")
    return resolved


def decode_to_wav(
    src: PathLike,
    dst: PathLike,
    ffmpeg: PathLike | None = None,
    duration_s: float | None = None,
) -> float:
    """Decode any container ffmpeg understands to 16 kHz mono 16-bit PCM WAV at dst.

    The command is an argument list (no shell). A duration limit, when given, is passed as
    an output option so that padded video tracks cannot lengthen the decoded audio. Returns
    the decoded duration in seconds read back from the written file.
    """
    source = Path(src)
    if not source.is_file():
        raise FileNotFoundError(f"audio source not found: {source}")
    if duration_s is not None and duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    executable = find_ffmpeg(ffmpeg)
    args = [
        executable,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        str(CHANNELS),
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
    ]
    if duration_s is not None:
        args += ["-t", f"{duration_s:.6f}"]
    args.append(str(dst))
    completed = subprocess.run(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg exited with status {completed.returncode} decoding {source}: {detail[-2000:]}"
        )
    return _wav_duration(dst)


def sha256_of(path: PathLike) -> str:
    """Hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_DIGEST_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_wav(path: PathLike, seconds: float, tone_hz: float = 440.0) -> Path:
    """Write a sine tone at half full scale as a 16 kHz mono 16-bit WAV and return its path.

    A test helper kept in the package so that the application's own smoke check can use it
    without depending on any recording.
    """
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")
    frames = int(round(seconds * SAMPLE_RATE))
    t = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    signal = 0.5 * np.sin(2.0 * np.pi * tone_hz * t)
    pcm = np.round(signal * (_PCM_FULL_SCALE - 1.0)).astype("<i2")
    target = Path(path)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH_BYTES)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    return target
