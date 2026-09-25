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
import struct
import subprocess
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


# Child processes started from a windowed application must not open a console window.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
FFMPEG_ENV = "TWINSCRIBE_FFMPEG"


def _executable_name() -> str:
    return "ffmpeg.exe" if os.name == "nt" else "ffmpeg"


def candidate_ffmpeg_paths() -> list[str]:
    """Where ffmpeg is looked for, in order, after an explicit argument.

    The environment variable named by FFMPEG_ENV; the search path; a bin folder beside the
    package (a portable copy); the executable an imageio-ffmpeg package carries when one is
    installed; then the usual places on each platform.
    """
    candidates: list[str] = []
    override = os.environ.get(FFMPEG_ENV)
    if override:
        candidates.append(override)
    on_path = shutil.which("ffmpeg")
    if on_path is not None:
        candidates.append(on_path)
    candidates.append(str(Path(__file__).resolve().parent.parent / "bin" / _executable_name()))
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        candidates.append(str(imageio_ffmpeg.get_ffmpeg_exe()))
    except Exception:  # noqa: BLE001 - an optional package that may be absent or broken
        pass
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidates.append(str(Path(local) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"))
    else:
        candidates.extend(["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"])
    return candidates


def find_ffmpeg(ffmpeg: PathLike | None = None) -> str:
    """Resolve the ffmpeg executable: the argument first, then the candidate places in order.

    Raises FileNotFoundError with a clear message when nothing yields an executable.
    """
    if ffmpeg is not None:
        candidate = str(ffmpeg)
        if Path(candidate).is_file():
            return candidate
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
        raise FileNotFoundError(f"ffmpeg not found at {candidate!r}")
    for candidate in candidate_ffmpeg_paths():
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        "ffmpeg not found: not on the search path, not in a bin folder beside the package, and "
        f"{FFMPEG_ENV} is not set; pass its location explicitly"
    )


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
        creationflags=NO_WINDOW,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg exited with status {completed.returncode} decoding {source}: {detail[-2000:]}"
        )
    return _wav_duration(dst)


def sha256_of(path: PathLike, progress: Callable[[float], None] | None = None) -> str:
    """Hex SHA-256 digest of a file, read in chunks.

    progress, when given, is called with the fraction of the file read, at most about a
    hundred times, so that hashing a very large recording shows movement.
    """
    digest = hashlib.sha256()
    total = os.path.getsize(path)
    step = max(_DIGEST_CHUNK_BYTES, total // 100)
    done = 0
    next_report = step
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_DIGEST_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            done += len(chunk)
            if progress is not None and total > 0 and done >= next_report:
                progress(min(1.0, done / total))
                next_report += step
    if progress is not None:
        progress(1.0)
    return digest.hexdigest()


# ----- containers and joins -----------------------------------------------------------------

_RIFF_HEADER_BYTES = 1 << 16
_FILETIME_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class MediaFacts:
    """What a container's header says: when the recorder started the file (UTC) and how long
    it runs; either is None when the header does not say."""

    start_utc: datetime | None
    duration_s: float | None


def _riff_chunks(data: bytes, pos: int, end: int):
    """The chunks between pos and end as (tag, body), descending into LIST chunks except the
    movie data, which is not a header and is left unread."""
    while pos + 8 <= end:
        tag = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = pos + 8
        if tag == b"LIST":
            if data[body:body + 4] == b"movi":
                return
            yield from _riff_chunks(data, body + 4, min(end, body + size))
        else:
            yield tag, data[body:min(end, body + size)]
        pos = body + size + (size & 1)


def avi_facts(path: PathLike) -> MediaFacts:
    """The start time and duration an AVI header carries.

    Court and interview recorders write the moment each file started as a UTC FILETIME in a
    TUTC chunk of the header. The duration is the audio stream's length over its rate from
    the stream header, else the frame count times the frame time from the main header. A
    file that is not an AVI gives no facts.
    """
    with open(path, "rb") as handle:
        head = handle.read(_RIFF_HEADER_BYTES)
    if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"AVI ":
        return MediaFacts(None, None)
    start: datetime | None = None
    audio_s: float | None = None
    frames_s: float | None = None
    for tag, body in _riff_chunks(head, 12, len(head)):
        if tag == b"TUTC" and len(body) >= 8:
            start = _FILETIME_EPOCH + timedelta(microseconds=struct.unpack_from("<Q", body, 0)[0] // 10)
        elif tag == b"avih" and len(body) >= 20:
            micros, _, _, _, frames = struct.unpack_from("<IIIII", body, 0)
            if micros and frames:
                frames_s = frames * micros / 1e6
        elif tag == b"strh" and len(body) >= 36 and body[:4] == b"auds" and audio_s is None:
            scale, rate, _, length = struct.unpack_from("<IIII", body, 20)
            if rate:
                audio_s = length * scale / rate
    return MediaFacts(start, audio_s if audio_s is not None else frames_s)


def _frames_of(path: PathLike) -> int:
    with _open_checked(path) as handle:
        return handle.getnframes()


def join_wavs(parts: Sequence[tuple[PathLike, float]], dst: PathLike) -> float:
    """Write the 16 kHz mono WAVs of `parts` into one WAV at dst, each placed at its offset in
    seconds from the start, in order: silence fills a gap, and a part that begins before the
    previous one ends cuts the previous one short at its own start. Returns the duration
    written, in seconds.
    """
    if not parts:
        raise ValueError("join_wavs needs at least one part")
    starts = [int(round(float(offset) * SAMPLE_RATE)) for _, offset in parts]
    if any(later < earlier for earlier, later in zip(starts, starts[1:])):
        raise ValueError("parts must be given in order of their offsets")
    lengths = [_frames_of(path) for path, _ in parts]
    for index in range(len(parts) - 1):
        lengths[index] = max(0, min(lengths[index], starts[index + 1] - starts[index]))
    silence = bytes(SAMPLE_RATE * SAMPLE_WIDTH_BYTES)
    written = 0
    with wave.open(str(dst), "wb") as out:
        out.setnchannels(CHANNELS)
        out.setsampwidth(SAMPLE_WIDTH_BYTES)
        out.setframerate(SAMPLE_RATE)
        for (path, _), start, length in zip(parts, starts, lengths):
            while written < start:
                step = min(SAMPLE_RATE, start - written)
                out.writeframes(silence[: step * SAMPLE_WIDTH_BYTES])
                written += step
            with _open_checked(path) as src:
                left = length
                while left > 0:
                    chunk = src.readframes(min(left, SAMPLE_RATE))
                    if not chunk:
                        break
                    out.writeframes(chunk)
                    left -= len(chunk) // SAMPLE_WIDTH_BYTES
                written += length - max(left, 0)
    return written / float(SAMPLE_RATE)


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
