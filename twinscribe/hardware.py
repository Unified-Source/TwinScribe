"""Machine probe and acceleration plan.

What the machine offers (operating system, architecture, cores, memory, an NVIDIA GPU), which
engine libraries are importable, and from those a plan: where each engine runs (the processor
or a CUDA device), with which compute type or provider, how many threads, and whether the two
transcription engines can run at the same time. The plan is chosen once per run, recorded in
the run record and shown before a batch starts, so a run always states what it used. Probes
never raise and never reach the network.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import importlib.util
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from twinscribe.audio import NO_WINDOW

ARCH_X86_64 = "x86_64"
ARCH_ARM64 = "arm64"
ARCH_OTHER = "other"

OS_WINDOWS = "windows"
OS_LINUX = "linux"
OS_MACOS = "macos"

DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
PREFERENCES: tuple[str, ...] = (DEVICE_AUTO, DEVICE_CPU, DEVICE_CUDA)

BACKEND_CT2 = "ct2"
BACKEND_ONNX = "onnx"
BACKEND_NONE = "none"

PROVIDER_CPU = "cpu"
PROVIDER_CUDA = "cuda"

MAX_DEFAULT_THREADS = 8
LOW_MEMORY_GB = 6.0

# Compute types in order of preference per device. int8 on the processor is the measured
# configuration; float16 on a CUDA device is the usual precision for these models.
_CT2_PREFERENCE: dict[str, tuple[str, ...]] = {
    DEVICE_CPU: ("int8", "int8_float32", "float32"),
    DEVICE_CUDA: ("float16", "int8_float16", "float32"),
}

_cuda_registered: list[str] = []
_cuda_handles: list[Any] = []


def normalise_arch(machine: str | None) -> str:
    """Map the many spellings of an architecture to x86_64, arm64 or other."""
    value = (machine or "").strip().lower()
    if value in ("amd64", "x86_64", "x64", "em64t"):
        return ARCH_X86_64
    if value in ("arm64", "aarch64", "arm64e") or value.startswith("armv8"):
        return ARCH_ARM64
    return ARCH_OTHER


def os_key(platform_name: str | None = None) -> str:
    """windows, linux or macos from a sys.platform value; other values pass through."""
    value = platform_name if platform_name is not None else sys.platform
    if value.startswith("win"):
        return OS_WINDOWS
    if value.startswith("linux"):
        return OS_LINUX
    if value.startswith("darwin"):
        return OS_MACOS
    return value


@dataclass(frozen=True)
class Machine:
    """The class of machine, never its identity."""

    os: str
    arch: str
    machine_raw: str
    cores: int
    memory_gb: float | None
    processor: str | None

    @property
    def platform_key(self) -> str:
        return f"{self.os}-{self.arch}"


def probe_machine() -> Machine:
    """Operating system, architecture, cores, memory and processor name."""
    from twinscribe.runrecord import machine_facts

    facts = machine_facts()
    return Machine(
        os=os_key(),
        arch=normalise_arch(platform.machine()),
        machine_raw=platform.machine(),
        cores=int(facts.get("logical_cores") or os.cpu_count() or 1),
        memory_gb=facts.get("memory_gb"),
        processor=facts.get("processor"),
    )


@dataclass(frozen=True)
class Gpu:
    """One NVIDIA device as the driver reports it."""

    name: str
    memory_mb: int | None
    driver: str | None


def probe_nvidia_gpus(timeout_s: float = 5.0) -> tuple[Gpu, ...]:
    """NVIDIA devices from the driver's command line tool; empty when there is none."""
    executable = _which("nvidia-smi")
    if executable is None:
        return ()
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
            creationflags=NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()
    gpus: list[Gpu] = []
    for line in completed.stdout.decode("utf-8", errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        memory: int | None = None
        if len(parts) > 1:
            try:
                memory = int(float(parts[1]))
            except ValueError:
                memory = None
        gpus.append(Gpu(name=parts[0], memory_mb=memory, driver=parts[2] if len(parts) > 2 else None))
    return tuple(gpus)


def _which(name: str) -> str | None:
    import shutil

    found = shutil.which(name)
    if found is not None:
        return found
    if sys.platform == "win32":
        for candidate in (
            Path(os.environ.get("ProgramFiles", "")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
            Path(os.environ.get("SystemRoot", "")) / "System32" / "nvidia-smi.exe",
        ):
            if candidate.is_file():
                return str(candidate)
    return None


@dataclass(frozen=True)
class Libraries:
    """Which engine libraries can be imported, and their versions when installed."""

    faster_whisper: bool
    ctranslate2: bool
    sherpa_onnx: bool
    onnxruntime: bool
    versions: dict[str, str] = field(default_factory=dict)

    @property
    def whisper_ct2(self) -> bool:
        return self.faster_whisper and self.ctranslate2

    @property
    def sherpa_cuda_build(self) -> bool:
        """True when the installed sherpa-onnx is its CUDA build (the version string says so)."""
        return "cuda" in self.versions.get("sherpa-onnx", "").lower()


def _present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_libraries() -> Libraries:
    """Importability by module lookup only; nothing heavy is imported here."""
    versions: dict[str, str] = {}
    for distribution in ("faster-whisper", "ctranslate2", "sherpa-onnx", "onnxruntime"):
        version = _version(distribution)
        if version is not None:
            versions[distribution] = version
    return Libraries(
        faster_whisper=_present("faster_whisper"),
        ctranslate2=_present("ctranslate2"),
        sherpa_onnx=_present("sherpa_onnx"),
        onnxruntime=_present("onnxruntime"),
        versions=versions,
    )


def _nvidia_package_dirs() -> list[Path]:
    """Folders of the CUDA runtime libraries installed as packages (cublas, cudnn and kin)."""
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    found: list[Path] = []
    for location in spec.submodule_search_locations:
        base = Path(location)
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            for leaf in ("bin", "lib"):
                folder = child / leaf
                if folder.is_dir():
                    found.append(folder)
    return found


def register_cuda_libraries() -> tuple[str, ...]:
    """Make CUDA runtime libraries installed as packages visible to the engines.

    On Windows the folders holding the DLLs join the process search path; on Linux the shared
    objects are loaded ahead of time so that a later load by name finds them. Idempotent; a
    no-op when no such package is installed. Returns the folders registered.
    """
    if _cuda_registered:
        return tuple(_cuda_registered)
    folders = _nvidia_package_dirs()
    if not folders:
        return ()
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        for folder in folders:
            if any(folder.glob("*.dll")):
                try:
                    _cuda_handles.append(os.add_dll_directory(str(folder)))
                    _cuda_registered.append(str(folder))
                except OSError:
                    continue
    elif sys.platform.startswith("linux"):
        wanted = ("libcublasLt.so", "libcublas.so", "libcudnn.so", "libcudart.so")
        for folder in folders:
            for library in sorted(folder.glob("*.so*")):
                if any(library.name.startswith(prefix) for prefix in wanted):
                    try:
                        _cuda_handles.append(ctypes.CDLL(str(library)))
                        if str(folder) not in _cuda_registered:
                            _cuda_registered.append(str(folder))
                    except OSError:
                        continue
    return tuple(_cuda_registered)


def cuda_device_count() -> int:
    """CUDA devices CTranslate2 can use, 0 without the library or its runtime."""
    if not _present("ctranslate2"):
        return 0
    register_cuda_libraries()
    try:
        ctranslate2 = importlib.import_module("ctranslate2")
        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 - a missing runtime must read as no device
        return 0


def ct2_compute_types(device: str) -> frozenset[str]:
    """Compute types CTranslate2 supports on a device; empty without the library."""
    if not _present("ctranslate2"):
        return frozenset()
    try:
        ctranslate2 = importlib.import_module("ctranslate2")
        return frozenset(str(t) for t in ctranslate2.get_supported_compute_types(device))
    except Exception:  # noqa: BLE001
        return frozenset()


def choose_compute_type(device: str, supported: frozenset[str] | set[str]) -> str:
    """The preferred compute type for a device among those supported, else "auto"."""
    for candidate in _CT2_PREFERENCE.get(device, ()):
        if candidate in supported:
            return candidate
    return "auto"


@dataclass(frozen=True)
class Placement:
    """Where one engine runs."""

    device: str
    index: int = 0
    compute_type: str | None = None
    provider: str = PROVIDER_CPU
    threads: int = 1

    def describe(self) -> str:
        where = f"{self.device}:{self.index}" if self.device == DEVICE_CUDA else "processor"
        detail = f", {self.compute_type}" if self.compute_type else ""
        threads = f", {self.threads} threads" if self.device == DEVICE_CPU else ""
        return f"{where}{detail}{threads}"


@dataclass(frozen=True)
class Plan:
    """The acceleration plan for one run.

    detector is the placement of the preferred detector backend; detector_ct2 and
    detector_onnx are the placements each backend would get, None when its library is
    absent, so that a level whose model exists only for the other backend can still run.
    backends says which detector libraries are usable.
    """

    platform_key: str
    preference: str
    detector_backend: str
    detector: Placement | None
    publisher: Placement
    diarizer: Placement
    parallel: bool
    notes: tuple[str, ...] = ()
    detector_ct2: Placement | None = None
    detector_onnx: Placement | None = None
    backends: dict[str, bool] = field(default_factory=dict)

    def placement_for(self, backend: str) -> Placement | None:
        """The placement of a detector backend, or None when its library is absent."""
        if backend == BACKEND_CT2:
            return self.detector_ct2
        if backend == BACKEND_ONNX:
            return self.detector_onnx
        return None

    def parallel_for(self, backend: str) -> bool:
        """Whether a detector of this backend runs beside the publisher (different devices)."""
        placement = self.placement_for(backend)
        return placement is not None and placement.device != self.publisher.device

    def describe(self) -> list[str]:
        """Lines for a person: one per engine, then the notes."""
        lines = []
        if self.detector is None:
            lines.append("Detector: none available; install faster-whisper with CTranslate2, or sherpa-onnx")
        else:
            backend = "CTranslate2" if self.detector_backend == BACKEND_CT2 else "ONNX through sherpa-onnx"
            lines.append(f"Detector (Whisper): {backend} on {self.detector.describe()}")
        lines.append(f"Publisher (transducer): sherpa-onnx, provider {self.publisher.provider}, {self.publisher.describe()}")
        lines.append(f"Speakers: sherpa-onnx, provider {self.diarizer.provider}, {self.diarizer.describe()}")
        lines.append("The two transcription engines run " + ("at the same time" if self.parallel else "one after the other"))
        lines.extend(self.notes)
        return lines

    def to_dict(self) -> dict[str, Any]:
        def placement(p: Placement | None) -> dict[str, Any] | None:
            if p is None:
                return None
            return {"device": p.device, "index": p.index, "compute_type": p.compute_type, "provider": p.provider, "threads": p.threads}

        return {
            "platform": self.platform_key,
            "preference": self.preference,
            "detector_backend": self.detector_backend,
            "detector": placement(self.detector),
            "detector_ct2": placement(self.detector_ct2),
            "detector_onnx": placement(self.detector_onnx),
            "publisher": placement(self.publisher),
            "diarizer": placement(self.diarizer),
            "parallel": self.parallel,
            "backends": dict(self.backends),
            "notes": list(self.notes),
        }


def default_thread_count(cores: int, threads: int | None = None) -> int:
    """The thread count for an engine: the request, else min of 8 and the cores."""
    if threads is not None:
        if threads < 1:
            raise ValueError(f"threads must be at least 1, got {threads}")
        return int(threads)
    return max(1, min(MAX_DEFAULT_THREADS, int(cores)))


def make_plan(
    machine: Machine,
    libraries: Libraries,
    gpus: tuple[Gpu, ...] = (),
    cuda_count: int = 0,
    preference: str = DEVICE_AUTO,
    threads: int | None = None,
    ct2_types_cpu: frozenset[str] | set[str] = frozenset(),
    ct2_types_cuda: frozenset[str] | set[str] = frozenset(),
) -> Plan:
    """The pure policy: from what was probed, decide where every engine runs.

    The detector uses CTranslate2 when faster-whisper and CTranslate2 import, else the ONNX
    path through sherpa-onnx. A CUDA device is used when the preference allows it and the
    library can see one; the transducer and the speaker models use a CUDA provider only when
    the installed sherpa-onnx is its CUDA build. The two transcription engines run at the
    same time only when they sit on different devices.
    """
    if preference not in PREFERENCES:
        raise ValueError(f"preference must be one of {', '.join(PREFERENCES)}, got {preference!r}")
    thread_count = default_thread_count(machine.cores, threads)
    notes: list[str] = []
    want_cuda = preference != DEVICE_CPU
    if preference == DEVICE_CUDA and cuda_count == 0:
        notes.append("A CUDA device was asked for but none is usable; the engines run on the processor")

    sherpa_provider = PROVIDER_CPU
    if want_cuda and cuda_count > 0:
        if libraries.sherpa_cuda_build:
            sherpa_provider = PROVIDER_CUDA
        elif libraries.sherpa_onnx:
            notes.append(
                "sherpa-onnx is the processor build; its CUDA build would run the transducer and the speaker models on the device"
            )
    sherpa_device = DEVICE_CUDA if sherpa_provider == PROVIDER_CUDA else DEVICE_CPU
    publisher = Placement(sherpa_device, 0, None, sherpa_provider, thread_count)
    diarizer = Placement(sherpa_device, 0, None, sherpa_provider, thread_count)

    detector_ct2: Placement | None = None
    if libraries.whisper_ct2:
        if want_cuda and cuda_count > 0:
            detector_ct2 = Placement(DEVICE_CUDA, 0, choose_compute_type(DEVICE_CUDA, ct2_types_cuda), PROVIDER_CUDA, thread_count)
        else:
            detector_ct2 = Placement(DEVICE_CPU, 0, choose_compute_type(DEVICE_CPU, ct2_types_cpu), PROVIDER_CPU, thread_count)
            if want_cuda and gpus and cuda_count == 0:
                notes.append(
                    "An NVIDIA device is present but CTranslate2 cannot use it; install the CUDA runtime packages "
                    "(cuBLAS and cuDNN) beside the libraries to run the detector on it"
                )
    detector_onnx: Placement | None = None
    if libraries.sherpa_onnx:
        detector_onnx = Placement(sherpa_device, 0, None, sherpa_provider, thread_count)

    if detector_ct2 is not None:
        backend = BACKEND_CT2
        detector: Placement | None = detector_ct2
    elif detector_onnx is not None:
        backend = BACKEND_ONNX
        detector = detector_onnx
        if machine.platform_key == f"{OS_WINDOWS}-{ARCH_ARM64}":
            notes.append("CTranslate2 publishes no wheel for Windows on ARM; the detector runs through sherpa-onnx")
        else:
            notes.append("faster-whisper or CTranslate2 is not installed; the detector runs through sherpa-onnx")
    else:
        backend = BACKEND_NONE
        detector = None
        notes.append("No detector library is installed; only playback and the outputs of earlier runs are available")

    parallel = detector is not None and detector.device != publisher.device
    if machine.memory_gb is not None and machine.memory_gb < LOW_MEMORY_GB:
        notes.append(f"Only {machine.memory_gb:.0f} GB of memory; the standard level may not fit, the quick level is safer")
    return Plan(
        platform_key=machine.platform_key,
        preference=preference,
        detector_backend=backend,
        detector=detector,
        publisher=publisher,
        diarizer=diarizer,
        parallel=parallel,
        notes=tuple(notes),
        detector_ct2=detector_ct2,
        detector_onnx=detector_onnx,
        backends={BACKEND_CT2: detector_ct2 is not None, BACKEND_ONNX: detector_onnx is not None},
    )


def current_plan(preference: str = DEVICE_AUTO, threads: int | None = None) -> Plan:
    """Probe this machine and decide the plan; the CUDA probe imports CTranslate2 when present."""
    machine = probe_machine()
    libraries = probe_libraries()
    gpus = probe_nvidia_gpus()
    want_cuda = preference != DEVICE_CPU
    cuda_count = cuda_device_count() if want_cuda and libraries.ctranslate2 else 0
    cpu_types = ct2_compute_types(DEVICE_CPU) if libraries.ctranslate2 else frozenset()
    cuda_types = ct2_compute_types(DEVICE_CUDA) if cuda_count > 0 else frozenset()
    return make_plan(machine, libraries, gpus, cuda_count, preference, threads, cpu_types, cuda_types)


def describe_machine(machine: Machine, gpus: tuple[Gpu, ...], libraries: Libraries) -> list[str]:
    """Lines for a person describing the machine and the libraries."""
    lines = [
        f"Machine: {machine.os}, {machine.arch} ({machine.machine_raw}), {machine.cores} cores, "
        f"{machine.memory_gb if machine.memory_gb is not None else '?'} GB, {machine.processor or 'unknown processor'}",
    ]
    if gpus:
        for gpu in gpus:
            memory = f", {gpu.memory_mb} MB" if gpu.memory_mb else ""
            driver = f", driver {gpu.driver}" if gpu.driver else ""
            lines.append(f"NVIDIA device: {gpu.name}{memory}{driver}")
    else:
        lines.append("NVIDIA device: none found")
    for name, present in (
        ("faster-whisper", libraries.faster_whisper),
        ("CTranslate2", libraries.ctranslate2),
        ("sherpa-onnx", libraries.sherpa_onnx),
        ("onnxruntime", libraries.onnxruntime),
    ):
        key = name.lower()
        version = libraries.versions.get(key, "")
        lines.append(f"{name}: " + (f"installed {version}".strip() if present else "not installed"))
    return lines
