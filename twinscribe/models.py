"""Model catalogue and the local model store.

Every engine loads its weights from a folder under one models root and nothing is fetched at
run time. The catalogue names the models the design measured: the folder each occupies under
the root, the files it must contain, the licence it is used under and the upstream source a
fetch tool outside this package pulls it from. The store reader finds which of them are
present, and the lock file records the digest of every fetched file so that a store can be
checked against what was originally downloaded.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twinscribe.paths import app_home
from twinscribe.runrecord import utc_now, write_json_atomic

MODELS_ENV = "TWINSCRIBE_MODELS"
LOCK_FILE = "models.lock.json"
LOCK_SCHEMA = "twinscribe.models-lock.v1"

ROLE_PUBLISHER = "publisher"
ROLE_DETECTOR = "detector"
ROLE_VAD = "vad"
ROLE_SEGMENTATION = "segmentation"
ROLE_EMBEDDING = "embedding"
ROLES: tuple[str, ...] = (ROLE_PUBLISHER, ROLE_DETECTOR, ROLE_VAD, ROLE_SEGMENTATION, ROLE_EMBEDDING)

STATUS_VERIFIED = "verified"
STATUS_MISMATCH = "mismatch"
STATUS_UNPINNED = "unpinned"
STATUS_MISSING = "missing"

_SHERPA_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
_HUB = "https://huggingface.co/"
_DIGEST_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class Source:
    """One upstream file: its URL and the name it takes inside the model folder.

    When archive is set, the URL points at a compressed archive and target names a member of
    it; the fetch tool extracts that member and drops the top-level folder of the archive.
    """

    url: str
    target: str
    archive: str | None = None


BACKEND_CT2 = "ct2"
BACKEND_ONNX = "onnx"


@dataclass(frozen=True)
class ModelSpec:
    """One model in the catalogue: where it sits, what it must contain, where it came from.

    backend names the library a detector model is for: ct2 (faster-whisper on CTranslate2)
    or onnx (sherpa-onnx); other roles leave it empty.
    """

    key: str
    role: str
    title: str
    licence: str
    credit: str
    required: tuple[str, ...]
    sources: tuple[Source, ...]
    note: str = ""
    backend: str = ""


def hub_sources(repo: str, files: tuple[str, ...], revision: str = "main") -> tuple[Source, ...]:
    """Sources for plain files in a model-hub repository at a fixed revision."""
    return tuple(Source(url=f"{_HUB}{repo}/resolve/{revision}/{name}", target=name) for name in files)


def archive_sources(url: str, members: tuple[str, ...]) -> tuple[Source, ...]:
    """Sources for members of one archive; the archive file name is taken from the URL."""
    archive = url.rsplit("/", 1)[-1]
    return tuple(Source(url=url, target=name, archive=archive) for name in members)


_WHISPER_FILES: tuple[str, ...] = (
    "model.bin",
    "config.json",
    "tokenizer.json",
    "vocabulary.json",
    "preprocessor_config.json",
)
# tokenizer.json must be present because the library falls back to a hub lookup without it,
# and the environment forbids that lookup.
_WHISPER_REQUIRED: tuple[str, ...] = ("model.bin", "config.json", "tokenizer.json", "vocabulary.json")
_TRANSDUCER_FILES: tuple[str, ...] = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")

KEY_PARAKEET_V2 = "parakeet-tdt-0.6b-v2-int8"
KEY_WHISPER_TURBO = "whisper-large-v3-turbo-ct2"
KEY_WHISPER_DISTIL = "whisper-distil-large-v3-ct2"
KEY_WHISPER_LARGE = "whisper-large-v3-ct2"
KEY_WHISPER_TURBO_ONNX = "whisper-turbo-onnx"
KEY_WHISPER_DISTIL_ONNX = "whisper-distil-large-v3-onnx"
KEY_WHISPER_LARGE_ONNX = "whisper-large-v3-onnx"
KEY_SILERO_VAD = "silero-vad"
KEY_SEGMENTATION = "pyannote-segmentation-3.0"
KEY_EMBEDDING = "titanet-large"

_ONNX_WHISPER_NOTE = (
    "Detector for machines without CTranslate2 (Windows on ARM); word times are spread inside "
    "the segment timestamps unless the export carries attention outputs."
)


def _onnx_whisper_files(stem: str) -> tuple[str, ...]:
    return (f"{stem}-encoder.int8.onnx", f"{stem}-decoder.int8.onnx", f"{stem}-tokens.txt")

CATALOGUE: tuple[ModelSpec, ...] = (
    ModelSpec(
        key=KEY_PARAKEET_V2,
        role=ROLE_PUBLISHER,
        title="NVIDIA Parakeet TDT 0.6B v2, int8 ONNX export",
        licence="CC BY 4.0",
        credit="NVIDIA Corporation; ONNX export distributed by the sherpa-onnx project",
        required=_TRANSDUCER_FILES,
        sources=archive_sources(
            _SHERPA_RELEASES + "asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
            _TRANSDUCER_FILES,
        ),
        note="The published engine.",
    ),
    ModelSpec(
        key=KEY_WHISPER_TURBO,
        role=ROLE_DETECTOR,
        title="Whisper large-v3-turbo, CTranslate2 conversion",
        licence="MIT",
        credit="OpenAI; conversion published by mobiuslabsgmbh",
        required=_WHISPER_REQUIRED,
        sources=hub_sources("mobiuslabsgmbh/faster-whisper-large-v3-turbo", _WHISPER_FILES),
        note="The detector the design measured; never published.",
        backend=BACKEND_CT2,
    ),
    ModelSpec(
        key=KEY_WHISPER_DISTIL,
        role=ROLE_DETECTOR,
        title="Distil-Whisper large-v3, CTranslate2 conversion",
        licence="MIT",
        credit="OpenAI and the Distil-Whisper authors; conversion published by Systran",
        required=_WHISPER_REQUIRED,
        sources=hub_sources("Systran/faster-distil-whisper-large-v3", _WHISPER_FILES),
        note="Faster detector for the quick profile.",
        backend=BACKEND_CT2,
    ),
    ModelSpec(
        key=KEY_WHISPER_LARGE,
        role=ROLE_DETECTOR,
        title="Whisper large-v3, CTranslate2 conversion",
        licence="MIT",
        credit="OpenAI; conversion published by Systran",
        required=_WHISPER_REQUIRED,
        sources=hub_sources("Systran/faster-whisper-large-v3", _WHISPER_FILES),
        note="Full-size detector for the careful profile.",
        backend=BACKEND_CT2,
    ),
    ModelSpec(
        key=KEY_WHISPER_TURBO_ONNX,
        role=ROLE_DETECTOR,
        title="Whisper large-v3-turbo, int8 ONNX export",
        licence="MIT",
        credit="OpenAI; ONNX export distributed by the sherpa-onnx project",
        required=_onnx_whisper_files("turbo"),
        sources=archive_sources(
            _SHERPA_RELEASES + "asr-models/sherpa-onnx-whisper-turbo.tar.bz2", _onnx_whisper_files("turbo")
        ),
        note=_ONNX_WHISPER_NOTE,
        backend=BACKEND_ONNX,
    ),
    ModelSpec(
        key=KEY_WHISPER_DISTIL_ONNX,
        role=ROLE_DETECTOR,
        title="Distil-Whisper large-v3, int8 ONNX export",
        licence="MIT",
        credit="OpenAI and the Distil-Whisper authors; ONNX export distributed by the sherpa-onnx project",
        required=_onnx_whisper_files("distil-large-v3"),
        sources=archive_sources(
            _SHERPA_RELEASES + "asr-models/sherpa-onnx-whisper-distil-large-v3.tar.bz2",
            _onnx_whisper_files("distil-large-v3"),
        ),
        note=_ONNX_WHISPER_NOTE,
        backend=BACKEND_ONNX,
    ),
    ModelSpec(
        key=KEY_WHISPER_LARGE_ONNX,
        role=ROLE_DETECTOR,
        title="Whisper large-v3, int8 ONNX export",
        licence="MIT",
        credit="OpenAI; ONNX export distributed by the sherpa-onnx project",
        required=_onnx_whisper_files("large-v3"),
        sources=archive_sources(
            _SHERPA_RELEASES + "asr-models/sherpa-onnx-whisper-large-v3.tar.bz2", _onnx_whisper_files("large-v3")
        ),
        note=_ONNX_WHISPER_NOTE,
        backend=BACKEND_ONNX,
    ),
    ModelSpec(
        key=KEY_SILERO_VAD,
        role=ROLE_VAD,
        title="Silero voice activity detector",
        licence="MIT",
        credit="Silero Team",
        required=("silero_vad.onnx",),
        sources=(Source(url=_SHERPA_RELEASES + "asr-models/silero_vad.onnx", target="silero_vad.onnx"),),
    ),
    ModelSpec(
        key=KEY_SEGMENTATION,
        role=ROLE_SEGMENTATION,
        title="pyannote segmentation-3.0, ONNX export",
        licence="MIT",
        credit="The pyannote.audio authors; ONNX export distributed by the sherpa-onnx project",
        required=("model.onnx",),
        sources=archive_sources(
            _SHERPA_RELEASES + "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
            ("model.onnx",),
        ),
    ),
    ModelSpec(
        key=KEY_EMBEDDING,
        role=ROLE_EMBEDDING,
        title="NVIDIA TitaNet-large speaker embeddings, ONNX export",
        licence="CC BY 4.0",
        credit="NVIDIA Corporation; ONNX export distributed by the sherpa-onnx project",
        required=("nemo_en_titanet_large.onnx",),
        sources=(
            Source(
                url=_SHERPA_RELEASES + "speaker-recongition-models/nemo_en_titanet_large.onnx",
                target="nemo_en_titanet_large.onnx",
            ),
        ),
        note="The spelling of the release tag is the upstream project's own.",
    ),
)

_BY_KEY: dict[str, ModelSpec] = {spec.key: spec for spec in CATALOGUE}


def spec_for(key: str) -> ModelSpec:
    """The catalogue entry for a folder key; KeyError lists the known keys."""
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; known models: {', '.join(sorted(_BY_KEY))}") from None


def specs_for_role(role: str) -> tuple[ModelSpec, ...]:
    """Catalogue entries that can play a role, in catalogue order."""
    if role not in ROLES:
        raise KeyError(f"unknown role {role!r}; roles: {', '.join(ROLES)}")
    return tuple(spec for spec in CATALOGUE if spec.role == role)


@dataclass(frozen=True)
class ModelSet:
    """What a models root holds: complete model folders, and folders that lack files."""

    root: Path
    present: dict[str, Path]
    incomplete: dict[str, tuple[str, ...]]

    def has(self, key: str) -> bool:
        return key in self.present

    def path(self, key: str) -> Path:
        """Folder of a present model; KeyError names the folder that was expected."""
        try:
            return self.present[key]
        except KeyError:
            spec = _BY_KEY.get(key)
            title = spec.title if spec is not None else key
            raise KeyError(f"{title} is not present; expected folder {self.root / key}") from None

    def file(self, key: str, name: str) -> Path:
        """One file inside a present model folder."""
        return self.path(key) / name

    @property
    def missing(self) -> tuple[str, ...]:
        """Catalogue keys with no complete folder under the root."""
        return tuple(spec.key for spec in CATALOGUE if spec.key not in self.present)

    def describe(self) -> str:
        """One line per catalogue entry: present, incomplete (naming the files) or absent."""
        lines = []
        for spec in CATALOGUE:
            if spec.key in self.present:
                state = "present"
            elif spec.key in self.incomplete:
                state = "incomplete, lacks " + ", ".join(self.incomplete[spec.key])
            else:
                state = "absent"
            lines.append(f"{spec.key:<32} {spec.role:<13} {state}")
        return "\n".join(lines)


def default_models_root() -> Path:
    """The models root: the environment variable, else a models folder beside the package
    (a checkout or a portable copy), else one under the application home."""
    override = os.environ.get(MODELS_ENV)
    if override:
        return Path(override)
    beside = Path(__file__).resolve().parent.parent / "models"
    if beside.is_dir():
        return beside
    return app_home() / "models"


def find_models(root: str | os.PathLike[str] | None = None) -> ModelSet:
    """Read a models root and report which catalogue models are complete there."""
    base = Path(root) if root is not None else default_models_root()
    present: dict[str, Path] = {}
    incomplete: dict[str, tuple[str, ...]] = {}
    for spec in CATALOGUE:
        folder = base / spec.key
        if not folder.is_dir():
            continue
        missing = tuple(name for name in spec.required if not (folder / name).is_file())
        if missing:
            incomplete[spec.key] = missing
        else:
            present[spec.key] = folder
    return ModelSet(root=base, present=present, incomplete=incomplete)


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_DIGEST_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def lock_path(root: str | os.PathLike[str]) -> Path:
    return Path(root) / LOCK_FILE


def read_lock(root: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Lock entries keyed "<model key>/<file>"; empty when the lock is absent or unreadable."""
    path = lock_path(root)
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(doc, dict) or doc.get("schema") != LOCK_SCHEMA:
        return {}
    entries = doc.get("entries")
    return dict(entries) if isinstance(entries, dict) else {}


def write_lock(root: str | os.PathLike[str], entries: dict[str, dict[str, Any]]) -> Path:
    """Write the lock document atomically and return its path."""
    path = lock_path(root)
    write_json_atomic({"schema": LOCK_SCHEMA, "updated_utc": utc_now(), "entries": entries}, path)
    return path


def lock_entry(spec: ModelSpec, source: Source, path: Path, revision: str | None = None) -> dict[str, Any]:
    """The lock record for one fetched file: digest, size, source and time."""
    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "url": source.url,
        "archive": source.archive,
        "revision": revision,
        "licence": spec.licence,
        "fetched_utc": utc_now(),
    }


@dataclass(frozen=True)
class StoreCheck:
    """Verdict for one required file of one model."""

    key: str
    file: str
    status: str
    detail: str


def verify_store(
    root: str | os.PathLike[str] | None = None, keys: tuple[str, ...] | None = None
) -> list[StoreCheck]:
    """Digest every required file of the chosen models against the lock.

    Statuses: verified (the digest matches the lock), mismatch (it does not), unpinned
    (present but never recorded in the lock) and missing. Hashing gigabyte files takes time,
    so this is for an explicit check rather than every start.
    """
    models = find_models(root)
    lock = read_lock(models.root)
    checks: list[StoreCheck] = []
    for spec in CATALOGUE:
        if keys is not None and spec.key not in keys:
            continue
        folder = models.root / spec.key
        for name in spec.required:
            path = folder / name
            if not path.is_file():
                checks.append(StoreCheck(spec.key, name, STATUS_MISSING, f"expected at {path}"))
                continue
            entry = lock.get(f"{spec.key}/{name}")
            if entry is None:
                checks.append(StoreCheck(spec.key, name, STATUS_UNPINNED, "present, no digest recorded"))
                continue
            digest = sha256_file(path)
            if digest == entry.get("sha256"):
                checks.append(StoreCheck(spec.key, name, STATUS_VERIFIED, digest[:16]))
            else:
                recorded = str(entry.get("sha256"))[:16]
                checks.append(StoreCheck(spec.key, name, STATUS_MISMATCH, f"have {digest[:16]}, lock {recorded}"))
    return checks
