"""Public corpora the bench measures on, pinned so that widening a download never redefines an
item already measured.

Each corpus names its licence, its credit line and the exact upstream revision its bytes come
from, and lists the files the store must hold. The fetch tool outside the package downloads
them into a corpora root and records every file's digest in the lock; the bench builds its
items from that store and nothing else. Item ids carry the upstream ids they are built from
(a call id, a meeting name with a window, an utterance id), so adding an item changes no
other, and a file whose digest no longer matches the lock is reported rather than used.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twinscribe.models import sha256_file
from twinscribe.paths import app_home
from twinscribe.runrecord import utc_now, write_json_atomic

CORPORA_ENV = "TWINSCRIBE_CORPORA"
LOCK_FILE = "corpora.lock.json"
LOCK_SCHEMA = "twinscribe.corpora-lock.v1"

STATUS_VERIFIED = "verified"
STATUS_MISMATCH = "mismatch"
STATUS_UNPINNED = "unpinned"
STATUS_MISSING = "missing"

KEY_HVB = "harper-valley-bank"
KEY_AMI = "ami-es2002a"
KEY_LIBRISPEECH = "librispeech-test-other"

# HarperValleyBank: role-played bank calls at 8 kHz, one channel per party. The commit pins
# the bytes. The call list is the first calls in the sorted order of their ids; it is only
# ever appended to, never reordered or edited, so that every call keeps its identity across
# widenings of the set.
HVB_COMMIT = "0bd721e877c4a85d8c13ff837e68661ea6200a98"
HVB_RAW = "https://raw.githubusercontent.com/cricketclub/gridspace-stanford-harper-valley/" + HVB_COMMIT + "/data/"
HVB_CALLS: tuple[str, ...] = (
    "0002f70f7386445b", "004860b1ab2e4c88", "0091a706bc604188", "00d676d7058c49bb",
    "00f7dce6fc3849a2", "010d38f5ada54e0d", "010eaccb7a23436f", "0126ffdce48049a9",
    "0188295665114e74", "01cefd6f5c044a6f", "01f7ec3700424bc0", "020e48edcf0940a4",
    "021cd80ca7cc464b", "0224c92b64d144d4", "02b03e407894474b", "02e41649e7c441fd",
    "02fd023b18d246d0", "0317f95f3d7441c5", "034a32d3b6e4435a", "035edd1d09c1433e",
    "0377245f73a54480", "0395f6997a8e4836", "03a17cc36d474151", "03aad8e17c8d4d81",
)
HVB_AGENT = "agent.wav"
HVB_CALLER = "caller.wav"
HVB_TRANSCRIPT = "transcript.json"
HVB_METADATA = "metadata.json"
HVB_CALL_FILES: tuple[str, ...] = (HVB_AGENT, HVB_CALLER, HVB_TRANSCRIPT, HVB_METADATA)

# AMI meeting ES2002a: four speakers, the headset mix and one far-field array microphone of
# the same words, and the manual annotation release that carries word times and segments.
AMI_MEETING = "ES2002a"
AMI_AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/"
AMI_ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AMI_ANNOTATIONS_ARCHIVE = "ami_public_manual_1.6.2.zip"
AMI_AGENTS: tuple[str, ...] = ("A", "B", "C", "D")
AMI_HEADSET = "ES2002a.Mix-Headset.wav"
AMI_ARRAY = "ES2002a.Array1-01.wav"
AMI_MEETINGS = "corpusResources/meetings.xml"
AMI_WINDOW: tuple[float, float] = (780.0, 960.0)

# LibriSpeech test-other: read speech, for decode smoke tests. The archive is large and only
# a pinned selection is extracted: the first chapter folders in sorted order and the first
# utterances of each, plus each chapter's transcript file.
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-other.tar.gz"
LIBRISPEECH_ARCHIVE = "test-other.tar.gz"
LIBRISPEECH_PREFIX = "LibriSpeech/test-other/"
LIBRISPEECH_CHAPTERS = 2
LIBRISPEECH_PER_CHAPTER = 10
LIBRISPEECH_FOLDER = "utterances"
SELECTION_LIBRISPEECH = "librispeech"


@dataclass(frozen=True)
class CorpusFile:
    """One file of the store: its URL, its path under the corpus folder, and, when it is a
    member of an archive, the archive's file name and the member's path inside it."""

    url: str
    target: str
    archive: str | None = None
    member: str | None = None


@dataclass(frozen=True)
class CorpusSpec:
    """One corpus: identity, licence, the revision its bytes come from, and its files.

    selection names a rule the fetch tool applies to an archive's member list when the files
    cannot be listed in advance; files is then empty, the archive URL is carried in
    archive_url, and the store is complete when the selected members are present.
    """

    key: str
    title: str
    licence: str
    credit: str
    revision: str
    files: tuple[CorpusFile, ...]
    note: str = ""
    selection: str | None = None
    archive_url: str | None = None


def hvb_files(sid: str) -> tuple[CorpusFile, ...]:
    """The four files of one call, under calls/<sid>/."""
    return (
        CorpusFile(url=f"{HVB_RAW}audio/agent/{sid}.wav", target=f"calls/{sid}/{HVB_AGENT}"),
        CorpusFile(url=f"{HVB_RAW}audio/caller/{sid}.wav", target=f"calls/{sid}/{HVB_CALLER}"),
        CorpusFile(url=f"{HVB_RAW}transcript/{sid}.json", target=f"calls/{sid}/{HVB_TRANSCRIPT}"),
        CorpusFile(url=f"{HVB_RAW}metadata/{sid}.json", target=f"calls/{sid}/{HVB_METADATA}"),
    )


def ami_annotation_members() -> tuple[str, ...]:
    """The members of the annotation archive the reference needs: the meeting map, and the
    words and segments of every speaker."""
    members = [AMI_MEETINGS]
    for agent in AMI_AGENTS:
        for kind in ("words", "segments"):
            members.append(f"{kind}/{AMI_MEETING}.{agent}.{kind}.xml")
    return tuple(members)


def _ami_files() -> tuple[CorpusFile, ...]:
    files = [
        CorpusFile(url=AMI_AUDIO_URL + AMI_HEADSET, target=f"audio/{AMI_HEADSET}"),
        CorpusFile(url=AMI_AUDIO_URL + AMI_ARRAY, target=f"audio/{AMI_ARRAY}"),
    ]
    for member in ami_annotation_members():
        files.append(
            CorpusFile(url=AMI_ANNOTATIONS_URL, target=f"annotations/{member}", archive=AMI_ANNOTATIONS_ARCHIVE, member=member)
        )
    return tuple(files)


CATALOGUE: tuple[CorpusSpec, ...] = (
    CorpusSpec(
        key=KEY_HVB,
        title="HarperValleyBank role-played telephone calls",
        licence="CC BY 4.0",
        credit="Gridspace and Stanford University (CS224S)",
        revision=HVB_COMMIT,
        files=tuple(file for sid in HVB_CALLS for file in hvb_files(sid)),
        note=(
            "8 kHz telephone audio, one channel per party, human transcripts with segment times. "
            "The caller channel defines the timeline; the agent channel starts later by the "
            "difference in channel length."
        ),
    ),
    CorpusSpec(
        key=KEY_AMI,
        title="AMI Meeting Corpus, meeting ES2002a",
        licence="CC BY 4.0",
        credit="AMI Consortium",
        revision="manual annotations 1.6.2",
        files=_ami_files(),
        note=(
            "Four speakers; the headset mix and one far-field array microphone of the same words. "
            "Word times come from forced alignment and are least reliable under overlap; the "
            "corpus defects page records participant A wearing the headset improperly."
        ),
    ),
    CorpusSpec(
        key=KEY_LIBRISPEECH,
        title="LibriSpeech test-other",
        licence="CC BY 4.0",
        credit="Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur; OpenSLR resource 12",
        revision="OpenSLR 12, test-other",
        files=(),
        note="Clean read speech, one reader per utterance; smoke and decode tests only.",
        selection=SELECTION_LIBRISPEECH,
        archive_url=LIBRISPEECH_URL,
    ),
)

_BY_KEY: dict[str, CorpusSpec] = {spec.key: spec for spec in CATALOGUE}


def spec_for(key: str) -> CorpusSpec:
    """The catalogue entry for a corpus key; KeyError lists the known keys."""
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown corpus {key!r}; known corpora: {', '.join(sorted(_BY_KEY))}") from None


def librispeech_members(
    names: Iterable[str], chapters: int = LIBRISPEECH_CHAPTERS, per_chapter: int = LIBRISPEECH_PER_CHAPTER
) -> list[tuple[str, str]]:
    """(member, target) pairs for the pinned selection out of an archive's member list.

    Chapter folders are taken in sorted order, the first `chapters` of them; within each the
    first `per_chapter` utterances in sorted order, plus the chapter's transcript file. The
    targets are flat under the utterances folder, named as upstream.
    """
    flac = sorted(name for name in names if name.startswith(LIBRISPEECH_PREFIX) and name.endswith(".flac"))
    by_chapter: dict[str, list[str]] = {}
    for name in flac:
        by_chapter.setdefault(name.rsplit("/", 1)[0], []).append(name)
    pairs: list[tuple[str, str]] = []
    for chapter in sorted(by_chapter)[:chapters]:
        speaker, chapter_id = chapter.rsplit("/", 2)[-2:]
        transcript = f"{chapter}/{speaker}-{chapter_id}.trans.txt"
        pairs.append((transcript, f"{LIBRISPEECH_FOLDER}/{speaker}-{chapter_id}.trans.txt"))
        for name in by_chapter[chapter][:per_chapter]:
            pairs.append((name, f"{LIBRISPEECH_FOLDER}/{name.rsplit('/', 1)[-1]}"))
    return pairs


def select_members(spec: CorpusSpec, names: Iterable[str]) -> list[tuple[str, str]]:
    """Apply a corpus's selection rule to an archive's member list."""
    if spec.selection == SELECTION_LIBRISPEECH:
        return librispeech_members(names)
    raise KeyError(f"corpus {spec.key!r} has no selection rule")


@dataclass(frozen=True)
class CorpusStore:
    """What a corpora root holds: complete corpus folders, and folders that lack files."""

    root: Path
    present: dict[str, Path]
    incomplete: dict[str, tuple[str, ...]]

    def has(self, key: str) -> bool:
        return key in self.present

    def path(self, key: str) -> Path:
        """Folder of a present corpus; KeyError names the folder that was expected."""
        try:
            return self.present[key]
        except KeyError:
            raise KeyError(f"{key} is not present; expected folder {self.root / key}") from None

    def file(self, key: str, target: str) -> Path:
        """One file inside a present corpus folder."""
        return self.path(key) / target

    def describe(self) -> str:
        """One line per catalogue entry: present, incomplete (naming the files) or absent."""
        lines = []
        for spec in CATALOGUE:
            if spec.key in self.present:
                state = "present"
            elif spec.key in self.incomplete:
                lacking = self.incomplete[spec.key]
                shown = ", ".join(lacking[:3]) + (f" and {len(lacking) - 3} more" if len(lacking) > 3 else "")
                state = "incomplete, lacks " + shown
            else:
                state = "absent"
            lines.append(f"{spec.key:<26} {state}")
        return "\n".join(lines)


def default_corpora_root() -> Path:
    """The corpora root: the environment variable, else a corpora folder beside the package,
    else one under the application home."""
    override = os.environ.get(CORPORA_ENV)
    if override:
        return Path(override)
    beside = Path(__file__).resolve().parent.parent.parent / "corpora"
    if beside.is_dir():
        return beside
    return app_home() / "corpora"


def librispeech_utterances(folder: Path) -> list[Path]:
    """The utterance files of a LibriSpeech corpus folder whose chapter transcript is present."""
    utterances = folder / LIBRISPEECH_FOLDER
    if not utterances.is_dir():
        return []
    found = []
    for flac in sorted(utterances.glob("*.flac")):
        speaker, chapter = flac.stem.split("-")[:2]
        if (utterances / f"{speaker}-{chapter}.trans.txt").is_file():
            found.append(flac)
    return found


def find_corpora(root: str | os.PathLike[str] | None = None) -> CorpusStore:
    """Read a corpora root and report which catalogue corpora are complete there."""
    base = Path(root) if root is not None else default_corpora_root()
    present: dict[str, Path] = {}
    incomplete: dict[str, tuple[str, ...]] = {}
    for spec in CATALOGUE:
        folder = base / spec.key
        if not folder.is_dir():
            continue
        if spec.selection is not None:
            if librispeech_utterances(folder):
                present[spec.key] = folder
            else:
                incomplete[spec.key] = (f"{LIBRISPEECH_FOLDER}/*.flac with transcripts",)
            continue
        missing = tuple(file.target for file in spec.files if not (folder / file.target).is_file())
        if missing:
            incomplete[spec.key] = missing
        else:
            present[spec.key] = folder
    return CorpusStore(root=base, present=present, incomplete=incomplete)


def lock_path(root: str | os.PathLike[str]) -> Path:
    return Path(root) / LOCK_FILE


def read_lock(root: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Lock entries keyed "<corpus key>/<target>"; empty when the lock is absent or unreadable."""
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


def lock_entry(spec: CorpusSpec, file: CorpusFile, path: Path, revision: str | None = None) -> dict[str, Any]:
    """The lock record for one fetched file: digest, size, source, revision, licence, time."""
    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "url": file.url,
        "archive": file.archive,
        "member": file.member,
        "revision": revision if revision is not None else spec.revision,
        "licence": spec.licence,
        "fetched_utc": utc_now(),
    }


@dataclass(frozen=True)
class StoreCheck:
    """Verdict for one file of one corpus."""

    key: str
    file: str
    status: str
    detail: str


def _files_to_check(spec: CorpusSpec, folder: Path, lock: dict[str, dict[str, Any]]) -> list[str]:
    if spec.selection is None:
        return [file.target for file in spec.files]
    prefix = spec.key + "/"
    pinned = [key[len(prefix):] for key in lock if key.startswith(prefix)]
    present = []
    utterances = folder / LIBRISPEECH_FOLDER
    if utterances.is_dir():
        present = [f"{LIBRISPEECH_FOLDER}/{p.name}" for p in sorted(utterances.iterdir()) if p.is_file()]
    return sorted(set(pinned) | set(present))


def verify_store(root: str | os.PathLike[str] | None = None, keys: tuple[str, ...] | None = None) -> list[StoreCheck]:
    """Digest every file of the chosen corpora against the lock.

    Statuses: verified (the digest matches the lock), mismatch (it does not), unpinned
    (present but never recorded in the lock) and missing.
    """
    store = find_corpora(root)
    lock = read_lock(store.root)
    checks: list[StoreCheck] = []
    for spec in CATALOGUE:
        if keys is not None and spec.key not in keys:
            continue
        folder = store.root / spec.key
        for target in _files_to_check(spec, folder, lock):
            path = folder / target
            if not path.is_file():
                checks.append(StoreCheck(spec.key, target, STATUS_MISSING, f"expected at {path}"))
                continue
            entry = lock.get(f"{spec.key}/{target}")
            if entry is None:
                checks.append(StoreCheck(spec.key, target, STATUS_UNPINNED, "present, no digest recorded"))
                continue
            digest = sha256_file(path)
            if digest == entry.get("sha256"):
                checks.append(StoreCheck(spec.key, target, STATUS_VERIFIED, digest[:16]))
            else:
                checks.append(StoreCheck(spec.key, target, STATUS_MISMATCH, f"have {digest[:16]}, lock {str(entry.get('sha256'))[:16]}"))
    return checks
