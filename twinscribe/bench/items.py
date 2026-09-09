"""Bench items: one recording and its reference per item, built from the corpora store into a
work folder as 16 kHz mono WAV. Item ids carry the upstream ids they are built from, so an
item is the same item on every machine that holds the same store.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from twinscribe import audio
from twinscribe.bench.corpora import (
    AMI_AGENTS,
    AMI_ARRAY,
    AMI_HEADSET,
    AMI_MEETING,
    AMI_MEETINGS,
    AMI_WINDOW,
    HVB_AGENT,
    HVB_CALLER,
    HVB_CALLS,
    HVB_TRANSCRIPT,
    KEY_AMI,
    KEY_HVB,
    KEY_LIBRISPEECH,
    LIBRISPEECH_FOLDER,
    CorpusStore,
)
from twinscribe.bench.references import (
    Reference,
    ami_reference,
    ami_segments,
    ami_speakers,
    ami_words,
    cut_window,
    hvb_alignment_notes,
    hvb_delays,
    hvb_reference,
    librispeech_reference,
    librispeech_transcripts,
    mix_channels,
    read_pcm16,
    write_pcm16,
)

KIND_HVB_CALL = "hvb-call"
KIND_AMI_MEETING = "ami-meeting"
KIND_LIBRISPEECH = "librispeech-utterance"

AMI_MICROPHONES: tuple[tuple[str, str], ...] = ((AMI_HEADSET, "headset"), (AMI_ARRAY, "sdm"))
ALIGNMENT_TOLERANCE_S = 0.05


@dataclass(frozen=True)
class ItemSpec:
    """How one item is built: its id, corpus, kind, the upstream source it names and, for a
    meeting, the window in seconds (None for the whole meeting)."""

    id: str
    corpus: str
    kind: str
    source: str
    window: tuple[float, float] | None = None


@dataclass(frozen=True)
class Item:
    """A built item: the 16 kHz WAV to run the engines on and the reference to score against."""

    spec: ItemSpec
    wav: Path
    reference: Reference


def safe_name(item_id: str) -> str:
    """A file-system name for an item id."""
    return item_id.replace("/", "__")


def list_items(store: CorpusStore) -> list[ItemSpec]:
    """Every item the store can build, in a fixed order: calls, meeting windows and whole
    meetings per microphone, utterances."""
    items: list[ItemSpec] = []
    if store.has(KEY_HVB):
        for sid in HVB_CALLS:
            if store.file(KEY_HVB, f"calls/{sid}/{HVB_TRANSCRIPT}").is_file():
                items.append(ItemSpec(id=f"hvb/{sid}", corpus=KEY_HVB, kind=KIND_HVB_CALL, source=sid))
    if store.has(KEY_AMI):
        tag = f"{int(AMI_WINDOW[0])}-{int(AMI_WINDOW[1])}"
        for file_name, microphone in AMI_MICROPHONES:
            items.append(ItemSpec(id=f"ami/{AMI_MEETING}/{microphone}/{tag}", corpus=KEY_AMI, kind=KIND_AMI_MEETING, source=file_name, window=AMI_WINDOW))
        for file_name, microphone in AMI_MICROPHONES:
            items.append(ItemSpec(id=f"ami/{AMI_MEETING}/{microphone}/full", corpus=KEY_AMI, kind=KIND_AMI_MEETING, source=file_name, window=None))
    if store.has(KEY_LIBRISPEECH):
        for flac in sorted(store.file(KEY_LIBRISPEECH, LIBRISPEECH_FOLDER).glob("*.flac")):
            items.append(ItemSpec(id=f"librispeech/{flac.stem}", corpus=KEY_LIBRISPEECH, kind=KIND_LIBRISPEECH, source=flac.name))
    return items


def _build_hvb(spec: ItemSpec, store: CorpusStore, work: Path, ffmpeg: str | os.PathLike[str] | None) -> Item:
    folder = store.file(KEY_HVB, f"calls/{spec.source}")
    with open(folder / HVB_TRANSCRIPT, "r", encoding="utf-8") as handle:
        transcript = json.load(handle)
    agent, agent_rate = read_pcm16(folder / HVB_AGENT)
    caller, caller_rate = read_pcm16(folder / HVB_CALLER)
    if agent_rate != caller_rate:
        raise ValueError(f"{spec.id}: channel rates differ ({agent_rate} and {caller_rate} Hz)")
    wav = work / f"{safe_name(spec.id)}.wav"
    if not wav.is_file():
        mixed = work / f"{safe_name(spec.id)}.mix{agent_rate}.wav"
        write_pcm16(mixed, mix_channels(agent, caller), agent_rate)
        try:
            audio.decode_to_wav(mixed, wav, ffmpeg)
        finally:
            mixed.unlink(missing_ok=True)
    reference = hvb_reference(spec.id, transcript, audio.duration_s(wav))
    channel_difference = (len(caller) - len(agent)) / float(agent_rate)
    notes = list(reference.notes) + hvb_alignment_notes(hvb_delays(transcript), channel_difference, ALIGNMENT_TOLERANCE_S)
    reference = Reference(
        item=reference.item, corpus=reference.corpus, audio_s=reference.audio_s, speakers=reference.speakers,
        segments=reference.segments, words=reference.words, word_times=reference.word_times, notes=tuple(notes),
    )
    return Item(spec=spec, wav=wav, reference=reference)


def _build_ami(spec: ItemSpec, store: CorpusStore, work: Path, ffmpeg: str | os.PathLike[str] | None) -> Item:
    decoded = work / f"ami__{Path(spec.source).stem}.16k.wav"
    if not decoded.is_file():
        audio.decode_to_wav(store.file(KEY_AMI, f"audio/{spec.source}"), decoded, ffmpeg)
    wav = work / f"{safe_name(spec.id)}.wav"
    if spec.window is None:
        wav = decoded
    elif not wav.is_file():
        samples, rate = read_pcm16(decoded)
        write_pcm16(wav, cut_window(samples, rate, spec.window), rate)
    annotations = store.file(KEY_AMI, "annotations")
    names = ami_speakers((annotations / AMI_MEETINGS).read_bytes(), AMI_MEETING)
    words_by: dict[str, list] = {}
    segments_by: dict[str, list] = {}
    for agent in AMI_AGENTS:
        speaker = names.get(agent, agent)
        words_by[speaker] = ami_words((annotations / "words" / f"{AMI_MEETING}.{agent}.words.xml").read_bytes(), speaker)
        segments_by[speaker] = ami_segments((annotations / "segments" / f"{AMI_MEETING}.{agent}.segments.xml").read_bytes(), speaker)
    reference = ami_reference(spec.id, words_by, segments_by, audio.duration_s(wav), spec.window)
    return Item(spec=spec, wav=wav, reference=reference)


def _build_librispeech(spec: ItemSpec, store: CorpusStore, work: Path, ffmpeg: str | os.PathLike[str] | None) -> Item:
    folder = store.file(KEY_LIBRISPEECH, LIBRISPEECH_FOLDER)
    utterance = Path(spec.source).stem
    speaker, chapter = utterance.split("-")[:2]
    wav = work / f"{safe_name(spec.id)}.wav"
    if not wav.is_file():
        audio.decode_to_wav(folder / spec.source, wav, ffmpeg)
    transcripts = librispeech_transcripts((folder / f"{speaker}-{chapter}.trans.txt").read_text(encoding="utf-8"))
    if utterance not in transcripts:
        raise KeyError(f"{spec.id}: no transcript line for {utterance}")
    reference = librispeech_reference(spec.id, utterance, transcripts[utterance], audio.duration_s(wav))
    return Item(spec=spec, wav=wav, reference=reference)


def build_item(spec: ItemSpec, store: CorpusStore, work: str | os.PathLike[str], ffmpeg: str | os.PathLike[str] | None = None) -> Item:
    """Build one item into the work folder; a WAV already there is reused."""
    folder = Path(work)
    folder.mkdir(parents=True, exist_ok=True)
    if spec.kind == KIND_HVB_CALL:
        return _build_hvb(spec, store, folder, ffmpeg)
    if spec.kind == KIND_AMI_MEETING:
        return _build_ami(spec, store, folder, ffmpeg)
    if spec.kind == KIND_LIBRISPEECH:
        return _build_librispeech(spec, store, folder, ffmpeg)
    raise KeyError(f"unknown item kind {spec.kind!r}")
