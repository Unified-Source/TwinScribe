"""Non-speech scenes: silence, music, background noise and other sound between and inside
the published engine's utterances.

The published engine decodes only the utterances the voice detector finds, so it cannot write
into a pause; but the voice detector passes music and some noise as speech, and the second
engine, whose text is never shown, writes freely inside both, which put review marks on music
and on noise. The pauses between utterances, and the utterances themselves, are therefore
classified: by level first (a quiet stretch is silence whatever a model says), then by an
audio tagging model when one is present. A scene is written into the transcript as a marker
naming what is there instead of speech; the detector's words inside a scene are set aside
before the review list is built; and an utterance the tagger is confident holds no speech has
its words set aside too, recorded in the run record rather than dropped without trace. A
voiced utterance is never set aside on level alone, because quiet speech is speech.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from twinscribe.audio import SAMPLE_RATE
from twinscribe.engines.base import Segment, Transcript, Word

KIND_SILENCE = "silence"
KIND_MUSIC = "music"
KIND_NOISE = "noise"
KIND_SOUND = "sound"
KINDS: tuple[str, ...] = (KIND_SILENCE, KIND_MUSIC, KIND_NOISE, KIND_SOUND)

# A pause shorter than this is part of the conversation, not a scene.
MIN_GAP_S = 2.0
# Regions are tagged in pieces no longer than this; the tagging model was trained on clips of
# this length and a long pause may hold more than one kind of sound.
WINDOW_S = 10.0
# An utterance shorter than this is never tagged: a single short word gives the tagger too
# little to go on (a one-word utterance was measured as music).
MIN_UTTERANCE_S = 1.5
# Below this level a stretch is silence whatever the tagger says.
SILENCE_DBFS = -50.0
# Speech probability at or above which a region is speech and yields no scene.
SPEECH_KEEP = 0.3
# An utterance's words are set aside only below this speech probability, and only with a
# confident non-speech class beside it.
SPEECH_SUPPRESS = 0.15
CONFIDENT = 0.5
MUSIC_MIN = 0.35
NOISE_MIN = 0.2
# Scenes of one kind closer than this are one scene.
JOIN_S = 0.05
QUIET_FLOOR_DBFS = -120.0

SPEECH_CLASSES: frozenset[str] = frozenset(
    {
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Child speech, kid speaking",
        "Conversation",
        "Narration, monologue",
        "Speech synthesizer",
        "Shout",
        "Bellow",
        "Whoop",
        "Yell",
        "Children shouting",
        "Screaming",
        "Whispering",
        "Babbling",
    }
)
MUSIC_CLASSES: frozenset[str] = frozenset(
    {
        "Musical instrument",
        "Singing",
        "Choir",
        "A capella",
        "Synthesizer",
        "Piano",
        "Keyboard (musical)",
        "Electric piano",
        "Organ",
        "Guitar",
        "Electric guitar",
        "Bass guitar",
        "Acoustic guitar",
        "Plucked string instrument",
        "Drum",
        "Drum kit",
        "Percussion",
        "Violin, fiddle",
        "Cello",
        "Trumpet",
        "Saxophone",
        "Flute",
        "Harp",
        "Rapping",
        "Humming",
        "Yodeling",
        "Chant",
        "Mantra",
        "Beatboxing",
        "Song",
        "Melody",
        "Strum",
        "Orchestra",
        "Bowed string instrument",
        "Brass instrument",
        "Wind instrument, woodwind instrument",
        "Accordion",
        "Harmonica",
        "Bagpipes",
        "Ukulele",
        "Banjo",
        "Sitar",
        "Mandolin",
        "Zither",
    }
)
NOISE_CLASSES: frozenset[str] = frozenset(
    {
        "Noise",
        "White noise",
        "Pink noise",
        "Static",
        "Hum",
        "Mains hum",
        "Environmental noise",
        "Rustle",
        "Hiss",
        "Wind",
        "Wind noise (microphone)",
        "Rustling leaves",
        "Rain",
        "Rain on surface",
        "Raindrop",
        "Thunderstorm",
        "Thunder",
        "Water",
        "Stream",
        "Waterfall",
        "Ocean",
        "Waves, surf",
        "Gurgling",
        "Vehicle",
        "Car",
        "Traffic noise, roadway noise",
        "Air conditioning",
        "Mechanical fan",
        "Engine",
        "Idling",
        "Inside, small room",
        "Inside, large room or hall",
        "Inside, public space",
        "Outside, urban or manmade",
        "Outside, rural or natural",
        "Crowd",
        "Chatter",
        "Hubbub, speech noise, speech babble",
        "Fire",
        "Crackle",
        "Buzz",
        "Sine wave",
        "Silence",
    }
)


@dataclass(frozen=True)
class Event:
    """One sound class the tagger heard in a region, with its probability."""

    name: str
    prob: float


@dataclass(frozen=True)
class Scene:
    """One stretch without speech: its span, its kind, the tagger's top class and confidence."""

    start: float
    end: float
    kind: str
    label: str = ""
    probability: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class Analysis:
    """What the scene pass found: the scenes, the utterances set aside, and how it was done."""

    scenes: tuple[Scene, ...]
    suppressed: tuple[Segment, ...]
    tagged: bool
    regions_tagged: int
    settings: dict[str, float | int | bool]


TagFn = Callable[[Sequence[tuple[float, float]]], Sequence[Sequence[Event]]]


def merge_spans(spans: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of spans as sorted, disjoint spans; touching spans merge."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def gaps_between(spans: Iterable[tuple[float, float]], audio_s: float, min_gap_s: float = MIN_GAP_S) -> list[tuple[float, float]]:
    """The stretches of [0, audio_s] outside the spans that are at least min_gap_s long."""
    if min_gap_s < 0.0:
        raise ValueError("min_gap_s must not be negative")
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merge_spans(spans):
        if start - cursor >= min_gap_s:
            gaps.append((cursor, min(start, audio_s)))
        cursor = max(cursor, end)
    if audio_s - cursor >= min_gap_s:
        gaps.append((cursor, audio_s))
    return [(s, e) for s, e in gaps if e - s >= min_gap_s]


def split_windows(start: float, end: float, window_s: float = WINDOW_S) -> list[tuple[float, float]]:
    """Cut a span into equal pieces no longer than window_s."""
    if window_s <= 0.0:
        raise ValueError("window_s must be positive")
    length = max(0.0, end - start)
    if length == 0.0:
        return []
    count = max(1, int(math.ceil(length / window_s - 1e-9)))
    step = length / count
    return [(start + index * step, start + (index + 1) * step if index < count - 1 else end) for index in range(count)]


def level_dbfs(samples: np.ndarray, start: float, end: float, rate: int = SAMPLE_RATE) -> float:
    """Root-mean-square level of a stretch in dB relative to full scale; very low for nothing."""
    lo = max(0, int(round(start * rate)))
    hi = min(len(samples), int(round(end * rate)))
    if hi <= lo:
        return QUIET_FLOOR_DBFS
    piece = np.asarray(samples[lo:hi], dtype=np.float64)
    rms = float(np.sqrt(np.mean(piece * piece)))
    if rms <= 0.0:
        return QUIET_FLOOR_DBFS
    return max(QUIET_FLOOR_DBFS, 20.0 * math.log10(rms))


def family_probability(events: Iterable[Event], names: Iterable[str]) -> float:
    """The largest probability among the events whose class is in names; 0.0 when none is."""
    wanted = set(names)
    return max((float(e.prob) for e in events if e.name in wanted), default=0.0)


def is_music_class(name: str) -> bool:
    return "music" in name.lower() or name in MUSIC_CLASSES


def speech_probability(events: Iterable[Event]) -> float:
    return family_probability(events, SPEECH_CLASSES)


def music_probability(events: Iterable[Event]) -> float:
    return max((float(e.prob) for e in events if is_music_class(e.name)), default=0.0)


def noise_probability(events: Iterable[Event]) -> float:
    return family_probability(events, NOISE_CLASSES)


def _top(events: Sequence[Event], predicate: Callable[[Event], bool]) -> Event | None:
    chosen = [e for e in events if predicate(e)]
    return max(chosen, key=lambda e: e.prob) if chosen else None


def classify(level: float, events: Sequence[Event] | None) -> tuple[str | None, str, float]:
    """The kind of a stretch without published words, from its level and the tagger's events.

    Returns (kind, label, probability); kind is None when the stretch holds speech after all
    (the review list, not a scene, is the right place for it). Below the silence level the
    stretch is silence whatever the tagger says. Without a tagger a stretch above that level
    is sound of an unknown kind.
    """
    if level < SILENCE_DBFS:
        probability = family_probability(events, {"Silence"}) if events else 1.0
        return KIND_SILENCE, "", probability
    if events is None:
        return KIND_SOUND, "", 0.0
    speech = speech_probability(events)
    if speech >= SPEECH_KEEP:
        return None, "", speech
    music = music_probability(events)
    noise = noise_probability(events)
    if music >= MUSIC_MIN and music >= noise:
        top = _top(events, lambda e: is_music_class(e.name))
        return KIND_MUSIC, top.name if top is not None and top.name != "Music" else "", music
    if noise >= NOISE_MIN:
        top = _top(events, lambda e: e.name in NOISE_CLASSES)
        return KIND_NOISE, top.name if top is not None and top.name not in ("Noise", "Silence") else "", noise
    top = max(events, key=lambda e: e.prob) if events else None
    if top is None or top.prob < NOISE_MIN:
        return KIND_SOUND, "", top.prob if top is not None else 0.0
    return KIND_SOUND, top.name, top.prob


def should_suppress(events: Sequence[Event] | None, duration_s: float) -> bool:
    """Whether a voiced utterance's words are set aside: long enough to judge, hardly any
    speech in it, and a confident non-speech class. Never on level alone."""
    if events is None or duration_s < MIN_UTTERANCE_S:
        return False
    if speech_probability(events) >= SPEECH_SUPPRESS:
        return False
    return music_probability(events) >= CONFIDENT or noise_probability(events) >= CONFIDENT


def merge_scenes(scenes: Iterable[Scene], join_s: float = JOIN_S) -> list[Scene]:
    """Adjacent scenes of one kind become one; the label of the longer part is kept."""
    merged: list[Scene] = []
    for scene in sorted(scenes, key=lambda s: (s.start, s.end)):
        if merged and merged[-1].kind == scene.kind and scene.start <= merged[-1].end + join_s:
            last = merged[-1]
            label = last.label if last.duration >= scene.duration else scene.label
            merged[-1] = Scene(
                start=last.start,
                end=max(last.end, scene.end),
                kind=last.kind,
                label=label,
                probability=max(last.probability, scene.probability),
            )
        else:
            merged.append(scene)
    return merged


def words_outside(words: Iterable[Word], scenes: Sequence[Scene]) -> list[Word]:
    """The words whose middle does not fall inside any scene."""
    if not scenes:
        return list(words)
    spans = [(s.start, s.end) for s in scenes]
    kept: list[Word] = []
    for word in words:
        middle = 0.5 * (word.start + word.end)
        if not any(start <= middle < end for start, end in spans):
            kept.append(word)
    return kept


def without_segments(transcript: Transcript, segments: Iterable[Segment]) -> Transcript:
    """A copy of the transcript without the given segments (matched by span and text)."""
    removed = {(s.start, s.end, s.text) for s in segments}
    if not removed:
        return transcript
    kept = tuple(s for s in transcript.segments if (s.start, s.end, s.text) not in removed)
    return replace(transcript, segments=kept)


def seconds_by_kind(scenes: Iterable[Scene]) -> dict[str, float]:
    """Total seconds per kind, in the fixed order of kinds, omitting kinds with none."""
    totals: dict[str, float] = {}
    for scene in scenes:
        totals[scene.kind] = totals.get(scene.kind, 0.0) + scene.duration
    return {kind: totals[kind] for kind in KINDS if kind in totals}


def settings_record() -> dict[str, float | int | bool]:
    return {
        "min_gap_s": MIN_GAP_S,
        "window_s": WINDOW_S,
        "min_utterance_s": MIN_UTTERANCE_S,
        "silence_dbfs": SILENCE_DBFS,
        "speech_keep": SPEECH_KEEP,
        "speech_suppress": SPEECH_SUPPRESS,
        "confident": CONFIDENT,
        "music_min": MUSIC_MIN,
        "noise_min": NOISE_MIN,
    }


def analyse(
    samples: np.ndarray,
    utterances: Sequence[Segment],
    audio_s: float,
    tag: TagFn | None = None,
) -> Analysis:
    """Find the scenes of one recording.

    The pauses between the utterances (at least MIN_GAP_S long) are cut into windows and
    classified; the utterances at least MIN_UTTERANCE_S long are tagged as well, and one the
    tagger is confident holds no speech becomes a scene with its words set aside. tag, when
    given, receives every region in one call and returns the tagger's events per region in the
    same order; without it, only the level decides, and a loud pause is sound of an unknown
    kind.
    """
    spans = merge_spans((u.start, u.end) for u in utterances)
    gap_windows = [window for gap in gaps_between(spans, audio_s) for window in split_windows(*gap)]
    candidates = [u for u in utterances if u.end - u.start >= MIN_UTTERANCE_S]
    regions = gap_windows + [(u.start, u.end) for u in candidates]
    events_per_region: Sequence[Sequence[Event]] | None = None
    if tag is not None and regions:
        events_per_region = tag(regions)
        if len(events_per_region) != len(regions):
            raise ValueError(f"the tagger returned {len(events_per_region)} results for {len(regions)} regions")

    scenes: list[Scene] = []
    for index, (start, end) in enumerate(gap_windows):
        events = tuple(events_per_region[index]) if events_per_region is not None else None
        kind, label, probability = classify(level_dbfs(samples, start, end), events)
        if kind is not None:
            scenes.append(Scene(start, end, kind, label, probability))

    suppressed: list[Segment] = []
    for offset, utterance in enumerate(candidates):
        if events_per_region is None:
            break
        events = tuple(events_per_region[len(gap_windows) + offset])
        if not should_suppress(events, utterance.end - utterance.start):
            continue
        kind, label, probability = classify(level_dbfs(samples, utterance.start, utterance.end), events)
        if kind is None:
            kind = KIND_SOUND
        scenes.append(Scene(utterance.start, utterance.end, kind, label, probability))
        if utterance.words:
            suppressed.append(utterance)

    return Analysis(
        scenes=tuple(merge_scenes(scenes)),
        suppressed=tuple(suppressed),
        tagged=events_per_region is not None,
        regions_tagged=len(regions) if events_per_region is not None else 0,
        settings=settings_record(),
    )
