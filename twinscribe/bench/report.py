"""The bench report: accuracy pooled per corpus group and per variant, per-item detail, speaker
labelling, the review list, and speed with every contended row excluded and counted. Rendered
from the results document alone, so a report can be re-rendered without the engines.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from twinscribe.bench.scoring import pooled_word_errors

KIND_TRANSCRIPT = "transcript"
KIND_DIARIZATION = "diarization"
KIND_REVIEW = "review"

GROUPS: tuple[tuple[str, str], ...] = (
    ("hvb/", "telephone calls (HarperValleyBank)"),
    ("ami/ES2002a/headset/", "meeting, headset mix (AMI ES2002a)"),
    ("ami/ES2002a/sdm/", "meeting, far-field microphone (AMI ES2002a)"),
    ("librispeech/", "read speech (LibriSpeech test-other)"),
)

LIMITS = (
    "The corpora establish how the engines differ from one another and how they fail, not "
    "accuracy on any particular organisation's audio: the calls are role-played, the meeting "
    "participants are recruited volunteers, the read speech is volunteers reading novels, and "
    "none was recorded under stressful conditions or documents Canadian English. A ranking "
    "quoted from under about two thousand reference words is not a ranking."
)


def _pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{100.0 * value:.{digits}f}"


def _num(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def group_of(item_id: str) -> str:
    for prefix, title in GROUPS:
        if item_id.startswith(prefix):
            return title
    return "other"


def _rows(items: Mapping[str, Mapping[str, Any]], kind: str) -> Iterable[tuple[str, str, Mapping[str, Any]]]:
    for item_id, item in items.items():
        for variant, result in (item.get("variants") or {}).items():
            if result.get("kind") == kind and result.get("scores") is not None:
                yield item_id, variant, result


def accuracy_table(items: Mapping[str, Mapping[str, Any]]) -> str:
    """Pooled word error rate per corpus group and variant, normalised tokens first."""
    lines = [
        "| group | variant | items | ref words | WER norm % | sub | del | ins | WER raw % |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for item_id, variant, result in _rows(items, KIND_TRANSCRIPT):
        by_key.setdefault((group_of(item_id), variant), []).append(result["scores"])
    order = [title for _, title in GROUPS] + ["other"]
    for group in order:
        for (g, variant), scores in sorted(by_key.items(), key=lambda kv: kv[0][1]):
            if g != group:
                continue
            norm = pooled_word_errors(s["norm"] for s in scores)
            raw = pooled_word_errors(s["raw"] for s in scores)
            lines.append(
                f"| {group} | {variant} | {norm['items']} | {norm['n_ref']} | {_pct(norm['wer'])} "
                f"| {norm['sub']} | {norm['del']} | {norm['ins']} | {_pct(raw['wer'])} |"
            )
    return "\n".join(lines)


def item_table(items: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "| item | variant | audio s | ref words | WER norm % (sub/del/ins) | WER raw % | CER % | lowest speaker recall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item_id, variant, result in _rows(items, KIND_TRANSCRIPT):
        s = result["scores"]
        norm, raw = s["norm"], s["raw"]
        recalls = [(v["recall"], k) for k, v in (s.get("per_speaker") or {}).items() if v.get("recall") is not None]
        lowest = "-" if not recalls else f"{_pct(min(recalls)[0])} ({min(recalls)[1]})"
        lines.append(
            f"| {item_id} | {variant} | {_num(items[item_id].get('audio_s'), 0)} | {norm['n_ref']} "
            f"| {_pct(norm['wer'])} ({norm['sub']}/{norm['del']}/{norm['ins']}) | {_pct(raw['wer'])} | {_pct(norm['cer'])} | {lowest} |"
        )
    return "\n".join(lines)


def diarization_table(items: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "| item | variant | ref speakers | labels found | DER 0.25 % (miss/fa/conf) | DER 0 % | JER % | unmatched |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item_id, variant, result in _rows(items, KIND_DIARIZATION):
        s = result["scores"]
        d = s.get("der_0.25") or {}
        d0 = s.get("der_0") or {}
        unmatched = ", ".join(d.get("unmatched_ref") or []) or "none"
        lines.append(
            f"| {item_id} | {variant} | {len(items[item_id].get('speakers') or [])} | {s.get('labels_found', '-')} "
            f"| {_pct(d.get('der'))} ({_pct(d.get('miss'))}/{_pct(d.get('false_alarm'))}/{_pct(d.get('confusion'))}) "
            f"| {_pct(d0.get('der'))} | {_pct(s.get('jer'))} | {unmatched} |"
        )
    return "\n".join(lines)


def review_table(items: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "| item | marks | on speech | precision % | dropped words | covered | recall % | audio to review % | word times |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item_id, variant, result in _rows(items, KIND_REVIEW):
        s = result["scores"]
        times = "approximate" if s.get("approximate") else ("annotated" if s.get("evaluated") else "none")
        if not s.get("evaluated"):
            lines.append(f"| {item_id} | {s['marks']} | - | - | - | - | - | - | {times} |")
            continue
        lines.append(
            f"| {item_id} | {s['marks']} | {s['marks_on_speech']} | {_pct(s['precision'])} | {s['dropped_words']} "
            f"| {s['dropped_covered']} | {_pct(s['recall'])} | {_pct(s['audio_to_review_fraction'])} | {times} |"
        )
    return "\n".join(lines)


def speed_table(items: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    """Per-variant speed pooled over uncontended rows, and the per-row detail."""
    pooled: dict[str, dict[str, float]] = {}
    detail = [
        "| item | variant | audio s | load s | decode s | audio s per s | other load (cores) | contended |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item_id, item in items.items():
        for variant, result in (item.get("variants") or {}).items():
            timing = result.get("timing") or {}
            verdict = result.get("verdict") or {}
            audio_s = float(timing.get("audio_s") or 0.0)
            decode_s = float(timing.get("transcribe_s") or 0.0)
            busy = verdict.get("busy")
            contended = "yes" if busy else ("unknown" if busy is None else "no")
            rate = None if decode_s <= 0.0 else audio_s / decode_s
            detail.append(
                f"| {item_id} | {variant} | {_num(audio_s, 0)} | {_num(timing.get('load_s'))} | {_num(decode_s)} "
                f"| {_num(rate, 2)} | {_num(verdict.get('other_load_cores'), 2)} | {contended} |"
            )
            entry = pooled.setdefault(variant, {"audio_s": 0.0, "decode_s": 0.0, "rows": 0, "excluded": 0})
            if busy is False:
                entry["audio_s"] += audio_s
                entry["decode_s"] += decode_s
                entry["rows"] += 1
            else:
                entry["excluded"] += 1
    summary = [
        "| variant | uncontended rows | audio s | decode s | audio s per s | rows excluded (contended or unknown) |",
        "|---|---|---|---|---|---|",
    ]
    for variant, entry in sorted(pooled.items()):
        rate = None if entry["decode_s"] <= 0.0 else entry["audio_s"] / entry["decode_s"]
        summary.append(
            f"| {variant} | {int(entry['rows'])} | {_num(entry['audio_s'], 0)} | {_num(entry['decode_s'])} | {_num(rate, 2)} | {int(entry['excluded'])} |"
        )
    return "\n".join(summary), "\n".join(detail)


def render_report(results: Mapping[str, Any]) -> str:
    """The whole report as Markdown."""
    items: Mapping[str, Mapping[str, Any]] = results.get("items") or {}
    meta = results.get("meta") or {}
    machine = meta.get("machine") or {}
    versions = meta.get("versions") or {}
    plan = meta.get("plan") or {}
    lines = [
        "# Bench report",
        "",
        f"Produced {meta.get('produced_utc', '-')}. Items: {len(items)}. Threads per engine: {meta.get('threads', '-')}. "
        f"Device preference: {meta.get('device', '-')}.",
        "",
        f"Machine class: {machine.get('processor', '-')}, {machine.get('logical_cores', '-')} cores, "
        f"{machine.get('memory_gb', '-')} GB; {machine.get('os', '-')}.",
        "",
        "Libraries: " + (", ".join(f"{k} {v}" for k, v in sorted(versions.items())) or "-") + ".",
        "",
        "Plan: " + ("; ".join(str(line) for line in plan.get("lines", [])) or "-") + ".",
        "",
        "## Accuracy, pooled per corpus group",
        "",
        accuracy_table(items),
        "",
        "## Accuracy per item",
        "",
        item_table(items),
        "",
        "## Speaker labelling",
        "",
        diarization_table(items),
        "",
        "## Review list (published engine checked by the detector)",
        "",
        review_table(items),
        "",
        "## Speed",
        "",
        "Rows with other load above the threshold during the cell, or with no verdict, are excluded from the pooled figures and counted.",
        "",
    ]
    summary, detail = speed_table(items)
    lines += [summary, "", detail, "", "## Limits", "", LIMITS, ""]
    notes = sorted({note for item in items.values() for note in (item.get("notes") or [])})
    if notes:
        lines += ["## Item notes", ""] + [f"- {note}" for note in notes] + [""]
    credits = meta.get("credits") or []
    if credits:
        lines += ["## Corpora and licences", ""] + [f"- {credit}" for credit in credits] + [""]
    return "\n".join(lines)
