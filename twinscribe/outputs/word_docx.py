"""Word document rendering with the standard library only.

A .docx file is a zip of XML parts. This module writes the handful of parts a transcript
needs: content types, package relationships, core and application properties, styles and the
document body. The body carries the file name as a title, the engine facts, a speaker table,
the transcript as one paragraph per line with a hanging indent so that wrapped text aligns,
and a closing note on the review list. The document properties name the author the caller
gives and this application; no other tool is involved.
"""

from __future__ import annotations

import os
import re
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from xml.sax.saxutils import escape

from twinscribe import __version__
from twinscribe.labelling import UNLABELLED_NAME
from twinscribe.outputs.plain_text import APPROXIMATE_NOTE, DRAFT_NOTICE, set_aside_note, transcript_entries
from twinscribe.outputs.transcript_doc import (
    approximate_word_times,
    clock,
    non_speech_summary,
    scene_phrase,
    speaker_names,
)

APPLICATION_NAME = "twinscribe"
MUTED = "7F7F7F"
SPEAKER_COLOURS: tuple[str, ...] = (
    "1F6F8B",
    "8B3A62",
    "B4501F",
    "4B7A1F",
    "3F4CB0",
    "8A6A1F",
    "1F7A6E",
    "6B4C9A",
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
    '<Override PartName="/docProps/core.xml" '
    'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
    '<Override PartName="/docProps/app.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
    "</Types>"
)

PACKAGE_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="word/document.xml"/>'
    '<Relationship Id="rId2" '
    'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
    'Target="docProps/core.xml"/>'
    '<Relationship Id="rId3" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
    'Target="docProps/app.xml"/>'
    "</Relationships>"
)

DOCUMENT_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    "</Relationships>"
)

STYLES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    f'<w:styles xmlns:w="{_W}">'
    "<w:docDefaults><w:rPrDefault><w:rPr>"
    '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/><w:sz w:val="22"/><w:szCs w:val="22"/>'
    '<w:lang w:val="en-CA"/>'
    "</w:rPr></w:rPrDefault>"
    '<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault>'
    "</w:docDefaults>"
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/>'
    '<w:pPr><w:spacing w:after="60"/></w:pPr><w:rPr><w:b/><w:sz w:val="40"/><w:szCs w:val="40"/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>'
    '<w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="360" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr>'
    '<w:rPr><w:b/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="Meta"><w:name w:val="Meta"/><w:basedOn w:val="Normal"/>'
    f'<w:pPr><w:spacing w:after="40"/></w:pPr><w:rPr><w:color w:val="{MUTED}"/><w:sz w:val="18"/>'
    '<w:szCs w:val="18"/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="TranscriptLine"><w:name w:val="Transcript Line"/>'
    '<w:basedOn w:val="Normal"/>'
    '<w:pPr><w:tabs><w:tab w:val="left" w:pos="1080"/><w:tab w:val="left" w:pos="3060"/></w:tabs>'
    '<w:spacing w:after="80"/><w:ind w:left="3060" w:hanging="3060"/></w:pPr></w:style>'
    '<w:style w:type="table" w:default="1" w:styleId="TableNormal"><w:name w:val="Normal Table"/></w:style>'
    '<w:style w:type="table" w:styleId="SpeakerTable"><w:name w:val="Speaker Table"/>'
    '<w:basedOn w:val="TableNormal"/>'
    "<w:tblPr><w:tblBorders>"
    '<w:top w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
    '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
    '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/>'
    "</w:tblBorders></w:tblPr></w:style>"
    "</w:styles>"
)


def _clean(text: object) -> str:
    return escape(_CONTROL_CHARS.sub("", str(text)))


def _run(
    text: object,
    bold: bool = False,
    colour: str | None = None,
    size_half_points: int | None = None,
    italic: bool = False,
) -> str:
    properties = ""
    if bold:
        properties += "<w:b/>"
    if italic:
        properties += "<w:i/>"
    if colour:
        properties += f'<w:color w:val="{colour}"/>'
    if size_half_points:
        properties += f'<w:sz w:val="{size_half_points}"/><w:szCs w:val="{size_half_points}"/>'
    rpr = f"<w:rPr>{properties}</w:rPr>" if properties else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{_clean(text)}</w:t></w:r>'


def _tab() -> str:
    return "<w:r><w:tab/></w:r>"


def _paragraph(runs: str, style: str | None = None) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}{runs}</w:p>"


def _cell(text: object, width: int, bold: bool = False, right: bool = False, colour: str | None = None) -> str:
    jc = '<w:jc w:val="right"/>' if right else ""
    ppr = f'<w:pPr><w:spacing w:after="0"/>{jc}</w:pPr>'
    return (
        f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/></w:tcPr>'
        f"<w:p>{ppr}{_run(text, bold=bold, colour=colour)}</w:p></w:tc>"
    )


def _speaker_table(doc: Mapping[str, Any]) -> str:
    names = speaker_names(doc)
    widths = (4400, 2200, 2200)
    rows = [
        "<w:tr>"
        + _cell("Speaker", widths[0], bold=True)
        + _cell("Words", widths[1], bold=True, right=True)
        + _cell("Speaking time", widths[2], bold=True, right=True)
        + "</w:tr>"
    ]
    colour_index = 0
    for entry in doc.get("speakers", []):
        label = entry.get("label")
        if label is not None:
            name = names.get(label, label)
            colour: str | None = SPEAKER_COLOURS[colour_index % len(SPEAKER_COLOURS)]
            colour_index += 1
        else:
            name = UNLABELLED_NAME
            colour = MUTED
        rows.append(
            "<w:tr>"
            + _cell(name, widths[0], bold=True, colour=colour)
            + _cell(f"{int(entry.get('words', 0)):,}", widths[1], right=True)
            + _cell(clock(float(entry.get("seconds", 0.0))), widths[2], right=True)
            + "</w:tr>"
        )
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    return (
        '<w:tbl><w:tblPr><w:tblStyle w:val="SpeakerTable"/><w:tblW w:w="0" w:type="auto"/></w:tblPr>'
        f"<w:tblGrid>{grid}</w:tblGrid>{''.join(rows)}</w:tbl>"
    )


def _engine_text(facts: Mapping[str, Any] | None) -> str:
    if not facts:
        return "none"
    text = " ".join(str(facts.get(k) or "") for k in ("engine", "model")).strip()
    preset = facts.get("preset")
    return f"{text} ({preset})" if preset else text


def document_body(doc: Mapping[str, Any]) -> str:
    """The body XML of the document part."""
    source = doc.get("source", {})
    engines = doc.get("engines", {})
    names = speaker_names(doc)
    labelled = [s for s in doc.get("speakers", []) if s.get("label") is not None]
    colours = {
        entry["label"]: SPEAKER_COLOURS[index % len(SPEAKER_COLOURS)] for index, entry in enumerate(labelled)
    }
    noun = "speaker" if len(labelled) == 1 else "speakers"
    review = doc.get("review", {})
    marks = int(review.get("marks", 0))
    stem = str(source.get("outputs") or str(source.get("name", "")).rsplit(".", 1)[0])

    parts = [
        _paragraph(_run(source.get("name", "")), "Title"),
        _paragraph(
            _run(
                f"Duration {clock(float(doc.get('duration_s', 0.0)), tenths=False)}   |   "
                f"{len(labelled)} {noun}   |   produced {doc.get('produced_utc', '')}   |   "
                f"{APPLICATION_NAME} {doc.get('version', '')}, quality level {doc.get('profile', '')}"
            ),
            "Meta",
        ),
        _paragraph(_run(f"Published engine: {_engine_text(engines.get('publisher'))}"), "Meta"),
        _paragraph(
            _run(
                f"Checked against: {_engine_text(engines.get('detector'))}; its text is never published"
                + (f"; {APPROXIMATE_NOTE}" if approximate_word_times(engines.get("detector")) else "")
            ),
            "Meta",
        ),
    ]
    if doc.get("speaker_failure"):
        parts.append(_paragraph(_run(f"Speaker labelling did not complete: {doc['speaker_failure']}"), "Meta"))
    parts.append(_paragraph(_run("Speakers"), "Heading1"))
    parts.append(_speaker_table(doc))
    if marks:
        share = 100.0 * float(review.get("fraction", 0.0))
        span_noun = "span" if marks == 1 else "spans"
        review_text = (
            f"Review list: {marks} {span_noun} where speech may be missing, {share:.1f}% of the recording; "
            f"see {stem}.review.json. "
        )
    else:
        review_text = "Review list: no span where speech may be missing was found. "
    parts.append(_paragraph(_run(review_text + DRAFT_NOTICE), "Meta"))
    summary = non_speech_summary(doc)
    if summary:
        parts.append(_paragraph(_run(f"Without speech: {summary}; marked in the transcript."), "Meta"))
    note = set_aside_note(doc)
    if note:
        parts.append(_paragraph(_run(note), "Meta"))
    parts.append(_paragraph(_run("Transcript"), "Heading1"))
    for start, kind, entry in transcript_entries(doc):
        if kind == "scene":
            runs = (
                _run(clock(start), colour=MUTED, size_half_points=18)
                + _tab()
                + _run(f"({scene_phrase(entry)})", colour=MUTED, italic=True)
            )
            parts.append(_paragraph(runs, "TranscriptLine"))
            continue
        label = entry.get("speaker")
        if label is not None:
            name = names.get(label, label)
            colour = colours.get(label, MUTED)
        else:
            name = UNLABELLED_NAME
            colour = MUTED
        runs = (
            _run(clock(start), colour=MUTED, size_half_points=18)
            + _tab()
            + _run(name, bold=True, colour=colour)
            + _tab()
            + _run(entry.get("text", ""))
        )
        parts.append(_paragraph(runs, "TranscriptLine"))
    if not doc.get("lines"):
        parts.append(_paragraph(_run("No words were published for this recording."), "Meta"))
    parts.append(
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1296" w:bottom="1440" w:left="1296" '
        'w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr>"
    )
    return "".join(parts)


def document_xml(doc: Mapping[str, Any]) -> str:
    """The complete document part."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<w:document xmlns:w="{_W}"><w:body>{document_body(doc)}</w:body></w:document>'
    )


def _w3c_time(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def core_properties_xml(title: str, author: str, moment: datetime | None = None) -> str:
    """docProps/core.xml with the title, the author and the time."""
    when = _w3c_time(moment or datetime.now(timezone.utc))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{_clean(title)}</dc:title>"
        f"<dc:creator>{_clean(author)}</dc:creator>"
        f"<cp:lastModifiedBy>{_clean(author)}</cp:lastModifiedBy>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{when}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{when}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def app_properties_xml() -> str:
    """docProps/app.xml naming this application."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        f"<Application>{APPLICATION_NAME} {__version__}</Application>"
        "</Properties>"
    )


def write_docx(doc: Mapping[str, Any], path: str | os.PathLike[str], author: str = "") -> None:
    """Write the Word document for a transcript document.

    The author goes into the core properties as creator and last modifier; when empty, the
    application name is used so that the field is never blank.
    """
    title = str(doc.get("source", {}).get("name", "transcript"))
    creator = " ".join(str(author).split()) or APPLICATION_NAME
    target = os.fspath(path)
    temporary = target + ".tmp"
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", CONTENT_TYPES)
            archive.writestr("_rels/.rels", PACKAGE_RELS)
            archive.writestr("docProps/core.xml", core_properties_xml(title, creator))
            archive.writestr("docProps/app.xml", app_properties_xml())
            archive.writestr("word/_rels/document.xml.rels", DOCUMENT_RELS)
            archive.writestr("word/styles.xml", STYLES)
            archive.writestr("word/document.xml", document_xml(doc))
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
