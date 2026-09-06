"""Colour theme for the twinscribe screens.

Fusion style with a hand-set palette in a light and a dark variant. One accent colour is
used only to show state (the current selection, marks still to review), a success colour
marks what has been resolved, and a warning colour flags the test-only panel when words
were really spoken in a span. No web fonts, icon fonts or external assets of any kind.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Theme:
    """State colours that widgets read directly rather than from the palette.

    `accent` is for unresolved marks and the current selection, `success` for resolved
    marks, `warning` for the test-only panel when reference words fall in a span, `track`
    for the timeline background and `outline` for the playhead and the current-mark frame.
    """

    dark: bool
    accent: QColor
    success: QColor
    warning: QColor
    track: QColor
    outline: QColor


_LIGHT = {
    "window": "#f2f2f2",
    "window_text": "#1c1c1c",
    "base": "#ffffff",
    "alternate_base": "#f7f7f7",
    "text": "#1c1c1c",
    "button": "#e6e6e6",
    "button_text": "#1c1c1c",
    "bright_text": "#ffffff",
    "tooltip_base": "#ffffdc",
    "tooltip_text": "#1c1c1c",
    "placeholder": "#8c8c8c",
    "disabled_text": "#9a9a9a",
    "light": "#ffffff",
    "midlight": "#e0e0e0",
    "mid": "#c8c8c8",
    "dark": "#a0a0a0",
    "shadow": "#707070",
    "accent": "#2b63c9",
    "highlighted_text": "#ffffff",
    "success": "#2a8a4f",
    "warning": "#b86e00",
    "track": "#d9d9d9",
    "outline": "#1c1c1c",
}

_DARK = {
    "window": "#2d2d30",
    "window_text": "#e8e8e8",
    "base": "#1e1e1e",
    "alternate_base": "#262629",
    "text": "#e8e8e8",
    "button": "#3a3a3d",
    "button_text": "#e8e8e8",
    "bright_text": "#ffffff",
    "tooltip_base": "#3a3a3d",
    "tooltip_text": "#e8e8e8",
    "placeholder": "#8a8a8a",
    "disabled_text": "#7a7a7a",
    "light": "#505055",
    "midlight": "#48484c",
    "mid": "#3c3c40",
    "dark": "#151515",
    "shadow": "#000000",
    "accent": "#4f8ef7",
    "highlighted_text": "#101010",
    "success": "#4cc38a",
    "warning": "#f2b13d",
    "track": "#3c3c40",
    "outline": "#e8e8e8",
}


def _table(dark: bool) -> dict[str, str]:
    return _DARK if dark else _LIGHT


def theme_for(dark: bool = False) -> Theme:
    """Return the state colours for the light or dark variant without touching the app."""
    table = _table(dark)
    return Theme(
        dark=dark,
        accent=QColor(table["accent"]),
        success=QColor(table["success"]),
        warning=QColor(table["warning"]),
        track=QColor(table["track"]),
        outline=QColor(table["outline"]),
    )


def palette_for(dark: bool = False) -> QPalette:
    """Build the hand-set QPalette for the light or dark variant."""
    table = _table(dark)
    palette = QPalette()
    role = QPalette.ColorRole
    pairs = (
        (role.Window, "window"),
        (role.WindowText, "window_text"),
        (role.Base, "base"),
        (role.AlternateBase, "alternate_base"),
        (role.Text, "text"),
        (role.Button, "button"),
        (role.ButtonText, "button_text"),
        (role.BrightText, "bright_text"),
        (role.ToolTipBase, "tooltip_base"),
        (role.ToolTipText, "tooltip_text"),
        (role.PlaceholderText, "placeholder"),
        (role.Light, "light"),
        (role.Midlight, "midlight"),
        (role.Mid, "mid"),
        (role.Dark, "dark"),
        (role.Shadow, "shadow"),
        (role.Highlight, "accent"),
        (role.HighlightedText, "highlighted_text"),
        (role.Link, "accent"),
    )
    for colour_role, key in pairs:
        palette.setColor(colour_role, QColor(table[key]))
    disabled = QPalette.ColorGroup.Disabled
    disabled_text = QColor(table["disabled_text"])
    for colour_role in (role.Text, role.WindowText, role.ButtonText):
        palette.setColor(disabled, colour_role, disabled_text)
    palette.setColor(disabled, role.Highlight, QColor(table["mid"]))
    return palette


def apply_theme(app: QApplication, dark: bool = False) -> Theme:
    """Select the Fusion style and the palette on `app`; return the state colours."""
    app.setStyle("Fusion")
    app.setPalette(palette_for(dark))
    return theme_for(dark)
