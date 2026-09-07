"""Colour theme for the twinscribe screens.

Fusion style with a hand-set palette in a light and a dark variant, and one stylesheet built
from the same table so that every control shares the same radii, borders and spacing. One
accent colour is used only to show state (the current selection, marks still to review, the
primary action), a success colour marks what has been resolved, a warning colour flags what
needs listening to, and a small set of speaker colours tells speakers apart. No web fonts,
icon fonts or external assets of any kind.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Theme:
    """State colours that widgets read directly rather than from the palette.

    `accent` is for unresolved marks, the current selection and the primary action, `success`
    for resolved marks and completed files, `warning` for spans that need listening to,
    `track` for the timeline background, `outline` for the playhead and the current-mark
    frame, `muted` for secondary text, and `speakers` tells speakers apart.
    """

    dark: bool
    accent: QColor
    success: QColor
    warning: QColor
    track: QColor
    outline: QColor
    muted: QColor
    speakers: tuple[QColor, ...]

    def speaker(self, index: int) -> QColor:
        """The colour for the speaker at `index` in order of first appearance."""
        return self.speakers[index % len(self.speakers)]


_LIGHT = {
    "window": "#f4f4f6",
    "window_text": "#1c1c1e",
    "base": "#ffffff",
    "alternate_base": "#f7f7f9",
    "text": "#1c1c1e",
    "button": "#ffffff",
    "button_text": "#1c1c1e",
    "bright_text": "#ffffff",
    "tooltip_base": "#1c1c1e",
    "tooltip_text": "#ffffff",
    "placeholder": "#7a7a80",
    "disabled_text": "#a8a8ad",
    "light": "#ffffff",
    "midlight": "#e6e6ea",
    "mid": "#cfcfd5",
    "dark": "#9a9aa0",
    "shadow": "#707076",
    "accent": "#2b63c9",
    "accent_hover": "#2455b0",
    "highlighted_text": "#ffffff",
    "success": "#2a8a4f",
    "warning": "#b86e00",
    "track": "#e3e3e8",
    "outline": "#1c1c1e",
}

_DARK = {
    "window": "#232326",
    "window_text": "#e8e8ea",
    "base": "#1a1a1d",
    "alternate_base": "#262629",
    "text": "#e8e8ea",
    "button": "#303034",
    "button_text": "#e8e8ea",
    "bright_text": "#ffffff",
    "tooltip_base": "#e8e8ea",
    "tooltip_text": "#1a1a1d",
    "placeholder": "#8d8d94",
    "disabled_text": "#6c6c72",
    "light": "#4a4a50",
    "midlight": "#333338",
    "mid": "#45454b",
    "dark": "#121214",
    "shadow": "#000000",
    "accent": "#4f8ef7",
    "accent_hover": "#6ba1f9",
    "highlighted_text": "#0f1218",
    "success": "#4cc38a",
    "warning": "#f2b13d",
    "track": "#2f2f34",
    "outline": "#e8e8ea",
}

_SPEAKERS_LIGHT: tuple[str, ...] = (
    "#1f6f8b",
    "#8b3a62",
    "#b4501f",
    "#4b7a1f",
    "#3f4cb0",
    "#8a6a1f",
    "#1f7a6e",
    "#6b4c9a",
)
_SPEAKERS_DARK: tuple[str, ...] = (
    "#5cc2e6",
    "#e08ab8",
    "#f0a070",
    "#a4d66b",
    "#9aa3f5",
    "#e6c46a",
    "#6fd6c4",
    "#c2a3f0",
)


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
        muted=QColor(table["placeholder"]),
        speakers=tuple(QColor(c) for c in (_SPEAKERS_DARK if dark else _SPEAKERS_LIGHT)),
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


def stylesheet(dark: bool = False) -> str:
    """One stylesheet for every control, built from the palette table."""
    t = _table(dark)
    return f"""
QWidget {{ font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 10pt; }}
QMainWindow, QDialog {{ background: {t['window']}; }}
QToolTip {{ background: {t['tooltip_base']}; color: {t['tooltip_text']}; border: none; padding: 5px 8px; }}
QPushButton {{ background: {t['button']}; color: {t['button_text']}; border: 1px solid {t['mid']};
    border-radius: 6px; padding: 5px 14px; min-height: 18px; }}
QPushButton:hover {{ background: {t['midlight']}; }}
QPushButton:pressed {{ background: {t['mid']}; }}
QPushButton:disabled {{ color: {t['disabled_text']}; border-color: {t['midlight']}; background: {t['window']}; }}
QPushButton#primary {{ background: {t['accent']}; color: {t['highlighted_text']}; border-color: {t['accent']};
    font-weight: 600; padding: 5px 18px; }}
QPushButton#primary:hover {{ background: {t['accent_hover']}; border-color: {t['accent_hover']}; }}
QPushButton#primary:disabled {{ background: {t['mid']}; border-color: {t['mid']}; color: {t['disabled_text']}; }}
QPushButton#flat {{ background: transparent; border: none; border-radius: 6px; padding: 4px 8px; }}
QPushButton#flat:hover {{ background: {t['midlight']}; }}
QPushButton#flat:pressed {{ background: {t['mid']}; }}
QPushButton#flat:disabled {{ background: transparent; }}
QToolButton {{ background: transparent; border: none; border-radius: 6px; padding: 4px; }}
QToolButton:hover {{ background: {t['midlight']}; }}
QToolButton:pressed {{ background: {t['mid']}; }}
QComboBox {{ background: {t['button']}; border: 1px solid {t['mid']}; border-radius: 6px; padding: 4px 10px;
    min-height: 20px; }}
QComboBox:hover {{ border-color: {t['dark']}; }}
QComboBox:disabled {{ color: {t['disabled_text']}; background: {t['window']}; border-color: {t['midlight']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {t['base']}; border: 1px solid {t['mid']}; padding: 4px;
    selection-background-color: {t['accent']}; selection-color: {t['highlighted_text']}; outline: 0; }}
QLineEdit, QSpinBox {{ background: {t['base']}; border: 1px solid {t['mid']}; border-radius: 6px; padding: 4px 8px; }}
QLineEdit:focus, QSpinBox:focus {{ border-color: {t['accent']}; }}
QListView {{ background: {t['window']}; border: none; outline: 0; }}
QTextEdit {{ background: {t['base']}; border: none; selection-background-color: {t['accent']};
    selection-color: {t['highlighted_text']}; }}
QPlainTextEdit {{ background: {t['base']}; border: 1px solid {t['midlight']}; border-radius: 6px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t['mid']}; border-radius: 4px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {t['dark']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {t['mid']}; border-radius: 4px; min-width: 28px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}
QSplitter::handle {{ background: {t['midlight']}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QProgressBar {{ background: {t['midlight']}; border: none; border-radius: 2px; max-height: 4px; }}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 2px; }}
QSlider::groove:horizontal {{ height: 4px; background: {t['midlight']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; background: {t['accent']}; }}
QStatusBar {{ background: {t['window']}; border-top: 1px solid {t['midlight']}; color: {t['placeholder']}; }}
QStatusBar::item {{ border: none; }}
QLabel#title {{ font-size: 15pt; font-weight: 600; }}
QLabel#brand {{ font-size: 13pt; font-weight: 700; }}
QLabel#muted {{ color: {t['placeholder']}; }}
QLabel#section {{ color: {t['placeholder']}; font-size: 9pt; font-weight: 600; }}
QLabel#hint {{ color: {t['placeholder']}; font-size: 11pt; }}
QFrame#panel {{ background: {t['base']}; border: 1px solid {t['midlight']}; border-radius: 8px; }}
QFrame#topbar {{ background: {t['window']}; border-bottom: 1px solid {t['midlight']}; }}
QFrame#sidebar {{ background: {t['window']}; }}
QFrame#detail {{ background: {t['base']}; }}
QFrame#playerbar {{ background: {t['window']}; border-top: 1px solid {t['midlight']}; }}
QGroupBox {{ border: 1px solid {t['midlight']}; border-radius: 6px; margin-top: 12px; padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QMenu {{ background: {t['base']}; border: 1px solid {t['mid']}; padding: 4px; }}
QMenu::item {{ padding: 5px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {t['accent']}; color: {t['highlighted_text']}; }}
QCheckBox, QRadioButton {{ spacing: 6px; }}
QMessageBox QLabel {{ font-size: 10pt; }}
"""


def apply_theme(app: QApplication, dark: bool = False) -> Theme:
    """Select the Fusion style and the palette on `app`; return the state colours."""
    app.setStyle("Fusion")
    app.setPalette(palette_for(dark))
    return theme_for(dark)


def apply_styles(app: QApplication, dark: bool = False) -> None:
    """Set the application font and the stylesheet built from the palette table.

    Kept apart from apply_theme because a stylesheet wraps the style in a proxy; screens that
    only need the palette (and tests that check the style by name) call apply_theme alone.
    """
    font = QFont(app.font())
    font.setFamilies(["Segoe UI", "Helvetica Neue", "Arial"])
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(stylesheet(dark))


def with_alpha(colour: QColor, alpha: int) -> QColor:
    """A copy of `colour` with the given alpha (0 to 255)."""
    copy = QColor(colour)
    copy.setAlpha(max(0, min(255, alpha)))
    return copy
