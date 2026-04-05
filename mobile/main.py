"""Momentum mobile app -- Kivy-based Android interface.

Reuses the core momentum modules (models, db, encouragement, assessments) with
a touch-friendly UI designed for phones.  All screens from the desktop GUI
menu bar are replicated here.
"""

from __future__ import annotations

import io
import logging
import os
import random
import re
import ssl
import sys
import threading
import time as _time
import urllib.request
from pathlib import Path
from typing import Callable

# Ensure the parent package is importable when running standalone on desktop
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from kivy.app import App
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.properties import (
    BooleanProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty,
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import NoTransition, Screen, ScreenManager
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from momentum import config as cfg
from momentum import db
from momentum.assessments import (
    BDEFS_INSTRUCTIONS,
    BDEFS_QUESTIONS,
    BISBAS_INSTRUCTIONS,
    BISBAS_QUESTIONS,
    RESULTS_GUIDE,
    STROOP_INSTRUCTIONS,
    StroopResult,
    bisbas_bespoke_guidance,
    bisbas_domain_advice,
    domain_advice,
    generate_stroop_trials,
    interpret_bdefs,
    interpret_bisbas,
    interpret_stroop,
    personalised_act_guidance,
    personalised_nudge,
    profile_from_latest_assessments,
    score_bdefs,
    score_bisbas,
    score_stroop,
    should_show_act_support,
)
from momentum.encouragement import get_break_message, get_nudge
from momentum.models import (
    ActJournalEntryCreate,
    AssessmentType,
    FocusSessionCreate,
    TaskCreate,
    TaskStatus,
    ThemeMode,
    TimerCycleMode,
)

log = logging.getLogger(__name__)

_CHART_FUNCS: tuple | None = None


def _get_chart_funcs() -> tuple | None:
    """Import chart rendering lazily to keep Android startup lightweight."""
    global _CHART_FUNCS
    if _CHART_FUNCS is not None:
        return _CHART_FUNCS
    try:
        from momentum.charts import bdefs_radar, bdefs_timeseries, bisbas_profile_bars

        _CHART_FUNCS = (bdefs_radar, bdefs_timeseries, bisbas_profile_bars)
    except Exception:
        log.debug("Chart module unavailable on this runtime", exc_info=True)
        _CHART_FUNCS = None
    return _CHART_FUNCS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_palette() -> dict[str, list[float]]:
    """Build the active mobile color palette from persisted config."""
    conf = cfg.load_config()
    if conf.theme_mode == ThemeMode.LIGHT:
        palette = {
            "bg": get_color_from_hex("#f5f6f8"),
            "panel": get_color_from_hex("#ffffff"),
            "text": get_color_from_hex("#1f2933"),
            "muted": get_color_from_hex("#5f6b76"),
            "accent": get_color_from_hex("#225b7a"),
            "toolbar": get_color_from_hex("#dbe3ea"),
            "input_bg": get_color_from_hex("#ffffff"),
            "timer": get_color_from_hex("#8b5e00"),
            "neutral_button": get_color_from_hex("#435366"),
            "success_button": get_color_from_hex("#256043"),
            "secondary_button": get_color_from_hex("#56417c"),
            "danger_button": get_color_from_hex("#7f3b3b"),
            "button_text": get_color_from_hex("#ffffff"),
        }
    else:
        palette = {
            "bg": get_color_from_hex("#2b2b2b"),
            "panel": get_color_from_hex("#333333"),
            "text": get_color_from_hex("#e0e0e0"),
            "muted": get_color_from_hex("#b5b5b5"),
            "accent": get_color_from_hex("#6a9fb5"),
            "toolbar": get_color_from_hex("#1e1e1e"),
            "input_bg": get_color_from_hex("#333333"),
            "timer": get_color_from_hex("#e8c547"),
            "neutral_button": get_color_from_hex("#3a3a3a"),
            "success_button": get_color_from_hex("#5a8a5a"),
            "secondary_button": get_color_from_hex("#5a5a7a"),
            "danger_button": get_color_from_hex("#9f6a6a"),
            "button_text": get_color_from_hex("#ffffff"),
        }
    if conf.accessibility_high_contrast:
        if conf.theme_mode == ThemeMode.LIGHT:
            palette["bg"] = get_color_from_hex("#ffffff")
            palette["panel"] = get_color_from_hex("#ffffff")
            palette["text"] = get_color_from_hex("#000000")
            palette["muted"] = get_color_from_hex("#111111")
            palette["accent"] = get_color_from_hex("#0047b3")
            palette["toolbar"] = get_color_from_hex("#d9dfe7")
            palette["input_bg"] = get_color_from_hex("#ffffff")
            palette["neutral_button"] = get_color_from_hex("#111111")
            palette["success_button"] = get_color_from_hex("#005a2b")
            palette["secondary_button"] = get_color_from_hex("#143f85")
            palette["danger_button"] = get_color_from_hex("#7a0000")
            palette["button_text"] = get_color_from_hex("#ffffff")
        else:
            palette["bg"] = get_color_from_hex("#000000")
            palette["panel"] = get_color_from_hex("#0d0d0d")
            palette["text"] = get_color_from_hex("#ffffff")
            palette["muted"] = get_color_from_hex("#f0f0f0")
            palette["accent"] = get_color_from_hex("#8fd3ff")
            palette["toolbar"] = get_color_from_hex("#000000")
            palette["input_bg"] = get_color_from_hex("#141414")
            palette["neutral_button"] = get_color_from_hex("#1f1f1f")
            palette["success_button"] = get_color_from_hex("#0f6a34")
            palette["secondary_button"] = get_color_from_hex("#28538a")
            palette["danger_button"] = get_color_from_hex("#8a1f1f")
            palette["button_text"] = get_color_from_hex("#ffffff")
    return palette


_PALETTE = _resolve_palette()
_APP_CFG = cfg.load_config()
_ACCENT = _PALETTE["accent"]
_TEXT = _PALETTE["text"]
_MUTED = _PALETTE["muted"]
_BG = _PALETTE["bg"]
_BUTTON_TEXT = _PALETTE["button_text"]

# Fallback photo IDs if IMAGES.md is missing
_FALLBACK_PHOTOS = [
    "photo-1506744038136-46273834b3fb",
    "photo-1470071459604-3b5ec3a7fe05",
    "photo-1441974231531-c6227db76b6e",
    "photo-1507525428034-b723cf961d3e",
    "photo-1501854140801-50d01698950b",
    "photo-1465189684280-6a8fa9b19a7a",
    "photo-1433086966358-54859d0ed716",
]


def _load_photos() -> list[str]:
    """Parse photo IDs from IMAGES.md, falling back to built-in list."""
    md_text = _find_md("IMAGES.md")
    ids: list[str] = []
    for line in md_text.splitlines():
        s = line.strip()
        if s.startswith("- photo-"):
            pid = s[2:].strip()
            if pid:
                ids.append(pid)
    seen: set[str] = set()
    unique = [p for p in ids if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]
    return unique if unique else _FALLBACK_PHOTOS


def _pil_to_kivy_image(pil_img: PILImage.Image) -> CoreImage:
    """Convert a PIL Image to a Kivy CoreImage (texture source)."""
    buf = io.BytesIO()
    pil_img.save(buf, format="png")
    buf.seek(0)
    return CoreImage(buf, ext="png")


def _find_md(name: str) -> str:
    """Locate a markdown file bundled with the app."""
    for candidate in [
        Path(os.environ.get("ANDROID_PRIVATE", ".")) / name,
        Path(__file__).resolve().parent / name,
        Path(__file__).resolve().parent.parent / name,
    ]:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return f"# {name}\n\nFile not found."


def _make_label(text, **kw):
    """Create a self-sizing Label with text wrapping."""
    app = App.get_running_app()
    default_text = list(app.text_color) if app and hasattr(app, "text_color") else _TEXT
    font_scale = float(app.font_scale) if app and hasattr(app, "font_scale") else 1.0
    defaults = dict(
        font_size=sp(14), color=default_text,
        size_hint_y=None, text_size=(None, None), halign="left", valign="top",
    )
    defaults.update(kw)
    defaults["font_size"] = float(defaults["font_size"]) * font_scale
    lbl = Label(text=text, **defaults)
    lbl.bind(width=lambda i, w: setattr(i, "text_size", (w - dp(8), None)))
    lbl.bind(texture_size=lambda i, ts: setattr(i, "height", ts[1] + dp(8)))
    return lbl


def _clean_inline(text: str) -> str:
    """Strip markdown inline formatting to plain text for Kivy labels."""
    # [link text](url) -> link text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # **bold** -> bold
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    # `code` -> code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def _render_markdown(container, md_text):
    """Render markdown into a Kivy BoxLayout, matching the desktop GUI."""
    import webbrowser as _webbrowser

    def _add_line_with_links(text, **kw):
        """Add a label, but if the text contains markdown links, make them tappable."""
        link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        matches = list(link_pattern.finditer(text))
        if not matches:
            container.add_widget(_make_label(_clean_inline(text), **kw))
            return
        pos = 0
        for m in matches:
            if m.start() > pos:
                container.add_widget(_make_label(
                    _clean_inline(text[pos:m.start()]), **kw
                ))
            link_text = m.group(1)
            link_url = m.group(2)
            btn = Button(
                text=link_text, font_size=kw.get("font_size", sp(12)),
                size_hint_y=None, height=dp(36),
                background_color=(0.32, 0.51, 0.70, 1),
                halign="left", valign="middle",
            )
            btn.bind(width=lambda i, w: setattr(i, "text_size", (w - dp(8), None)))
            btn.bind(on_release=lambda _, url=link_url: _webbrowser.open(url))
            container.add_widget(btn)
            pos = m.end()
        if pos < len(text):
            container.add_widget(_make_label(_clean_inline(text[pos:]), **kw))

    in_code = False
    for line in md_text.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            container.add_widget(_make_label(
                "  " + line, font_size=sp(11), color=_MUTED,
            ))
            continue
        if not s:
            container.add_widget(Widget(size_hint_y=None, height=dp(6)))
        elif s.startswith("### "):
            container.add_widget(_make_label(
                _clean_inline(s[4:]), font_size=sp(15), bold=True, color=_ACCENT,
            ))
        elif s.startswith("## "):
            container.add_widget(_make_label(
                _clean_inline(s[3:]), font_size=sp(17), bold=True, color=_ACCENT,
            ))
        elif s.startswith("# "):
            container.add_widget(_make_label(
                _clean_inline(s[2:]), font_size=sp(20), bold=True, color=_ACCENT,
            ))
        elif re.match(r"^\|[-\s|:]+\|$", s):
            # Table separator row -- skip
            continue
        elif s.startswith("|") and s.endswith("|"):
            # Table row
            cells = [c.strip() for c in s.strip("|").split("|")]
            row_text = "   ".join(cells)
            container.add_widget(_make_label(
                row_text, font_size=sp(12), color=_TEXT,
            ))
        elif re.match(r"^\d+\.\s", s):
            # Numbered list
            clean = _clean_inline(re.sub(r"^\d+\.\s", "", s))
            num = s.split(".")[0]
            container.add_widget(_make_label(
                f"  {num}. {clean}", font_size=sp(12), color=_MUTED,
            ))
        elif s.startswith("- ") or s.startswith("* "):
            _add_line_with_links(s[2:], font_size=sp(12), color=_MUTED)
        elif re.match(r"^[-*]{3,}$", s):
            # Horizontal rule
            container.add_widget(Widget(size_hint_y=None, height=dp(12)))
        else:
            _add_line_with_links(s)


def _show_error_popup(title: str, text: str) -> None:
    """Display a consistent error popup for recoverable UI callback failures."""
    app = App.get_running_app()
    fg = list(app.text_color) if app else list(_TEXT)
    accent = list(app.accent_color) if app else list(_ACCENT)
    button_text = list(app.button_text_color) if app else list(_BUTTON_TEXT)
    content = BoxLayout(orientation="vertical", padding=10, spacing=10)
    label = Label(text=text, font_size=sp(13), color=fg, text_size=(dp(240), None), size_hint_y=None)
    label.bind(texture_size=lambda inst, val: setattr(inst, 'height', val[1]))
    content.add_widget(label)
    close = Button(
        text="OK",
        size_hint_y=None,
        height=dp(44),
        background_color=accent,
        color=button_text,
    )
    popup = Popup(title=title, content=content, size_hint=(0.86, None), height=dp(220))
    close.bind(on_release=lambda _: popup.dismiss())
    content.add_widget(close)
    popup.open()


def _show_info_popup(title: str, text: str) -> None:
    """Display a neutral informational popup for successful actions."""
    app = App.get_running_app()
    fg = list(app.text_color) if app else list(_TEXT)
    accent = list(app.accent_color) if app else list(_ACCENT)
    button_text = list(app.button_text_color) if app else list(_BUTTON_TEXT)
    content = BoxLayout(orientation="vertical", padding=10, spacing=10)
    label = Label(text=text, font_size=sp(13), color=fg, text_size=(dp(240), None), size_hint_y=None)
    label.bind(texture_size=lambda inst, val: setattr(inst, 'height', val[1]))
    content.add_widget(label)
    close = Button(
        text="OK",
        size_hint_y=None,
        height=dp(44),
        background_color=accent,
        color=button_text,
    )
    popup = Popup(
        title=title,
        content=content,
        size_hint=(0.86, None),
        height=dp(220),
        auto_dismiss=True,
    )
    close.bind(on_release=lambda _: popup.dismiss())
    content.add_widget(close)
    popup.open()


def _run_ui_action(
    action: Callable[[], None],
    *,
    title: str = "Action failed",
    prefix: str = "Something went wrong while handling that action.",
) -> None:
    """Run a UI callback safely and surface failures without crashing the app."""
    try:
        action()
    except Exception as exc:
        log.exception("Unhandled mobile UI callback exception")
        message = f"{prefix}\n\n{exc}"
        Clock.schedule_once(
            lambda _dt, popup_title=title, popup_message=message: _show_error_popup(
                popup_title, popup_message
            ),
            0,
        )


# ---------------------------------------------------------------------------
# Kivy UI definition (KV language)
# ---------------------------------------------------------------------------

KV = """
#:import dp kivy.metrics.dp
#:import sp kivy.metrics.sp

<AccentLabel@Label>:
    color: app.accent_color
    font_size: sp(16) * app.font_scale
    bold: True
    size_hint_y: None
    height: dp(32)
    text_size: self.width, None
    halign: 'left'

<DarkButton@Button>:
    background_color: app.accent_color
    color: app.button_text_color
    font_size: sp(14) * app.font_scale
    size_hint_y: None
    height: dp(44)
    bold: True

<Toolbar>:
    size_hint_y: None
    height: dp(48)
    spacing: dp(2)
    padding: [dp(2), dp(2)]
    canvas.before:
        Color:
            rgba: app.toolbar_color
        Rectangle:
            pos: self.pos
            size: self.size
    Button:
        text: 'Home'
        font_size: sp(12) * app.font_scale
        bold: True
        background_color: app.accent_color
        color: app.button_text_color
        size_hint_x: 0.25
        on_release: root.go('home', 'right')
    Button:
        text: 'Settings'
        font_size: sp(11) * app.font_scale
        background_color: app.neutral_button_color
        color: app.button_text_color
        size_hint_x: 0.25
        on_release: root.go('settings', 'left')
    Button:
        text: 'Help'
        font_size: sp(12) * app.font_scale
        background_color: app.neutral_button_color
        color: app.button_text_color
        size_hint_x: 0.25
        on_release: root.go('help_menu', 'left')
    Button:
        text: 'Tests'
        font_size: sp(12) * app.font_scale
        background_color: app.neutral_button_color
        color: app.button_text_color
        size_hint_x: 0.25
        on_release: root.go('tests_menu', 'left')

<TaskRow>:
    size_hint_y: None
    height: dp(48)
    canvas.before:
        Color:
            rgba: root.bg_color
        Rectangle:
            pos: self.pos
            size: self.size
    Label:
        text: root.icon
        size_hint_x: 0.1
        color: app.accent_color
        font_size: sp(16) * app.font_scale
    Label:
        text: root.title_text
        size_hint_x: 0.7
        text_size: self.size
        halign: 'left'
        valign: 'middle'
        color: app.text_color
        font_size: sp(14) * app.font_scale
    Button:
        text: root.btn_text
        size_hint_x: 0.2
        background_color: root.btn_color
        color: app.button_text_color
        on_release: root.on_complete()

<HomeScreen>:
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: app.bg_color
            Rectangle:
                pos: self.pos
                size: self.size
        BoxLayout:
            id: banner_box
            size_hint_y: None
            height: dp(100)
            padding: [dp(8), dp(4)]
        Label:
            text: root.status_text
            size_hint_y: None
            height: dp(32)
            color: app.accent_color
            font_size: sp(13) * app.font_scale
        Label:
            text: 'Tasks'
            size_hint_y: None
            height: dp(28)
            color: app.accent_color
            font_size: sp(18) * app.font_scale
            bold: True
            halign: 'left'
            text_size: self.size
            padding: [dp(12), 0]
        ScrollView:
            size_hint_y: 0.35
            do_scroll_x: False
            BoxLayout:
                id: task_list
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(2)
                padding: [dp(8), 0]
        BoxLayout:
            size_hint_y: None
            height: dp(44)
            spacing: dp(8)
            padding: [dp(8), dp(4)]
            Button:
                text: 'Add task'
                background_color: app.accent_color
                color: app.button_text_color
                on_release: root.show_add_dialog()
            Button:
                text: 'Break down'
                background_color: app.success_button_color
                color: app.button_text_color
                on_release: root.show_breakdown_dialog()
            Button:
                text: 'Completed'
                background_color: app.secondary_button_color
                color: app.button_text_color
                on_release: root.toggle_show_completed()
        Label:
            text: 'Timer'
            size_hint_y: None
            height: dp(28)
            color: app.accent_color
            font_size: sp(18) * app.font_scale
            bold: True
            halign: 'left'
            text_size: self.size
            padding: [dp(12), 0]
        Label:
            text: root.timer_display
            size_hint_y: None
            height: dp(50)
            color: app.timer_color
            font_size: sp(36) * app.font_scale
            bold: True
        ProgressBar:
            value: root.timer_progress
            max: 100
            size_hint_y: None
            height: dp(8)
        BoxLayout:
            size_hint_y: None
            height: dp(48)
            spacing: dp(8)
            padding: [dp(8), dp(4)]
            Button:
                text: root.focus_button_text
                background_color: app.accent_color
                color: app.button_text_color
                font_size: sp(14) * app.font_scale
                bold: True
                on_release: root.start_focus()
            Button:
                text: root.break_button_text
                background_color: app.secondary_button_color
                color: app.button_text_color
                font_size: sp(14) * app.font_scale
                on_release: root.start_break()
            Button:
                text: 'Stop'
                background_color: app.danger_button_color
                color: app.button_text_color
                font_size: sp(14) * app.font_scale
                on_release: root.stop_timer()
        Label:
            text: root.nudge_text
            size_hint_y: None
            height: dp(56)
            color: app.muted_color
            font_size: sp(13) * app.font_scale
            italic: True
            text_size: self.width - dp(24), None
            halign: 'center'
        BoxLayout:
            size_hint_y: None
            height: dp(40)
            spacing: dp(8)
            padding: [dp(8), 0]
            Button:
                text: 'New encouragement'
                background_color: app.secondary_button_color
                color: app.button_text_color
                on_release: root.refresh_nudge()
        BoxLayout:
            size_hint_y: None
            height: dp(40) if root.act_controls_visible else dp(0)
            spacing: dp(8)
            padding: [dp(8), 0]
            opacity: 1 if root.act_controls_visible else 0
            disabled: not root.act_controls_visible
            Button:
                text: 'ACT check-in'
                background_color: app.success_button_color
                color: app.button_text_color
                on_release: root.open_act_checkin()
            Button:
                text: 'ACT history'
                background_color: app.neutral_button_color
                color: app.button_text_color
                on_release: root.open_act_history()
        Widget:
            size_hint_y: None
            height: dp(4)
        Toolbar:

<ScrollScreen>:
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: app.bg_color
            Rectangle:
                pos: self.pos
                size: self.size
        ScrollView:
            do_scroll_x: False
            BoxLayout:
                id: content
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                padding: [dp(16), dp(12)]
                spacing: dp(4)
        Toolbar:

<TestsMenuScreen>:
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: app.bg_color
            Rectangle:
                pos: self.pos
                size: self.size
        BoxLayout:
            orientation: 'vertical'
            padding: [dp(16), dp(24)]
            spacing: dp(12)
            AccentLabel:
                text: 'Self-Assessment Tests'
            DarkButton:
                text: 'Take BDEFS Assessment'
                on_release: root.go_bdefs()
            DarkButton:
                text: 'Take BIS/BAS Profile'
                background_color: app.secondary_button_color
                on_release: root.go_bisbas()
            DarkButton:
                text: 'Take Stroop Test'
                background_color: app.neutral_button_color
                on_release: root.go_stroop()
            Widget:
                size_hint_y: None
                height: dp(16)
            AccentLabel:
                text: 'Results'
            DarkButton:
                text: 'View Past Results'
                background_color: app.success_button_color
                on_release: root.go_results()
            Widget:
        Toolbar:

<HelpMenuScreen>:
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: app.bg_color
            Rectangle:
                pos: self.pos
                size: self.size
        BoxLayout:
            orientation: 'vertical'
            padding: [dp(16), dp(24)]
            spacing: dp(12)
            AccentLabel:
                text: 'Help'
            DarkButton:
                text: 'How to Use'
                on_release: root.go_howto()
            DarkButton:
                text: 'The Science'
                background_color: app.success_button_color
                on_release: root.go_science()
            DarkButton:
                text: 'About'
                background_color: app.secondary_button_color
                on_release: root.go_about()
            Widget:
        Toolbar:

<BdefsScreen>:
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: app.bg_color
            Rectangle:
                pos: self.pos
                size: self.size
        ScrollView:
            do_scroll_x: False
            BoxLayout:
                id: bdefs_content
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                padding: [dp(12), dp(8)]
                spacing: dp(4)
        Toolbar:

<BisbasScreen>:
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: app.bg_color
            Rectangle:
                pos: self.pos
                size: self.size
        ScrollView:
            do_scroll_x: False
            BoxLayout:
                id: bisbas_content
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                padding: [dp(12), dp(8)]
                spacing: dp(4)
        Toolbar:

<StroopScreen>:
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: app.bg_color
            Rectangle:
                pos: self.pos
                size: self.size
        BoxLayout:
            orientation: 'vertical'
            padding: [dp(16), dp(16)]
            spacing: dp(12)
            Label:
                text: 'Type the COLOUR of the text, not the word.'
                color: app.text_color
                halign: 'center'
                font_size: sp(15) * app.font_scale
                size_hint_y: None
                height: dp(30)
            Label:
                text: root.progress_text
                color: app.muted_color
                font_size: sp(13) * app.font_scale
                size_hint_y: None
                height: dp(24)
            Label:
                text: root.word_text
                font_size: sp(48) * app.font_scale
                bold: True
                color: root.word_color
                size_hint_y: None
                height: dp(80)
            TextInput:
                id: stroop_input
                hint_text: 'Type colour here...'
                multiline: False
                font_size: sp(18) * app.font_scale
                size_hint_y: None
                height: dp(48)
                background_color: app.input_bg_color
                foreground_color: app.text_color
                on_text_validate: root.on_answer(self.text)
            Label:
                text: root.feedback_text
                color: app.muted_color
                font_size: sp(13) * app.font_scale
                size_hint_y: None
                height: dp(24)
            Widget:
        Toolbar:
"""


# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------


class Toolbar(BoxLayout):
    """Bottom navigation bar present on every screen."""

    def go(self, screen_name, direction):
        def _navigate() -> None:
            sm = App.get_running_app().root
            sm.transition = NoTransition()
            sm.current = screen_name

        _run_ui_action(_navigate, prefix="Navigation failed.")

    def on_parent(self, *args):
        """Bind to owning screen's on_enter so we highlight on every visit."""
        def _bind(dt):
            screen = self._find_screen()
            if screen:
                screen.bind(on_enter=lambda *_a: self._update_highlight())
            self._update_highlight()
        Clock.schedule_once(_bind, 0)

    def _find_screen(self):
        w = self.parent
        while w is not None:
            if isinstance(w, Screen):
                return w
            w = w.parent
        return None

    def _update_highlight(self):
        app = App.get_running_app()
        sm = app.root
        if sm is None or app is None:
            return
        current = sm.current
        accent = list(app.accent_color)
        neutral = list(app.neutral_button_color)
        screen_map = {
            "home": 0,
            "settings": 1,
        }
        help_screens = {"help_menu", "howto", "science", "about"}
        test_screens = {"tests_menu", "bdefs", "bisbas", "stroop", "results"}
        for i, child in enumerate(self.children[::-1]):
            if not hasattr(child, "background_color"):
                continue
            is_active = False
            if current in screen_map and screen_map[current] == i:
                is_active = True
            elif current in help_screens and i == 2:
                is_active = True
            elif current in test_screens and i == 3:
                is_active = True
            child.background_color = accent if is_active else neutral


class TaskRow(BoxLayout):
    """A single task row in the task list."""

    task_id = NumericProperty(0)
    icon = StringProperty("[ ]")
    title_text = StringProperty("")
    btn_text = StringProperty("Done")
    btn_color = ListProperty([0.29, 0.478, 0.29, 1])
    bg_color = ListProperty([0.2, 0.2, 0.2, 1])
    is_completed = BooleanProperty(False)
    home_ref = ObjectProperty(None, allownone=True)

    def on_complete(self):
        if self.home_ref:
            if self.is_completed:
                self.home_ref.uncomplete_task(self.task_id)
            else:
                self.home_ref.complete_task(self.task_id)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos) and self.home_ref and not self.is_completed:
            self.home_ref.select_task(self.task_id)
        return super().on_touch_down(touch)


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


class HomeScreen(Screen):
    """Main screen: tasks, timer, encouragement."""

    status_text = StringProperty("Loading...")
    timer_display = StringProperty("00:00")
    timer_progress = NumericProperty(0)
    nudge_text = StringProperty("")
    focus_button_text = StringProperty("Focus 15m")
    break_button_text = StringProperty("Break 5m")
    act_controls_visible = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conn = None
        self._timer_running = False
        self._timer_seconds_left = 0
        self._timer_total = 0
        self._timer_task_id = None
        self._timer_is_break = False
        self._timer_event = None
        self._selected_task_id = None
        self._auto_cycle_task_id = None
        self._banner_loaded = False

    def on_enter(self):
        def _load() -> None:
            if self.conn is None:
                self.conn = db.get_connection()
            self._refresh_profile_ui()
            self.nudge_text = personalised_nudge(get_nudge(), self._profile())
            Clock.schedule_once(lambda _dt: self.refresh_all(), 0)
            if not self._banner_loaded:
                self._banner_loaded = True
                # Show fallback banner immediately so the user always sees it
                fallback = self._make_fallback_banner()
                self._set_banner(fallback)
                # Then try to fetch a real image in the background
                app = App.get_running_app()
                if app is None or not app.reduce_visual_load:
                    threading.Thread(target=self._fetch_banner, daemon=True).start()

        _run_ui_action(_load, prefix="Could not load the home screen.")

    def _profile(self):
        latest_bisbas = db.list_assessments(
            self.conn, assessment_type=AssessmentType.BISBAS, limit=1
        )
        latest_bdefs = db.list_assessments(
            self.conn, assessment_type=AssessmentType.BDEFS, limit=1
        )
        latest_stroop = db.list_assessments(
            self.conn, assessment_type=AssessmentType.STROOP, limit=1
        )
        return profile_from_latest_assessments(
            latest_bisbas=latest_bisbas[0] if latest_bisbas else None,
            latest_bdefs=latest_bdefs[0] if latest_bdefs else None,
            latest_stroop=latest_stroop[0] if latest_stroop else None,
        )

    def _refresh_profile_ui(self):
        profile = self._profile()
        self.focus_button_text = f"Focus {profile.focus_minutes}m"
        self.break_button_text = f"Break {profile.break_minutes}m"
        self.act_controls_visible = should_show_act_support(profile)

    def _act_guidance(self) -> str:
        return personalised_act_guidance(self._profile())

    def _act_prompt_details(self) -> dict[str, str]:
        profile = self._profile()
        reassurance = profile.add_reassurance
        breakdown = profile.suggest_breakdown
        return {
            "values_focus": (
                "Write one value to steer this next step (for example: care, stability, "
                "learning, contribution)."
            ),
            "challenge_context": (
                "Name the situation making momentum hard right now. Keep it concrete and "
                "specific to today."
            ),
            "thoughts_feelings": (
                "List the thoughts and feelings present without arguing with them."
                if reassurance
                else "Name the main thoughts and emotions showing up right now."
            ),
            "defusion_reframe": (
                "Try: 'I am noticing the thought that ...' then write a gentler or more "
                "workable framing."
            ),
            "committed_action": (
                "Choose one tiny action you can complete in 2-5 minutes."
                if breakdown
                else "Choose one values-aligned next action you can realistically do now."
            ),
        }

    # -- Banner image --

    @staticmethod
    def _find_font(size: int = 32):
        """Try multiple font paths (Android, Linux, macOS) before falling back."""
        candidates = [
            # Android system fonts
            "/system/fonts/Roboto-Bold.ttf",
            "/system/fonts/DroidSans-Bold.ttf",
            "/system/fonts/Roboto-Regular.ttf",
            "/system/fonts/DroidSans.ttf",
            # Linux desktop
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            # macOS
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
        # Pillow >= 10.1 supports size param on load_default
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    @classmethod
    def _make_fallback_banner(cls):
        """Create a solid-colour banner with 'Momentum' text."""
        img = PILImage.new("RGB", (500, 120), (58, 90, 106))
        cls._draw_title(img)
        return img

    def _fetch_banner(self):
        """Download a random peaceful photo in a background thread."""
        photos = _load_photos()
        attempts = min(5, len(photos))
        tried: set[str] = set()
        for _ in range(attempts):
            photo_id = random.choice(photos)
            if photo_id in tried:
                continue
            tried.add(photo_id)
            url = (
                f"https://images.unsplash.com/{photo_id}"
                f"?w=500&h=120&fit=crop&crop=center&auto=format&q=80"
            )
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Momentum/0.1"})
                # Try with default SSL first, fall back to unverified context
                # (Android may lack system CA certs for urllib)
                try:
                    resp = urllib.request.urlopen(req, timeout=8)
                except (urllib.error.URLError, ssl.SSLError):
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    resp = urllib.request.urlopen(req, timeout=8, context=ctx)
                data = resp.read()
                resp.close()
                pil_img = PILImage.open(io.BytesIO(data))
                pil_img = pil_img.resize((500, 120), PILImage.LANCZOS)
                self._draw_title(pil_img)
                Clock.schedule_once(lambda dt, img=pil_img: self._set_banner(img), 0)
                return
            except Exception:
                log.warning("Banner fetch failed for %s", photo_id, exc_info=True)
        log.warning("Could not fetch any banner image; fallback already shown.")

    @classmethod
    def _draw_title(cls, image):
        """Draw 'Momentum' centred on the banner with a drop shadow."""
        draw = ImageDraw.Draw(image)
        font = cls._find_font(32)
        text = "Momentum"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (image.width - tw) // 2
        y = (image.height - th) // 2
        draw.text((x + 2, y + 2), text, fill=(0, 0, 0, 180), font=font)
        draw.text((x, y), text, fill="white", font=font)

    def _set_banner(self, pil_img):
        """Display the fetched image in the banner area (main thread)."""
        core_img = _pil_to_kivy_image(pil_img)
        banner_box = self.ids.banner_box
        banner_box.clear_widgets()
        img_widget = KivyImage(texture=core_img.texture, allow_stretch=True,
                               keep_ratio=False, size_hint=(1, 1))
        banner_box.add_widget(img_widget)

    # -- Data refresh --

    def refresh_all(self):
        self.refresh_tasks()
        self.refresh_status()

    def refresh_tasks(self):
        task_list = self.ids.task_list
        task_list.clear_widgets()
        prev_selected = self._selected_task_id
        self._selected_task_id = None

        _NORMAL_BG = [0.2, 0.2, 0.2, 1]
        _SEL_BG = [0.25, 0.25, 0.38, 1]
        _GREEN = [0.29, 0.478, 0.29, 1]
        _PURPLE = [0.35, 0.35, 0.48, 1]

        active = db.list_tasks(self.conn, status=TaskStatus.ACTIVE)
        pending = db.list_tasks(self.conn, status=TaskStatus.PENDING)
        rows: list[TaskRow] = []
        for task in active + pending:
            row = TaskRow()
            row.task_id = task.id
            row.icon = "[~]" if task.status == TaskStatus.ACTIVE else "[ ]"
            prefix = "  " if task.is_subtask else ""
            row.title_text = f"{prefix}#{task.id} {task.title}"
            row.home_ref = self
            row.btn_text = "Done"
            row.btn_color = _GREEN
            row.bg_color = _NORMAL_BG
            task_list.add_widget(row)
            rows.append(row)

        # Restore previous selection, or fall back to first task
        sel_row = None
        if prev_selected is not None:
            for r in rows:
                if r.task_id == prev_selected:
                    sel_row = r
                    break
        if sel_row is None and rows:
            sel_row = rows[0]
        if sel_row:
            self._selected_task_id = sel_row.task_id
            sel_row.bg_color = _SEL_BG

        if getattr(self, "_show_completed", False):
            done = db.list_tasks(self.conn, status=TaskStatus.DONE)
            for task in done:
                row = TaskRow()
                row.task_id = task.id
                row.icon = "[x]"
                prefix = "  " if task.is_subtask else ""
                row.title_text = f"{prefix}#{task.id} {task.title}"
                row.home_ref = self
                row.btn_text = "Undo"
                row.btn_color = _PURPLE
                row.is_completed = True
                row.bg_color = _NORMAL_BG
                task_list.add_widget(row)

    def refresh_status(self):
        summary = db.get_status(self.conn)
        today = summary.today
        streak = summary.streak_days
        self.status_text = (
            f"Today: {today.tasks_completed} done, {today.focus_minutes}m focused  |  "
            f"Streak: {streak} day{'s' if streak != 1 else ''}"
        )

    # -- Task dialogs --

    def show_add_dialog(self):
        content = BoxLayout(orientation="vertical", spacing=10, padding=10)
        ti = TextInput(hint_text="What do you need to do?", multiline=False,
                       size_hint_y=None, height=dp(44))
        content.add_widget(ti)
        btn_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=8)
        popup = Popup(title="Add task", content=content, size_hint=(0.9, 0.3))

        def on_add(_):
            title = ti.text.strip()
            if title:
                db.add_task(self.conn, TaskCreate(title=title))
                self.refresh_all()
            popup.dismiss()

        add_btn = Button(text="Add")
        add_btn.bind(on_release=on_add)
        cancel_btn = Button(text="Cancel")
        cancel_btn.bind(on_release=lambda _: popup.dismiss())
        btn_row.add_widget(add_btn)
        btn_row.add_widget(cancel_btn)
        content.add_widget(btn_row)
        popup.open()

    def show_breakdown_dialog(self):
        if self._selected_task_id is None:
            return
        task = db.get_task(self.conn, self._selected_task_id)
        if task is None:
            return
        content = BoxLayout(orientation="vertical", spacing=10, padding=10)
        content.add_widget(Label(text=f'Breaking down: "{task.title}"', font_size=sp(14)))
        ti = TextInput(hint_text="Add a sub-step", multiline=False,
                       size_hint_y=None, height=dp(44))
        content.add_widget(ti)
        btn_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=8)
        popup = Popup(title="Break down", content=content, size_hint=(0.9, 0.35))

        def on_add(_):
            step = ti.text.strip()
            if step:
                db.add_task(self.conn, TaskCreate(title=step, parent_id=task.id))
                ti.text = ""
                self.refresh_tasks()

        add_btn = Button(text="Add step")
        add_btn.bind(on_release=on_add)
        done_btn = Button(text="Done")
        done_btn.bind(on_release=lambda _: popup.dismiss())
        btn_row.add_widget(add_btn)
        btn_row.add_widget(done_btn)
        content.add_widget(btn_row)
        popup.open()

    def toggle_show_completed(self):
        self._show_completed = not getattr(self, "_show_completed", False)
        self.refresh_tasks()

    def select_task(self, task_id):
        """Highlight a task row as selected (for breakdown, timer, etc.)."""
        self._selected_task_id = task_id
        _SEL_BG = [0.25, 0.25, 0.38, 1]
        _NORMAL_BG = [0.2, 0.2, 0.2, 1]
        for child in self.ids.task_list.children:
            if isinstance(child, TaskRow) and not child.is_completed:
                child.bg_color = _SEL_BG if child.task_id == task_id else _NORMAL_BG

    def complete_task(self, task_id):
        db.complete_task(self.conn, task_id)
        self.nudge_text = personalised_nudge(get_nudge(), self._profile())
        self.refresh_all()

    def uncomplete_task(self, task_id):
        db.uncomplete_task(self.conn, task_id)
        self.refresh_all()

    # -- Timer --

    def start_focus(self):
        _run_ui_action(
            lambda: self._start_timer(self._profile().focus_minutes, is_break=False),
            prefix="Could not start focus timer.",
        )

    def start_break(self):
        _run_ui_action(
            lambda: self._start_timer(self._profile().break_minutes, is_break=True),
            prefix="Could not start break timer.",
        )

    def _cycle_mode(self) -> TimerCycleMode:
        return cfg.load_config().timer_cycle_mode

    def _start_timer(self, minutes, is_break=False, task_id=None):
        if self._timer_running:
            return
        self._timer_running = True
        self._timer_is_break = is_break
        self._timer_total = minutes * 60
        self._timer_seconds_left = self._timer_total
        active_task_id = task_id if task_id is not None else self._selected_task_id
        if not is_break and active_task_id is not None:
            self._timer_task_id = active_task_id
            self._auto_cycle_task_id = active_task_id
            db.set_task_active(self.conn, active_task_id)
            self.refresh_tasks()
        else:
            self._timer_task_id = None
            if not is_break:
                self._auto_cycle_task_id = None
        self._timer_event = Clock.schedule_interval(self._tick, 1)

    def _tick(self, dt):
        if not self._timer_running:
            if self._timer_event:
                self._timer_event.cancel()
            return
        elapsed = self._timer_total - self._timer_seconds_left
        self.timer_progress = (elapsed / self._timer_total * 100) if self._timer_total else 0
        mins, secs = divmod(self._timer_seconds_left, 60)
        self.timer_display = f"{mins:02d}:{secs:02d}"
        if self._timer_seconds_left <= 0:
            self._timer_running = False
            if self._timer_event:
                self._timer_event.cancel()
            self._on_timer_complete()
            return
        self._timer_seconds_left -= 1

    def _on_timer_complete(self):
        cycle_mode = self._cycle_mode()
        if self._timer_is_break:
            if cycle_mode == TimerCycleMode.AUTO:
                self.nudge_text = "Break complete. Starting the next focus block."
                Clock.schedule_once(
                    lambda _dt: self._start_timer(
                        self._profile().focus_minutes,
                        is_break=False,
                        task_id=self._auto_cycle_task_id,
                    ),
                    0.2,
                )
            else:
                self.nudge_text = get_break_message()
                _show_info_popup(
                    "Break complete",
                    "Break time is up. Start your next focus block when ready.",
                )
        else:
            minutes = self._timer_total // 60
            session_in = FocusSessionCreate(
                task_id=self._timer_task_id, duration_minutes=minutes
            )
            db.log_focus_session(self.conn, session_in)
            self.refresh_status()
            self._refresh_profile_ui()
            self.nudge_text = personalised_nudge(get_nudge(), self._profile())
            if cycle_mode == TimerCycleMode.AUTO:
                self.nudge_text = (
                    f"{minutes}-minute focus session logged. Starting break now."
                )
                Clock.schedule_once(
                    lambda _dt: self._start_timer(
                        self._profile().break_minutes,
                        is_break=True,
                    ),
                    0.2,
                )
            else:
                _show_info_popup(
                    "Focus complete",
                    f"{minutes}-minute focus session logged.",
                )

    def stop_timer(self):
        def _stop() -> None:
            self._timer_running = False
            self._auto_cycle_task_id = None
            if self._timer_event:
                self._timer_event.cancel()
            self.timer_display = "00:00"
            self.timer_progress = 0

        _run_ui_action(_stop, prefix="Could not stop the timer.")

    def refresh_nudge(self):
        _run_ui_action(
            lambda: setattr(
                self,
                "nudge_text",
                personalised_nudge(get_nudge(), self._profile()),
            ),
            prefix="Could not refresh encouragement.",
        )

    def open_act_checkin(self) -> None:
        """Open a structured ACT journaling check-in popup."""

        def _open() -> None:
            if not self.act_controls_visible:
                _show_info_popup(
                    "ACT check-in",
                    "ACT prompts unlock when recent assessments suggest extra support is useful.",
                )
                return
            prompts = (
                ("values_focus", "Value focus"),
                ("challenge_context", "Current challenge"),
                ("thoughts_feelings", "Thoughts & feelings"),
                ("defusion_reframe", "Defusion / reframe"),
                ("committed_action", "Committed action"),
            )
            details = self._act_prompt_details()
            app = App.get_running_app()
            button_text = list(app.button_text_color)
            content = BoxLayout(orientation="vertical", spacing=8, padding=10)
            content.add_widget(_make_label(self._act_guidance(), font_size=sp(12), color=_MUTED))
            fields: dict[str, TextInput] = {}
            for key, title in prompts:
                title_row = BoxLayout(size_hint_y=None, height=dp(30), spacing=dp(6))
                title_row.add_widget(_make_label(title, font_size=sp(12), bold=True, size_hint_x=0.88))
                info_btn = Button(
                    text="ⓘ",
                    size_hint_x=0.12,
                    background_color=list(app.neutral_button_color),
                    color=button_text,
                    font_size=sp(14),
                )
                info_btn.bind(
                    on_release=lambda _btn, t=title, d=details[key]: _show_info_popup(t, d)
                )
                title_row.add_widget(info_btn)
                content.add_widget(title_row)
                ti = TextInput(
                    multiline=True,
                    size_hint_y=None,
                    height=dp(70),
                    background_color=app.input_bg_color,
                    foreground_color=app.text_color,
                )
                content.add_widget(ti)
                fields[key] = ti
            btn_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=8)
            save_btn = Button(
                text="Save",
                background_color=app.accent_color,
                color=button_text,
            )
            close_btn = Button(
                text="Close",
                background_color=app.neutral_button_color,
                color=button_text,
            )
            popup = Popup(
                title="ACT Journal Check-In",
                content=content,
                size_hint=(0.94, 0.92),
                auto_dismiss=False,
            )

            def _save(_btn) -> None:
                values = {k: v.text.strip() for k, v in fields.items()}
                if any(not val for val in values.values()):
                    _show_error_popup(
                        "Incomplete entry",
                        "Please fill in all ACT check-in fields.",
                    )
                    return
                created = db.add_act_journal_entry(
                    self.conn,
                    ActJournalEntryCreate(
                        values_focus=values["values_focus"],
                        challenge_context=values["challenge_context"],
                        thoughts_feelings=values["thoughts_feelings"],
                        defusion_reframe=values["defusion_reframe"],
                        committed_action=values["committed_action"],
                    ),
                )
                self.nudge_text = (
                    f"ACT check-in saved. Next action: {created.committed_action}"
                )
                popup.dismiss()

            save_btn.bind(on_release=lambda btn: _run_ui_action(lambda: _save(btn)))
            close_btn.bind(on_release=lambda _: popup.dismiss())
            btn_row.add_widget(save_btn)
            btn_row.add_widget(close_btn)
            content.add_widget(btn_row)
            popup.open()

        _run_ui_action(_open, prefix="Could not open ACT check-in.")

    def open_act_history(self) -> None:
        """Show recent ACT journaling entries."""

        def _open() -> None:
            if not self.act_controls_visible:
                _show_info_popup(
                    "ACT history",
                    "ACT history appears once ACT support is enabled by recent assessment signals.",
                )
                return
            entries = db.list_act_journal_entries(self.conn, limit=20)
            scroll = ScrollView(do_scroll_x=False)
            inner = BoxLayout(
                orientation="vertical",
                spacing=6,
                padding=10,
                size_hint_y=None,
            )
            inner.bind(minimum_height=inner.setter("height"))
            inner.add_widget(_make_label(self._act_guidance(), font_size=sp(12), color=_MUTED))
            inner.add_widget(Widget(size_hint_y=None, height=dp(4)))
            if not entries:
                inner.add_widget(
                    _make_label("No ACT journal entries yet.", color=_MUTED)
                )
            else:
                for entry in entries:
                    inner.add_widget(
                        _make_label(
                            f"#{entry.id}  {entry.created_at:%Y-%m-%d %H:%M}",
                            font_size=sp(13),
                            bold=True,
                            color=_ACCENT,
                        )
                    )
                    inner.add_widget(
                        _make_label(f"Values: {entry.values_focus}", font_size=sp(12))
                    )
                    inner.add_widget(
                        _make_label(
                            f"Challenge: {entry.challenge_context}",
                            font_size=sp(12),
                            color=_MUTED,
                        )
                    )
                    inner.add_widget(
                        _make_label(
                            f"Thoughts/feelings: {entry.thoughts_feelings}",
                            font_size=sp(12),
                            color=_MUTED,
                        )
                    )
                    inner.add_widget(
                        _make_label(
                            f"Defusion/reframe: {entry.defusion_reframe}",
                            font_size=sp(12),
                            color=_MUTED,
                        )
                    )
                    inner.add_widget(
                        _make_label(
                            f"Committed action: {entry.committed_action}",
                            font_size=sp(12),
                            color=_MUTED,
                        )
                    )
                    inner.add_widget(Widget(size_hint_y=None, height=dp(8)))
            scroll.add_widget(inner)
            content = BoxLayout(orientation="vertical", padding=10, spacing=8)
            content.add_widget(scroll)
            close = Button(text="Close", size_hint_y=None, height=dp(44))
            popup = Popup(
                title="ACT Journal History",
                content=content,
                size_hint=(0.94, 0.88),
            )
            close.bind(on_release=lambda _: popup.dismiss())
            content.add_widget(close)
            popup.open()

        _run_ui_action(_open, prefix="Could not open ACT journal history.")


# ---------------------------------------------------------------------------
# Generic scrollable-content screen (reused by several pages)
# ---------------------------------------------------------------------------


class ScrollScreen(Screen):
    """A screen with a toolbar and a scrollable content area (#content)."""
    pass


# ---------------------------------------------------------------------------
# Menu -> Settings
# ---------------------------------------------------------------------------


class SettingsScreen(ScrollScreen):
    """Settings: data paths, appearance/accessibility, and management."""

    def on_enter(self):
        c = self.ids.content
        c.clear_widgets()

        app = App.get_running_app()
        accent = list(app.accent_color) if app else list(_ACCENT)
        text = list(app.text_color) if app else list(_TEXT)
        button_text = list(app.button_text_color) if app else list(_BUTTON_TEXT)
        muted = list(app.muted_color) if app else list(_MUTED)
        neutral = list(app.neutral_button_color) if app else [0.25, 0.25, 0.25, 1]
        secondary = list(app.secondary_button_color) if app else [0.35, 0.35, 0.48, 1]
        success = list(app.success_button_color) if app else [0.29, 0.478, 0.29, 1]
        danger = list(app.danger_button_color) if app else [0.55, 0.35, 0.35, 1]
        input_bg = list(app.input_bg_color) if app else [0.2, 0.2, 0.2, 1]
        font_scale = app.font_scale if app else 1.0

        current = cfg.load_config()
        resolved = cfg.get_db_path()
        db_text = current.db_path if current.db_path else f"{resolved} (default)"

        c.add_widget(_make_label("Settings", font_size=sp(20), bold=True, color=accent))
        c.add_widget(Widget(size_hint_y=None, height=dp(8)))

        # -- Database location --
        c.add_widget(_make_label("Database Location", font_size=sp(16), bold=True, color=accent))
        self._db_label = _make_label(db_text, font_size=sp(12), color=muted)
        c.add_widget(self._db_label)

        # -- Cloud sync --
        c.add_widget(Widget(size_hint_y=None, height=dp(8)))
        c.add_widget(_make_label("Sync via Cloud", font_size=sp(16), bold=True, color=accent))
        cloud_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        for provider in ("OneDrive", "Dropbox", "Google Drive"):
            btn = Button(
                text=provider,
                font_size=sp(12) * font_scale,
                background_color=list(neutral),
                color=list(button_text),
            )
            key = provider.lower().replace(" ", "-")
            btn.bind(on_release=lambda _, p=key: self._sync(p))
            cloud_row.add_widget(btn)
        c.add_widget(cloud_row)

        # -- Custom path --
        c.add_widget(Widget(size_hint_y=None, height=dp(8)))
        c.add_widget(_make_label("Custom Database Path", font_size=sp(16), bold=True, color=accent))
        path_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        self._path_input = TextInput(
            hint_text="Enter path...",
            multiline=False,
            font_size=sp(13) * font_scale,
            size_hint_x=0.7,
            background_color=input_bg,
            foreground_color=text,
        )
        path_row.add_widget(self._path_input)
        set_btn = Button(
            text="Set",
            size_hint_x=0.3,
            font_size=sp(13) * font_scale,
            background_color=list(accent),
            color=list(button_text),
        )
        set_btn.bind(on_release=lambda _: self._set_custom())
        path_row.add_widget(set_btn)
        c.add_widget(path_row)

        # -- Appearance --
        c.add_widget(Widget(size_hint_y=None, height=dp(12)))
        c.add_widget(_make_label("Appearance", font_size=sp(16), bold=True, color=accent))
        theme_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        for label, mode, color in (
            ("Dark", ThemeMode.DARK.value, neutral),
            ("Light", ThemeMode.LIGHT.value, secondary),
        ):
            tb = ToggleButton(
                text=label,
                group="theme_mode",
                state="down" if current.theme_mode.value == mode else "normal",
                background_color=list(color),
                color=list(button_text),
                font_size=sp(12) * font_scale,
            )
            tb.bind(
                on_release=lambda inst, m=mode: self._set_theme(m) if inst.state == "down" else None
            )
            theme_row.add_widget(tb)
        c.add_widget(theme_row)
        c.add_widget(Widget(size_hint_y=None, height=dp(8)))
        c.add_widget(_make_label("Timer cycle mode", font_size=sp(16), bold=True, color=accent))
        cycle_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        for label, mode, color in (
            ("Manual", TimerCycleMode.MANUAL.value, neutral),
            ("Auto focus↔break", TimerCycleMode.AUTO.value, success),
        ):
            tb = ToggleButton(
                text=label,
                group="timer_cycle_mode",
                state="down" if current.timer_cycle_mode.value == mode else "normal",
                background_color=list(color),
                color=list(button_text),
                font_size=sp(11) * font_scale,
            )
            tb.bind(
                on_release=lambda inst, m=mode: (
                    self._set_timer_cycle_mode(m) if inst.state == "down" else None
                )
            )
            cycle_row.add_widget(tb)
        c.add_widget(cycle_row)
        c.add_widget(_make_label(
            "Manual keeps focus and break separate. Auto chains focus and break "
            "until you press Stop.",
            font_size=sp(11),
            color=muted,
        ))

        # -- Accessibility --
        c.add_widget(Widget(size_hint_y=None, height=dp(8)))
        c.add_widget(_make_label("Accessibility", font_size=sp(16), bold=True, color=accent))
        access_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        large_text_btn = ToggleButton(
            text="Larger text",
            state="down" if current.accessibility_large_text else "normal",
            background_color=list(success),
            color=list(button_text),
            font_size=sp(11) * font_scale,
        )
        high_contrast_btn = ToggleButton(
            text="High contrast",
            state="down" if current.accessibility_high_contrast else "normal",
            background_color=list(secondary),
            color=list(button_text),
            font_size=sp(11) * font_scale,
        )
        reduce_visual_btn = ToggleButton(
            text="Reduce visuals",
            state="down" if current.accessibility_reduce_visual_load else "normal",
            background_color=list(neutral),
            color=list(button_text),
            font_size=sp(11) * font_scale,
        )
        access_row.add_widget(large_text_btn)
        access_row.add_widget(high_contrast_btn)
        access_row.add_widget(reduce_visual_btn)
        c.add_widget(access_row)

        def _apply_accessibility(_):
            self._set_accessibility(
                large_text=large_text_btn.state == "down",
                high_contrast=high_contrast_btn.state == "down",
                reduce_visual_load=reduce_visual_btn.state == "down",
            )

        large_text_btn.bind(on_release=_apply_accessibility)
        high_contrast_btn.bind(on_release=_apply_accessibility)
        reduce_visual_btn.bind(on_release=_apply_accessibility)

        # -- Updates --
        c.add_widget(Widget(size_hint_y=None, height=dp(12)))
        c.add_widget(_make_label("Updates", font_size=sp(16), bold=True, color=accent))
        auto_check_btn = ToggleButton(
            text="Check at startup",
            state="down" if current.check_updates_at_startup else "normal",
            background_color=list(neutral),
            color=list(button_text),
            font_size=sp(11) * font_scale,
        )
        check_now_btn = Button(
            text="Check now",
            background_color=list(secondary),
            color=list(button_text),
            font_size=sp(11) * font_scale,
        )
        updates_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        updates_row.add_widget(auto_check_btn)
        updates_row.add_widget(check_now_btn)
        c.add_widget(updates_row)

        def _set_auto_check_updates(_):
            conf = cfg.load_config()
            conf.check_updates_at_startup = auto_check_btn.state == "down"
            cfg.save_config(conf)

        def _check_updates_now(_):
            self._show_msg("Checking...", "Checking for updates...")
            # TODO: Implement actual update checking
            self._show_msg("Up to date", "You are running the latest version.")

        auto_check_btn.bind(on_release=_set_auto_check_updates)
        check_now_btn.bind(on_release=_check_updates_now)

        # -- Data management --
        c.add_widget(Widget(size_hint_y=None, height=dp(12)))
        c.add_widget(_make_label("Data Management", font_size=sp(16), bold=True, color=accent))
        del_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))

        def _delete_results(_):
            home = self.manager.get_screen("home")
            count = len(db.list_assessments(home.conn, limit=9999))
            if count == 0:
                self._show_msg("No Data", "No assessment results.")
                return
            db.delete_all_assessments(home.conn)
            self._show_msg("Deleted", f"Deleted {count} assessment result(s).")

        def _delete_tasks(_):
            home = self.manager.get_screen("home")
            count = len(db.list_tasks(home.conn))
            if count == 0:
                self._show_msg("No Data", "No tasks.")
                return
            db.delete_all_tasks(home.conn)
            self._show_msg("Deleted", f"Deleted {count} task(s) and sessions.")

        btn_del_tasks = Button(
            text="Delete all tasks",
            font_size=sp(11) * font_scale,
            background_color=list(danger),
            color=list(button_text),
        )
        btn_del_tasks.bind(on_release=_delete_tasks)
        del_row.add_widget(btn_del_tasks)
        btn_del_results = Button(
            text="Delete test results",
            font_size=sp(11) * font_scale,
            background_color=list(danger),
            color=list(button_text),
        )
        btn_del_results.bind(on_release=_delete_results)
        del_row.add_widget(btn_del_results)
        c.add_widget(del_row)

        # -- Browse & delete individual entries --
        c.add_widget(Widget(size_hint_y=None, height=dp(8)))
        c.add_widget(_make_label(
            "Browse & Delete Entries", font_size=sp(16), bold=True, color=accent,
        ))
        browse_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        for tbl_label, tbl_key in [("Tasks", "tasks"), ("Tests", "assessments")]:
            btn = Button(
                text=f"Browse {tbl_label}",
                font_size=sp(11) * font_scale,
                background_color=list(neutral),
                color=list(button_text),
            )
            btn.bind(on_release=lambda _, k=tbl_key: self._browse(k))
            browse_row.add_widget(btn)
        c.add_widget(browse_row)
        self._browse_box = BoxLayout(
            orientation="vertical", size_hint_y=None, spacing=dp(2),
        )
        self._browse_box.bind(minimum_height=self._browse_box.setter("height"))
        c.add_widget(self._browse_box)

        # -- Reset --
        c.add_widget(Widget(size_hint_y=None, height=dp(16)))
        reset_btn = Button(
            text="Reset to Default",
            size_hint_y=None,
            height=dp(44),
            background_color=list(danger),
            color=list(button_text),
            font_size=sp(14) * font_scale,
        )
        reset_btn.bind(on_release=lambda _: self._reset())
        c.add_widget(reset_btn)
        c.add_widget(Widget(size_hint_y=None, height=dp(20)))

    def _sync(self, provider):
        result = cfg.set_cloud_sync(provider)
        if result is None:
            self._show_msg("Not Found", f"Could not find {provider} folder.")
            return
        self._db_label.text = result.db_path or ""
        self._reconnect()
        self._show_msg("Sync Configured", f"Database: {result.db_path}")

    def _set_custom(self):
        p = self._path_input.text.strip()
        if p:
            result = cfg.set_db_path(p)
            self._db_label.text = result.db_path or ""
            self._reconnect()
            self._show_msg("Path Set", f"Database: {result.db_path}")

    def _reset(self):
        cfg.reset_db_path()
        new_path = cfg.get_db_path()
        self._db_label.text = f"{new_path} (default)"
        self._reconnect()

    def _set_theme(self, mode: str):
        _run_ui_action(
            lambda: self._apply_theme(mode),
            title="Appearance update failed",
            prefix="Could not apply the selected appearance mode.",
        )

    def _apply_theme(self, mode: str) -> None:
        cfg.set_theme_mode(mode)
        app = App.get_running_app()
        if app is not None:
            app.reload_palette()
        self._refresh_home_runtime_state()
        Clock.schedule_once(lambda dt: self.on_enter(), 0)

    def _set_timer_cycle_mode(self, mode: str):
        _run_ui_action(
            lambda: self._apply_timer_cycle_mode(mode),
            title="Timer mode update failed",
            prefix="Could not update timer cycle mode.",
        )

    def _apply_timer_cycle_mode(self, mode: str) -> None:
        cfg.set_timer_cycle_mode(mode)
        home = self.manager.get_screen("home")
        if mode == TimerCycleMode.AUTO.value:
            home.nudge_text = (
                "Auto cycle enabled: focus and break sessions will chain "
                "until you press Stop."
            )
        else:
            home.nudge_text = (
                "Manual cycle enabled: start each focus and break session yourself."
            )

    def _set_accessibility(self, *, large_text: bool, high_contrast: bool, reduce_visual_load: bool):
        _run_ui_action(
            lambda: self._apply_accessibility(
                large_text=large_text,
                high_contrast=high_contrast,
                reduce_visual_load=reduce_visual_load,
            ),
            title="Accessibility update failed",
            prefix="Could not apply accessibility settings.",
        )

    def _apply_accessibility(
        self,
        *,
        large_text: bool,
        high_contrast: bool,
        reduce_visual_load: bool,
    ) -> None:
        cfg.set_accessibility_options(
            large_text=large_text,
            high_contrast=high_contrast,
            reduce_visual_load=reduce_visual_load,
        )
        app = App.get_running_app()
        if app is not None:
            app.reload_palette()
        self._refresh_home_runtime_state()
        Clock.schedule_once(lambda dt: self.on_enter(), 0)

    def _refresh_home_runtime_state(self):
        if self.manager is None:
            return
        home = self.manager.get_screen("home")
        if home.conn is None:
            home.conn = db.get_connection()
        home._refresh_profile_ui()
        home.nudge_text = personalised_nudge(get_nudge(), home._profile())
        app = App.get_running_app()
        if app is not None and app.reduce_visual_load:
            fallback = home._make_fallback_banner()
            home._set_banner(fallback)
        else:
            threading.Thread(target=home._fetch_banner, daemon=True).start()

    def _reconnect(self):
        home = self.manager.get_screen("home")
        if home.conn:
            home.conn.close()
        home.conn = db.get_connection()

    def _browse(self, table):
        """Populate the browse box with entries from *table*."""
        app = App.get_running_app()
        text = list(app.text_color) if app else list(_TEXT)
        muted = list(app.muted_color) if app else list(_MUTED)
        danger = list(app.danger_button_color) if app else [0.55, 0.35, 0.35, 1]
        button_text = list(app.button_text_color) if app else list(_BUTTON_TEXT)

        box = self._browse_box
        box.clear_widgets()
        home = self.manager.get_screen("home")
        if table == "tasks":
            tasks = db.list_tasks(home.conn)
            if not tasks:
                box.add_widget(_make_label("  No tasks.", color=muted))
                return
            for t in tasks:
                row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(4))
                row.add_widget(_make_label(
                    f"#{t.id} [{t.status.value}] {t.title}",
                    font_size=sp(11), color=text,
                ))
                del_btn = Button(
                    text="Del", size_hint_x=None, width=dp(50),
                    font_size=sp(10), background_color=danger, color=button_text,
                )
                del_btn.bind(
                    on_release=lambda _, tid=t.id: self._delete_entry("tasks", tid)
                )
                row.add_widget(del_btn)
                box.add_widget(row)
        elif table == "assessments":
            results = db.list_assessments(home.conn, limit=50)
            if not results:
                box.add_widget(_make_label("  No assessments.", color=muted))
                return
            for r in results:
                taken = r.taken_at.strftime("%Y-%m-%d %H:%M")
                row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(4))
                row.add_widget(_make_label(
                    f"#{r.id} {r.assessment_type.value} {r.score}/{r.max_score} ({taken})",
                    font_size=sp(11), color=text,
                ))
                del_btn = Button(
                    text="Del", size_hint_x=None, width=dp(50),
                    font_size=sp(10), background_color=danger, color=button_text,
                )
                del_btn.bind(
                    on_release=lambda _, rid=r.id: self._delete_entry("assessments", rid)
                )
                row.add_widget(del_btn)
                box.add_widget(row)

    def _delete_entry(self, table, entry_id):
        home = self.manager.get_screen("home")
        if table == "tasks":
            db.delete_task(home.conn, entry_id)
        elif table == "assessments":
            db.delete_assessment(home.conn, entry_id)
        self._browse(table)

    @staticmethod
    def _show_msg(title, text):
        app = App.get_running_app()
        fg = list(app.text_color) if app else list(_TEXT)
        accent = list(app.accent_color) if app else list(_ACCENT)
        content = BoxLayout(orientation="vertical", padding=10, spacing=10)
        content.add_widget(Label(text=text, font_size=sp(13), color=fg))
        btn = Button(
            text="OK", size_hint_y=None, height=dp(44),
            background_color=accent, color=fg,
        )
        popup = Popup(title=title, content=content, size_hint=(0.85, 0.3))
        btn.bind(on_release=lambda _: popup.dismiss())
        content.add_widget(btn)
        popup.open()


# ---------------------------------------------------------------------------
# Help -> How to Use
# ---------------------------------------------------------------------------


class HowToScreen(ScrollScreen):
    """Displays README.md content."""

    def on_enter(self):
        c = self.ids.content
        c.clear_widgets()
        _render_markdown(c, _find_md("README.md"))


# ---------------------------------------------------------------------------
# Help -> The Science
# ---------------------------------------------------------------------------


class ScienceScreen(ScrollScreen):
    """Displays SCIENCE.md content."""

    def on_enter(self):
        c = self.ids.content
        c.clear_widgets()
        _render_markdown(c, _find_md("SCIENCE.md"))


# ---------------------------------------------------------------------------
# Help -> About
# ---------------------------------------------------------------------------


class AboutScreen(ScrollScreen):
    """Simple about screen."""

    def on_enter(self):
        c = self.ids.content
        c.clear_widgets()
        c.add_widget(_make_label("Momentum", font_size=sp(22), bold=True, color=_ACCENT))
        c.add_widget(Widget(size_hint_y=None, height=dp(8)))
        c.add_widget(_make_label("Version 0.3.0"))
        c.add_widget(Widget(size_hint_y=None, height=dp(12)))
        c.add_widget(_make_label(
            "A gentle tool to help people with executive dysfunction "
            "get back on track, one small step at a time."
        ))
        c.add_widget(Widget(size_hint_y=None, height=dp(12)))
        c.add_widget(_make_label(
            "Momentum is a supportive tool, not a replacement for "
            "professional treatment. If you are experiencing depression, "
            "please reach out to a healthcare provider."
        ))
        c.add_widget(Widget(size_hint_y=None, height=dp(20)))
        c.add_widget(_make_label(
            "Created by Robin \u00d6berg", font_size=sp(15), bold=True, color=_ACCENT,
        ))
        c.add_widget(_make_label(
            "Data Scientist, MSc Social Anthropology, "
            "MSc Applied Cultural Analysis.", color=_MUTED,
        ))
        c.add_widget(_make_label("robinoberg@live.com", color=_MUTED, font_size=sp(12)))
        c.add_widget(Widget(size_hint_y=None, height=dp(16)))
        c.add_widget(_make_label(
            "Copyright \u00a9 2026 Robin \u00d6berg.", color=_MUTED,
        ))
        c.add_widget(_make_label(
            "Licensed under the MIT License.", color=_MUTED,
        ))
        c.add_widget(Widget(size_hint_y=None, height=dp(8)))
        c.add_widget(_make_label(
            "https://github.com/yidaki53/momentum", color=_MUTED, font_size=sp(12),
        ))


# ---------------------------------------------------------------------------
# Help menu
# ---------------------------------------------------------------------------


class HelpMenuScreen(Screen):
    """Sub-menu: How to Use, The Science, About."""

    def go_howto(self):
        _run_ui_action(
            lambda: self._go("howto"),
            prefix="Could not open How to Use.",
        )

    def go_science(self):
        _run_ui_action(
            lambda: self._go("science"),
            prefix="Could not open The Science.",
        )

    def go_about(self):
        _run_ui_action(
            lambda: self._go("about"),
            prefix="Could not open About.",
        )

    def _go(self, target: str) -> None:
        self.manager.transition = NoTransition()
        self.manager.current = target


# ---------------------------------------------------------------------------
# Tests menu
# ---------------------------------------------------------------------------


class TestsMenuScreen(Screen):
    """Sub-menu: BDEFS, BIS/BAS, Stroop, View Results."""

    def go_bdefs(self):
        _run_ui_action(
            lambda: self._go("bdefs"),
            prefix="Could not open the BDEFS assessment.",
        )
    def go_bisbas(self):
        _run_ui_action(
            lambda: self._go("bisbas"),
            prefix="Could not open the BIS/BAS profile.",
        )

    def go_stroop(self):
        _run_ui_action(
            lambda: self._go("stroop"),
            prefix="Could not open the Stroop test.",
        )

    def go_results(self):
        _run_ui_action(
            lambda: self._go("results"),
            prefix="Could not open test results.",
        )

    def _go(self, target: str) -> None:
        self.manager.transition = NoTransition()
        self.manager.current = target


# ---------------------------------------------------------------------------
# Tests -> BDEFS
# ---------------------------------------------------------------------------


class BdefsScreen(Screen):
    """BDEFS executive function self-assessment."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vars: dict[str, list[tuple[str, list[ToggleButton]]]] = {}

    def on_enter(self):
        content = self.ids.bdefs_content
        content.clear_widgets()
        self._vars = {}

        # Instruction text
        content.add_widget(_make_label(
            BDEFS_INSTRUCTIONS, font_size=sp(12), color=_MUTED,
        ))
        content.add_widget(Widget(size_hint_y=None, height=dp(8)))

        content.add_widget(_make_label(
            "Rate each statement\n1 = Never   2 = Sometimes   3 = Often   4 = Very Often",
            font_size=sp(13), color=_MUTED,
        ))

        q_index = 0
        for domain, questions in BDEFS_QUESTIONS.items():
            content.add_widget(Widget(size_hint_y=None, height=dp(8)))
            content.add_widget(_make_label(domain, font_size=sp(16), bold=True, color=_ACCENT))

            domain_groups: list[tuple[str, list[ToggleButton]]] = []
            for q in questions:
                content.add_widget(_make_label(q, font_size=sp(13)))
                group_name = f"bdefs_q{q_index}"
                btn_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(4),
                                    padding=[dp(8), 0])
                buttons: list[ToggleButton] = []
                for val in range(1, 5):
                    tb = ToggleButton(
                        text=str(val), group=group_name, font_size=sp(14),
                        background_color=_ACCENT if val == 1 else (0.25, 0.25, 0.25, 1),
                    )
                    if val == 1:
                        tb.state = "down"
                    tb.bind(state=self._make_toggle_cb())
                    btn_row.add_widget(tb)
                    buttons.append(tb)
                content.add_widget(btn_row)
                domain_groups.append((group_name, buttons))
                q_index += 1
            self._vars[domain] = domain_groups

        content.add_widget(Widget(size_hint_y=None, height=dp(12)))
        submit = Button(
            text="Submit", size_hint_y=None, height=dp(48),
            background_color=_ACCENT, font_size=sp(16), bold=True,
        )
        submit.bind(on_release=lambda _: self._submit())
        content.add_widget(submit)
        content.add_widget(Widget(size_hint_y=None, height=dp(20)))

    @staticmethod
    def _make_toggle_cb():
        def cb(inst, state):
            inst.background_color = list(_ACCENT) if state == "down" else [0.25, 0.25, 0.25, 1]
        return cb

    def _get_val(self, buttons):
        for i, b in enumerate(buttons):
            if b.state == "down":
                return i + 1
        return 1

    def _submit(self):
        answers = {}
        for domain, groups in self._vars.items():
            answers[domain] = [self._get_val(btns) for _, btns in groups]

        home = self.manager.get_screen("home")
        create_model = score_bdefs(answers)
        saved = db.save_assessment(home.conn, create_model)

        msg = f"Total: {saved.score}/{saved.max_score}\n\n"
        for d, s in saved.domain_scores.items():
            n_qs = len(BDEFS_QUESTIONS[d])
            msg += f"{d}: {s}/{n_qs * 4}\n"
        msg += f"\n{interpret_bdefs(saved.score, saved.max_score)}"

        scroll = ScrollView(do_scroll_x=False)
        inner = BoxLayout(orientation="vertical", spacing=6, padding=10,
                          size_hint_y=None)
        inner.bind(minimum_height=inner.setter("height"))
        inner.add_widget(_make_label(msg, font_size=sp(13)))
        inner.add_widget(Widget(size_hint_y=None, height=dp(8)))
        inner.add_widget(_make_label(
            "Domain Advice", font_size=sp(15), bold=True, color=_ACCENT,
        ))
        for d, s in saved.domain_scores.items():
            n_qs = len(BDEFS_QUESTIONS[d])
            advice = domain_advice(d, s, n_qs * 4)
            inner.add_widget(_make_label(d, font_size=sp(13), bold=True, color=_ACCENT))
            inner.add_widget(_make_label(advice, font_size=sp(11), color=_MUTED))
        scroll.add_widget(inner)
        content = BoxLayout(orientation="vertical", padding=10, spacing=10)
        content.add_widget(scroll)
        close = Button(text="Close", size_hint_y=None, height=dp(44))
        popup = Popup(title="Assessment Result", content=content, size_hint=(0.9, 0.75))
        close.bind(on_release=lambda _: popup.dismiss())
        content.add_widget(close)
        popup.open()

        self.manager.transition = NoTransition()
        self.manager.current = "tests_menu"


# ---------------------------------------------------------------------------
# Tests -> BIS/BAS
# ---------------------------------------------------------------------------


class BisbasScreen(Screen):
    """BIS/BAS motivational-style self-assessment."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vars: dict[str, list[list[ToggleButton]]] = {}

    def on_enter(self):
        content = self.ids.bisbas_content
        content.clear_widgets()
        self._vars = {}

        content.add_widget(_make_label(
            BISBAS_INSTRUCTIONS, font_size=sp(12), color=_MUTED,
        ))
        content.add_widget(Widget(size_hint_y=None, height=dp(8)))
        content.add_widget(_make_label(
            "Rate each statement\n1 = Very false   2 = Somewhat false   "
            "3 = Somewhat true   4 = Very true",
            font_size=sp(13), color=_MUTED,
        ))

        q_index = 0
        for domain, questions in BISBAS_QUESTIONS.items():
            content.add_widget(Widget(size_hint_y=None, height=dp(8)))
            content.add_widget(_make_label(domain, font_size=sp(16), bold=True, color=_ACCENT))

            domain_groups: list[list[ToggleButton]] = []
            for q in questions:
                content.add_widget(_make_label(q, font_size=sp(13)))
                group_name = f"bisbas_q{q_index}"
                btn_row = BoxLayout(
                    size_hint_y=None, height=dp(40), spacing=dp(4), padding=[dp(8), 0]
                )
                buttons: list[ToggleButton] = []
                for val in range(1, 5):
                    tb = ToggleButton(
                        text=str(val),
                        group=group_name,
                        font_size=sp(14),
                        background_color=_ACCENT if val == 1 else (0.25, 0.25, 0.25, 1),
                    )
                    if val == 1:
                        tb.state = "down"
                    tb.bind(state=self._make_toggle_cb())
                    btn_row.add_widget(tb)
                    buttons.append(tb)
                content.add_widget(btn_row)
                domain_groups.append(buttons)
                q_index += 1
            self._vars[domain] = domain_groups

        content.add_widget(Widget(size_hint_y=None, height=dp(12)))
        submit = Button(
            text="Submit", size_hint_y=None, height=dp(48),
            background_color=_ACCENT, font_size=sp(16), bold=True,
        )
        submit.bind(on_release=lambda _: self._submit())
        content.add_widget(submit)
        content.add_widget(Widget(size_hint_y=None, height=dp(20)))

    @staticmethod
    def _make_toggle_cb():
        def cb(inst, state):
            inst.background_color = list(_ACCENT) if state == "down" else [0.25, 0.25, 0.25, 1]
        return cb

    @staticmethod
    def _get_val(buttons: list[ToggleButton]) -> int:
        for i, b in enumerate(buttons):
            if b.state == "down":
                return i + 1
        return 1

    def _submit(self):
        answers: dict[str, list[int]] = {}
        for domain, groups in self._vars.items():
            answers[domain] = [self._get_val(btns) for btns in groups]

        home = self.manager.get_screen("home")
        create_model = score_bisbas(answers)
        saved = db.save_assessment(home.conn, create_model)

        msg = f"Total: {saved.score}/{saved.max_score}\n\n"
        for d, s in saved.domain_scores.items():
            max_domain = len(BISBAS_QUESTIONS.get(d, [])) * 4
            msg += f"{d}: {s}/{max_domain if max_domain else 1}\n"

        scroll = ScrollView(do_scroll_x=False)
        inner = BoxLayout(orientation="vertical", spacing=6, padding=10, size_hint_y=None)
        inner.bind(minimum_height=inner.setter("height"))
        inner.add_widget(_make_label(msg, font_size=sp(13)))
        inner.add_widget(Widget(size_hint_y=None, height=dp(8)))
        inner.add_widget(_make_label(
            "Domain Advice", font_size=sp(15), bold=True, color=_ACCENT,
        ))
        for d, s in saved.domain_scores.items():
            max_domain = len(BISBAS_QUESTIONS.get(d, [])) * 4
            advice = bisbas_domain_advice(d, s, max_domain if max_domain else 1)
            inner.add_widget(_make_label(d, font_size=sp(13), bold=True, color=_ACCENT))
            inner.add_widget(_make_label(advice, font_size=sp(11), color=_MUTED))
        inner.add_widget(Widget(size_hint_y=None, height=dp(6)))
        inner.add_widget(_make_label(
            "Reference lines in the chart are guidance anchors, not diagnostic cutoffs.",
            font_size=sp(11),
            color=_MUTED,
        ))
        inner.add_widget(_make_label(
            bisbas_bespoke_guidance(saved.domain_scores),
            font_size=sp(12),
            color=_MUTED,
        ))
        inner.add_widget(_make_label(
            interpret_bisbas(saved.score, saved.max_score, saved.domain_scores),
            font_size=sp(12),
            color=_MUTED,
        ))
        scroll.add_widget(inner)

        content = BoxLayout(orientation="vertical", padding=10, spacing=10)
        chart_funcs = _get_chart_funcs()
        if chart_funcs is not None:
            try:
                _bdefs_radar, _bdefs_timeseries, bisbas_profile_bars = chart_funcs
                bisbas_img = bisbas_profile_bars(
                    saved,
                    title="BIS/BAS Motivational Profile",
                )
                bisbas_core = _pil_to_kivy_image(bisbas_img)
                content.add_widget(KivyImage(
                    texture=bisbas_core.texture,
                    size_hint_y=None,
                    height=dp(210),
                    allow_stretch=True,
                    keep_ratio=True,
                ))
            except Exception:
                log.debug("BIS/BAS chart popup render failed", exc_info=True)
        content.add_widget(scroll)
        close = Button(text="Close", size_hint_y=None, height=dp(44))
        popup = Popup(title="BIS/BAS Result", content=content, size_hint=(0.9, 0.75))
        close.bind(on_release=lambda _: popup.dismiss())
        content.add_widget(close)
        popup.open()

        home._refresh_profile_ui()
        home.nudge_text = personalised_nudge(get_nudge(), home._profile())

        self.manager.transition = NoTransition()
        self.manager.current = "tests_menu"


# ---------------------------------------------------------------------------
# Tests -> Stroop
# ---------------------------------------------------------------------------


class StroopScreen(Screen):
    """Stroop colour-word test."""

    word_text = StringProperty("")
    word_color = ListProperty([1, 1, 1, 1])
    progress_text = StringProperty("")
    feedback_text = StringProperty("")

    _colour_map = {
        "red": [0.878, 0.376, 0.376, 1],
        "green": [0.376, 0.753, 0.376, 1],
        "blue": [0.376, 0.565, 0.878, 1],
        "yellow": [0.878, 0.816, 0.376, 1],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._trials = []
        self._state: dict = {}

    def on_enter(self):
        self._trials = generate_stroop_trials()
        self._state = {"idx": 0, "correct": 0, "total_time": 0.0, "per_trial": []}
        self.feedback_text = ""
        # Show instruction popup before starting
        content = BoxLayout(orientation="vertical", padding=10, spacing=10)
        content.add_widget(_make_label(STROOP_INSTRUCTIONS, font_size=sp(12), color=_MUTED))
        start_btn = Button(text="Start", size_hint_y=None, height=dp(44),
                           background_color=list(_ACCENT))
        popup = Popup(title="Stroop Colour-Word Test", content=content,
                      size_hint=(0.9, 0.55), auto_dismiss=False)

        def _begin(_):
            popup.dismiss()
            self._show_trial()

        start_btn.bind(on_release=_begin)
        content.add_widget(start_btn)
        popup.open()

    def _show_trial(self):
        idx = self._state["idx"]
        if idx >= len(self._trials):
            self._finish()
            return
        trial = self._trials[idx]
        self.word_text = trial.word.upper()
        self.word_color = self._colour_map[trial.ink_colour]
        self.progress_text = f"Trial {idx + 1} of {len(self._trials)}"
        self.feedback_text = ""
        self._state["t0"] = _time.time()
        inp = self.ids.stroop_input
        inp.text = ""
        inp.focus = True

    def on_answer(self, text):
        idx = self._state["idx"]
        if idx >= len(self._trials):
            return
        elapsed = _time.time() - self._state["t0"]
        trial = self._trials[idx]
        answer = text.strip().lower()
        correct = answer == trial.ink_colour
        if correct:
            self._state["correct"] += 1
            self.feedback_text = "Correct!"
        else:
            self.feedback_text = f"Wrong -- it was {trial.ink_colour}"
        self._state["total_time"] += elapsed
        self._state["per_trial"].append((correct, elapsed))
        self._state["idx"] += 1
        Clock.schedule_once(lambda dt: self._show_trial(), 0.8)

    def _finish(self):
        result = StroopResult(
            trials=len(self._trials),
            correct=self._state["correct"],
            total_time_s=self._state["total_time"],
            per_trial=self._state["per_trial"],
        )
        create_model = score_stroop(result)
        home = self.manager.get_screen("home")
        saved = db.save_assessment(home.conn, create_model)
        avg_ms = int(result.avg_time_s * 1000)
        msg = (
            f"Score: {saved.score}/{saved.max_score}\n"
            f"Accuracy: {result.accuracy_pct:.0f}%\n"
            f"Avg response: {avg_ms} ms\n\n"
            f"{interpret_stroop(saved.score, saved.max_score, avg_ms)}"
        )
        self.word_text = ""
        self.progress_text = "Complete!"
        self.feedback_text = ""

        content = BoxLayout(orientation="vertical", padding=10, spacing=10)
        content.add_widget(_make_label(msg, font_size=sp(13)))
        close = Button(text="Close", size_hint_y=None, height=dp(44))
        popup = Popup(title="Stroop Result", content=content, size_hint=(0.9, 0.5))

        def _close(_):
            popup.dismiss()
            self.manager.transition = NoTransition()
            self.manager.current = "tests_menu"

        close.bind(on_release=_close)
        content.add_widget(close)
        popup.open()


# ---------------------------------------------------------------------------
# Tests -> View Results
# ---------------------------------------------------------------------------


class ResultsScreen(ScrollScreen):
    """Past assessment results with charts."""

    def on_enter(self):
        c = self.ids.content
        c.clear_widgets()

        home = self.manager.get_screen("home")
        if home.conn is None:
            home.conn = db.get_connection()
        results = db.list_assessments(home.conn, limit=50)

        # Guide text
        c.add_widget(_make_label(
            RESULTS_GUIDE, font_size=sp(11), color=_MUTED,
        ))
        c.add_widget(Widget(size_hint_y=None, height=dp(8)))

        if not results:
            c.add_widget(_make_label(
                "No assessment results yet.\n\nTake a test from the Tests menu.",
                font_size=sp(15), color=_MUTED,
            ))
            return

        # -- Charts --
        bdefs_results = [r for r in results if r.assessment_type == AssessmentType.BDEFS]
        bisbas_results = [r for r in results if r.assessment_type == AssessmentType.BISBAS]
        chart_funcs = _get_chart_funcs()
        charts_available = chart_funcs is not None

        if charts_available:
            bdefs_radar, bdefs_timeseries, bisbas_profile_bars = chart_funcs
        else:
            c.add_widget(_make_label(
                "Charts are unavailable on this device/runtime; showing text results only.",
                font_size=sp(11),
                color=_MUTED,
            ))
            c.add_widget(Widget(size_hint_y=None, height=dp(6)))

        if bdefs_results and charts_available:
            try:
                latest_r = bdefs_results[0]  # most recent first
                prev_r = bdefs_results[1] if len(bdefs_results) > 1 else None
                radar_img = bdefs_radar(
                    latest=latest_r, previous=prev_r,
                    title="Latest Executive Function Profile",
                )
                core_img = _pil_to_kivy_image(radar_img)
                radar_widget = KivyImage(
                    texture=core_img.texture, size_hint_y=None,
                    height=dp(300), allow_stretch=True, keep_ratio=True,
                )
                c.add_widget(radar_widget)
            except Exception:
                log.debug("Radar chart failed", exc_info=True)

            try:
                ts_img = bdefs_timeseries(results)
                if ts_img is not None:
                    core_ts = _pil_to_kivy_image(ts_img)
                    ts_widget = KivyImage(
                        texture=core_ts.texture, size_hint_y=None,
                        height=dp(180), allow_stretch=True, keep_ratio=True,
                    )
                    c.add_widget(ts_widget)
            except Exception:
                log.debug("Timeseries chart failed", exc_info=True)
        if bisbas_results and charts_available:
            latest_bisbas = bisbas_results[0]
            try:
                bisbas_img = bisbas_profile_bars(
                    latest_bisbas,
                    title="Latest BIS/BAS Motivational Profile",
                )
                bisbas_core = _pil_to_kivy_image(bisbas_img)
                bisbas_widget = KivyImage(
                    texture=bisbas_core.texture,
                    size_hint_y=None,
                    height=dp(220),
                    allow_stretch=True,
                    keep_ratio=True,
                )
                c.add_widget(bisbas_widget)
            except Exception:
                log.debug("BIS/BAS chart failed", exc_info=True)
            c.add_widget(_make_label(
                "BIS/BAS reference lines are guidance anchors, not diagnostic cutoffs.",
                font_size=sp(11),
                color=_MUTED,
            ))
            c.add_widget(_make_label(
                bisbas_bespoke_guidance(latest_bisbas.domain_scores),
                font_size=sp(12),
                color=_MUTED,
            ))

        # -- Textual results list --
        c.add_widget(_make_label(
            f"{len(results)} assessment(s)", font_size=sp(17), bold=True, color=_ACCENT,
        ))

        for r in results:
            taken = r.taken_at.strftime("%Y-%m-%d %H:%M")
            type_str = r.assessment_type.value.upper()

            c.add_widget(Widget(size_hint_y=None, height=dp(8)))
            c.add_widget(_make_label(
                f"{type_str}  --  {taken}", font_size=sp(14), bold=True, color=_ACCENT,
            ))
            c.add_widget(_make_label(f"Score: {r.score}/{r.max_score}", font_size=sp(13)))

            if r.assessment_type == AssessmentType.BDEFS:
                for d, s in r.domain_scores.items():
                    n_qs = len(BDEFS_QUESTIONS.get(d, []))
                    max_d = n_qs * 4 if n_qs else 1
                    c.add_widget(_make_label(f"  {d}: {s}", font_size=sp(12), color=_MUTED))
                    advice = domain_advice(d, s, max_d)
                    c.add_widget(_make_label(
                        f"    {advice}", font_size=sp(11), color=_MUTED,
                    ))
                c.add_widget(_make_label(
                    interpret_bdefs(r.score, r.max_score),
                    font_size=sp(12), color=_MUTED,
                ))
            elif r.assessment_type == AssessmentType.STROOP:
                avg_ms = r.domain_scores.get("avg_time_ms", 0)
                c.add_widget(_make_label(
                    f"  Avg response: {avg_ms} ms", font_size=sp(12), color=_MUTED,
                ))
                c.add_widget(_make_label(
                    interpret_stroop(r.score, r.max_score, avg_ms),
                    font_size=sp(12), color=_MUTED,
                ))
            elif r.assessment_type == AssessmentType.BISBAS:
                for d, s in r.domain_scores.items():
                    max_domain = len(BISBAS_QUESTIONS.get(d, [])) * 4
                    max_domain = max_domain if max_domain else 1
                    c.add_widget(_make_label(
                        f"  {d}: {s}/{max_domain}", font_size=sp(12), color=_MUTED,
                    ))
                    advice = bisbas_domain_advice(d, s, max_domain)
                    c.add_widget(_make_label(
                        f"    {advice}", font_size=sp(11), color=_MUTED,
                    ))
                c.add_widget(_make_label(
                    interpret_bisbas(r.score, r.max_score, r.domain_scores),
                    font_size=sp(12), color=_MUTED,
                ))
                c.add_widget(_make_label(
                    bisbas_bespoke_guidance(r.domain_scores),
                    font_size=sp(12), color=_MUTED,
                ))

        c.add_widget(Widget(size_hint_y=None, height=dp(20)))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class MomentumApp(App):
    """Kivy application entry point."""

    title = "Momentum"
    bg_color = ListProperty(list(_PALETTE["bg"]))
    text_color = ListProperty(list(_PALETTE["text"]))
    muted_color = ListProperty(list(_PALETTE["muted"]))
    accent_color = ListProperty(list(_PALETTE["accent"]))
    button_text_color = ListProperty(list(_PALETTE["button_text"]))
    toolbar_color = ListProperty(list(_PALETTE["toolbar"]))
    input_bg_color = ListProperty(list(_PALETTE["input_bg"]))
    timer_color = ListProperty(list(_PALETTE["timer"]))
    neutral_button_color = ListProperty(list(_PALETTE["neutral_button"]))
    success_button_color = ListProperty(list(_PALETTE["success_button"]))
    secondary_button_color = ListProperty(list(_PALETTE["secondary_button"]))
    danger_button_color = ListProperty(list(_PALETTE["danger_button"]))
    font_scale = NumericProperty(1.0)
    reduce_visual_load = BooleanProperty(False)

    def reload_palette(self):
        """Reload theme/accessibility config and update bound KV properties."""
        global _PALETTE, _APP_CFG, _ACCENT, _TEXT, _MUTED, _BG, _BUTTON_TEXT
        _APP_CFG = cfg.load_config()
        _PALETTE = _resolve_palette()
        _ACCENT = _PALETTE["accent"]
        _TEXT = _PALETTE["text"]
        _MUTED = _PALETTE["muted"]
        _BG = _PALETTE["bg"]
        _BUTTON_TEXT = _PALETTE["button_text"]

        self.bg_color = list(_PALETTE["bg"])
        self.text_color = list(_PALETTE["text"])
        self.muted_color = list(_PALETTE["muted"])
        self.accent_color = list(_PALETTE["accent"])
        self.button_text_color = list(_PALETTE["button_text"])
        self.toolbar_color = list(_PALETTE["toolbar"])
        self.input_bg_color = list(_PALETTE["input_bg"])
        self.timer_color = list(_PALETTE["timer"])
        self.neutral_button_color = list(_PALETTE["neutral_button"])
        self.success_button_color = list(_PALETTE["success_button"])
        self.secondary_button_color = list(_PALETTE["secondary_button"])
        self.danger_button_color = list(_PALETTE["danger_button"])
        self.font_scale = 1.35 if _APP_CFG.accessibility_large_text else 1.0
        self.reduce_visual_load = _APP_CFG.accessibility_reduce_visual_load

    def build(self):
        self.reload_palette()
        Builder.load_string(KV)
        sm = ScreenManager()
        sm.add_widget(HomeScreen(name="home"))
        sm.add_widget(SettingsScreen(name="settings"))
        sm.add_widget(HelpMenuScreen(name="help_menu"))
        sm.add_widget(HowToScreen(name="howto"))
        sm.add_widget(ScienceScreen(name="science"))
        sm.add_widget(AboutScreen(name="about"))
        sm.add_widget(TestsMenuScreen(name="tests_menu"))
        sm.add_widget(BdefsScreen(name="bdefs"))
        sm.add_widget(BisbasScreen(name="bisbas"))
        sm.add_widget(StroopScreen(name="stroop"))
        sm.add_widget(ResultsScreen(name="results"))
        return sm

    def on_stop(self):
        home = self.root.get_screen("home")
        if home.conn:
            home.conn.close()


if __name__ == "__main__":
    MomentumApp().run()
