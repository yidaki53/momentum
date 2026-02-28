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
import sys
import threading
import time as _time
import urllib.request
from pathlib import Path

# Ensure the parent package is importable when running standalone on desktop
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.properties import (
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty,
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen, ScreenManager, SlideTransition
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.core.image import Image as CoreImage
from kivy.uix.image import Image as KivyImage
from kivy.uix.widget import Widget

from PIL import Image as PILImage, ImageDraw, ImageFont

from momentum import config as cfg
from momentum import db
from momentum.assessments import (
    BDEFS_INSTRUCTIONS,
    BDEFS_QUESTIONS,
    BDEFS_SCALE,
    RESULTS_GUIDE,
    STROOP_INSTRUCTIONS,
    StroopResult,
    generate_stroop_trials,
    interpret_bdefs,
    interpret_stroop,
    score_bdefs,
    score_stroop,
)
from momentum.charts import bdefs_radar, bdefs_timeseries
from momentum.encouragement import get_break_message, get_nudge
from momentum.models import (
    AssessmentResult,
    AssessmentType,
    FocusSessionCreate,
    TaskCreate,
    TaskStatus,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ACCENT = (0.416, 0.624, 0.710, 1)      # #6a9fb5
_TEXT = (0.878, 0.878, 0.878, 1)          # #e0e0e0
_MUTED = (0.71, 0.71, 0.71, 1)           # #b5b5b5
_BG = (0.169, 0.169, 0.169, 1)           # #2b2b2b

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
    defaults = dict(
        font_size=sp(14), color=_TEXT,
        size_hint_y=None, text_size=(None, None), halign="left", valign="top",
    )
    defaults.update(kw)
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
            clean = _clean_inline(s[2:])
            container.add_widget(_make_label(
                "  -- " + clean, font_size=sp(12), color=_MUTED,
            ))
        elif re.match(r"^[-*]{3,}$", s):
            # Horizontal rule
            container.add_widget(Widget(size_hint_y=None, height=dp(12)))
        else:
            container.add_widget(_make_label(_clean_inline(s)))


# ---------------------------------------------------------------------------
# Kivy UI definition (KV language)
# ---------------------------------------------------------------------------

KV = """
#:import get_color_from_hex kivy.utils.get_color_from_hex
#:import dp kivy.metrics.dp
#:import sp kivy.metrics.sp

# ---- Reusable styles ----

<AccentLabel@Label>:
    color: get_color_from_hex('#6a9fb5')
    font_size: sp(16)
    bold: True
    size_hint_y: None
    height: dp(32)
    text_size: self.width, None
    halign: 'left'

<DarkButton@Button>:
    background_color: get_color_from_hex('#6a9fb5')
    font_size: sp(14)
    size_hint_y: None
    height: dp(44)
    bold: True

# ---- Toolbar ----

<Toolbar>:
    size_hint_y: None
    height: dp(44)
    spacing: dp(2)
    padding: [dp(2), dp(2)]
    canvas.before:
        Color:
            rgba: get_color_from_hex('#1e1e1e')
        Rectangle:
            pos: self.pos
            size: self.size

    Button:
        text: 'Home'
        font_size: sp(12)
        bold: True
        background_color: get_color_from_hex('#6a9fb5')
        size_hint_x: 0.18
        on_release: root.go('home', 'right')
    Button:
        text: 'Settings'
        font_size: sp(11)
        background_color: get_color_from_hex('#3a3a3a')
        size_hint_x: 0.22
        on_release: root.go('settings', 'left')
    Button:
        text: 'Help'
        font_size: sp(12)
        background_color: get_color_from_hex('#3a3a3a')
        size_hint_x: 0.18
        on_release: root.go('help_menu', 'left')
    Button:
        text: 'Tests'
        font_size: sp(12)
        background_color: get_color_from_hex('#3a3a3a')
        size_hint_x: 0.18
        on_release: root.go('tests_menu', 'left')

# ---- Task row ----

<TaskRow>:
    size_hint_y: None
    height: dp(48)
    canvas.before:
        Color:
            rgba: get_color_from_hex('#333333')
        Rectangle:
            pos: self.pos
            size: self.size
    Label:
        text: root.icon
        size_hint_x: 0.1
        color: get_color_from_hex('#6a9fb5')
        font_size: sp(16)
    Label:
        text: root.title_text
        size_hint_x: 0.7
        text_size: self.size
        halign: 'left'
        valign: 'middle'
        color: get_color_from_hex('#e0e0e0')
        font_size: sp(14)
    Button:
        text: 'Done'
        size_hint_x: 0.2
        background_color: get_color_from_hex('#4a7a4a')
        on_release: root.on_complete()

# ---- Home screen ----

<HomeScreen>:
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: get_color_from_hex('#2b2b2b')
            Rectangle:
                pos: self.pos
                size: self.size

        Toolbar:

        # Nature image banner
        BoxLayout:
            id: banner_box
            size_hint_y: None
            height: dp(80)
            padding: [dp(8), dp(4)]

        Label:
            text: root.status_text
            size_hint_y: None
            height: dp(32)
            color: get_color_from_hex('#6a9fb5')
            font_size: sp(13)

        Label:
            text: 'Tasks'
            size_hint_y: None
            height: dp(28)
            color: get_color_from_hex('#6a9fb5')
            font_size: sp(18)
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
                background_color: get_color_from_hex('#6a9fb5')
                on_release: root.show_add_dialog()
            Button:
                text: 'Break down'
                background_color: get_color_from_hex('#5a8a5a')
                on_release: root.show_breakdown_dialog()

        Label:
            text: 'Timer'
            size_hint_y: None
            height: dp(28)
            color: get_color_from_hex('#6a9fb5')
            font_size: sp(18)
            bold: True
            halign: 'left'
            text_size: self.size
            padding: [dp(12), 0]

        Label:
            text: root.timer_display
            size_hint_y: None
            height: dp(50)
            color: get_color_from_hex('#e8c547')
            font_size: sp(36)
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
                text: 'Focus 15m'
                background_color: get_color_from_hex('#6a9fb5')
                font_size: sp(14)
                bold: True
                on_release: root.start_focus()
            Button:
                text: 'Break 5m'
                background_color: get_color_from_hex('#7a6a9f')
                font_size: sp(14)
                on_release: root.start_break()
            Button:
                text: 'Stop'
                background_color: get_color_from_hex('#9f6a6a')
                font_size: sp(14)
                on_release: root.stop_timer()

        Label:
            text: root.nudge_text
            size_hint_y: None
            height: dp(56)
            color: get_color_from_hex('#b5b5b5')
            font_size: sp(13)
            italic: True
            text_size: self.width - dp(24), None
            halign: 'center'

        Button:
            text: 'New encouragement'
            size_hint_y: None
            height: dp(40)
            background_color: get_color_from_hex('#5a5a7a')
            on_release: root.refresh_nudge()

        Widget:
            size_hint_y: None
            height: dp(4)

# ---- Scrollable content screen (reused by Settings, Science, Help, Results) ----

<ScrollScreen>:
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: get_color_from_hex('#2b2b2b')
            Rectangle:
                pos: self.pos
                size: self.size
        Toolbar:
        ScrollView:
            do_scroll_x: False
            BoxLayout:
                id: content
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                padding: [dp(16), dp(12)]
                spacing: dp(4)

# ---- Tests menu screen ----

<TestsMenuScreen>:
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: get_color_from_hex('#2b2b2b')
            Rectangle:
                pos: self.pos
                size: self.size
        Toolbar:
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
                text: 'Take Stroop Test'
                background_color: get_color_from_hex('#7a6a9f')
                on_release: root.go_stroop()
            Widget:
                size_hint_y: None
                height: dp(16)
            AccentLabel:
                text: 'Results'
            DarkButton:
                text: 'View Past Results'
                background_color: get_color_from_hex('#5a8a5a')
                on_release: root.go_results()
            Widget:

# ---- Help menu screen ----

<HelpMenuScreen>:
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: get_color_from_hex('#2b2b2b')
            Rectangle:
                pos: self.pos
                size: self.size
        Toolbar:
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
                background_color: get_color_from_hex('#5a8a5a')
                on_release: root.go_science()
            DarkButton:
                text: 'About'
                background_color: get_color_from_hex('#5a5a7a')
                on_release: root.go_about()
            Widget:

# ---- BDEFS screen ----

<BdefsScreen>:
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: get_color_from_hex('#2b2b2b')
            Rectangle:
                pos: self.pos
                size: self.size
        Toolbar:
        ScrollView:
            do_scroll_x: False
            BoxLayout:
                id: bdefs_content
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                padding: [dp(12), dp(8)]
                spacing: dp(4)

# ---- Stroop screen ----

<StroopScreen>:
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: get_color_from_hex('#2b2b2b')
            Rectangle:
                pos: self.pos
                size: self.size
        Toolbar:
        BoxLayout:
            orientation: 'vertical'
            padding: [dp(16), dp(16)]
            spacing: dp(12)

            Label:
                text: 'Type the COLOUR of the text, not the word.'
                color: get_color_from_hex('#e0e0e0')
                halign: 'center'
                font_size: sp(15)
                size_hint_y: None
                height: dp(30)

            Label:
                text: root.progress_text
                color: get_color_from_hex('#b5b5b5')
                font_size: sp(13)
                size_hint_y: None
                height: dp(24)

            Label:
                text: root.word_text
                font_size: sp(48)
                bold: True
                color: root.word_color
                size_hint_y: None
                height: dp(80)

            TextInput:
                id: stroop_input
                hint_text: 'Type colour here...'
                multiline: False
                font_size: sp(18)
                size_hint_y: None
                height: dp(48)
                background_color: get_color_from_hex('#333333')
                foreground_color: get_color_from_hex('#e0e0e0')
                on_text_validate: root.on_answer(self.text)

            Label:
                text: root.feedback_text
                color: get_color_from_hex('#b5b5b5')
                font_size: sp(13)
                size_hint_y: None
                height: dp(24)

            Widget:
"""


# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------


class Toolbar(BoxLayout):
    """Top navigation bar present on every screen."""

    def go(self, screen_name, direction):
        sm = App.get_running_app().root
        sm.transition = SlideTransition(direction=direction)
        sm.current = screen_name


class TaskRow(BoxLayout):
    """A single task row in the task list."""

    task_id = NumericProperty(0)
    icon = StringProperty("[ ]")
    title_text = StringProperty("")
    home_ref = ObjectProperty(None, allownone=True)

    def on_complete(self):
        if self.home_ref:
            self.home_ref.complete_task(self.task_id)


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


class HomeScreen(Screen):
    """Main screen: tasks, timer, encouragement."""

    status_text = StringProperty("Loading...")
    timer_display = StringProperty("00:00")
    timer_progress = NumericProperty(0)
    nudge_text = StringProperty("")

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
        self._banner_loaded = False

    def on_enter(self):
        if self.conn is None:
            self.conn = db.get_connection()
        self.nudge_text = get_nudge()
        Clock.schedule_once(lambda dt: self.refresh_all(), 0)
        if not self._banner_loaded:
            self._banner_loaded = True
            threading.Thread(target=self._fetch_banner, daemon=True).start()

    # -- Banner image --

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
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = resp.read()
                pil_img = PILImage.open(io.BytesIO(data))
                pil_img = pil_img.resize((500, 120), PILImage.LANCZOS)
                self._draw_title(pil_img)
                Clock.schedule_once(lambda dt, img=pil_img: self._set_banner(img), 0)
                return
            except Exception:
                log.debug("Banner fetch failed for %s", photo_id, exc_info=True)
        # Fallback: solid colour banner with title
        log.warning("Could not fetch any banner image; using fallback.")
        fallback = PILImage.new("RGB", (500, 120), (42, 62, 71))
        self._draw_title(fallback)
        Clock.schedule_once(lambda dt, img=fallback: self._set_banner(img), 0)

    @staticmethod
    def _draw_title(image):
        """Draw 'Momentum' centred on the banner with a drop shadow."""
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
        except OSError:
            try:
                font = ImageFont.load_default(size=32)
            except TypeError:
                font = ImageFont.load_default()
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
        self._selected_task_id = None
        active = db.list_tasks(self.conn, status=TaskStatus.ACTIVE)
        pending = db.list_tasks(self.conn, status=TaskStatus.PENDING)
        for task in active + pending:
            row = TaskRow()
            row.task_id = task.id
            row.icon = "[~]" if task.status == TaskStatus.ACTIVE else "[ ]"
            prefix = "  " if task.is_subtask else ""
            row.title_text = f"{prefix}#{task.id} {task.title}"
            row.home_ref = self
            task_list.add_widget(row)
            if self._selected_task_id is None:
                self._selected_task_id = task.id

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

    def complete_task(self, task_id):
        db.complete_task(self.conn, task_id)
        self.nudge_text = get_nudge()
        self.refresh_all()

    # -- Timer --

    def start_focus(self):
        self._start_timer(15, is_break=False)

    def start_break(self):
        self._start_timer(5, is_break=True)

    def _start_timer(self, minutes, is_break=False):
        if self._timer_running:
            return
        self._timer_running = True
        self._timer_is_break = is_break
        self._timer_total = minutes * 60
        self._timer_seconds_left = self._timer_total
        if not is_break and self._selected_task_id is not None:
            self._timer_task_id = self._selected_task_id
            db.set_task_active(self.conn, self._selected_task_id)
            self.refresh_tasks()
        else:
            self._timer_task_id = None
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
        if self._timer_is_break:
            self.nudge_text = get_break_message()
        else:
            minutes = self._timer_total // 60
            session_in = FocusSessionCreate(
                task_id=self._timer_task_id, duration_minutes=minutes
            )
            db.log_focus_session(self.conn, session_in)
            self.refresh_status()
            self.nudge_text = get_nudge()

    def stop_timer(self):
        self._timer_running = False
        if self._timer_event:
            self._timer_event.cancel()
        self.timer_display = "00:00"
        self.timer_progress = 0

    def refresh_nudge(self):
        self.nudge_text = get_nudge()


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
    """Settings: DB location, cloud sync, reset."""

    def on_enter(self):
        c = self.ids.content
        c.clear_widgets()

        current = cfg.load_config()
        resolved = cfg.get_db_path()
        db_text = current.db_path if current.db_path else f"{resolved} (default)"

        c.add_widget(_make_label("Settings", font_size=sp(20), bold=True, color=_ACCENT))
        c.add_widget(Widget(size_hint_y=None, height=dp(8)))

        # -- Database location --
        c.add_widget(_make_label("Database Location", font_size=sp(16), bold=True, color=_ACCENT))
        self._db_label = _make_label(db_text, font_size=sp(12), color=_MUTED)
        c.add_widget(self._db_label)

        # -- Cloud sync --
        c.add_widget(Widget(size_hint_y=None, height=dp(8)))
        c.add_widget(_make_label("Sync via Cloud", font_size=sp(16), bold=True, color=_ACCENT))
        cloud_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        for provider in ("OneDrive", "Dropbox", "Google Drive"):
            btn = Button(text=provider, font_size=sp(12),
                         background_color=(0.25, 0.25, 0.25, 1))
            key = provider.lower().replace(" ", "-")
            btn.bind(on_release=lambda _, p=key: self._sync(p))
            cloud_row.add_widget(btn)
        c.add_widget(cloud_row)

        # -- Custom path --
        c.add_widget(Widget(size_hint_y=None, height=dp(8)))
        c.add_widget(_make_label("Custom Database Path", font_size=sp(16), bold=True, color=_ACCENT))
        path_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        self._path_input = TextInput(
            hint_text="Enter path...", multiline=False, font_size=sp(13),
            size_hint_x=0.7,
            background_color=(0.2, 0.2, 0.2, 1), foreground_color=_TEXT,
        )
        path_row.add_widget(self._path_input)
        set_btn = Button(text="Set", size_hint_x=0.3, font_size=sp(13),
                         background_color=_ACCENT)
        set_btn.bind(on_release=lambda _: self._set_custom())
        path_row.add_widget(set_btn)
        c.add_widget(path_row)

        # -- Reset --
        c.add_widget(Widget(size_hint_y=None, height=dp(16)))
        reset_btn = Button(
            text="Reset to Default", size_hint_y=None, height=dp(44),
            background_color=(0.55, 0.35, 0.35, 1), font_size=sp(14),
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

    def _reconnect(self):
        home = self.manager.get_screen("home")
        if home.conn:
            home.conn.close()
        home.conn = db.get_connection()

    @staticmethod
    def _show_msg(title, text):
        content = BoxLayout(orientation="vertical", padding=10, spacing=10)
        content.add_widget(Label(text=text, font_size=sp(13), color=_TEXT))
        btn = Button(text="OK", size_hint_y=None, height=dp(44))
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
        c.add_widget(_make_label("Version 0.1.0"))
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
        c.add_widget(_make_label("Author: Robin Oberg", color=_MUTED))
        c.add_widget(_make_label("robinoberg@live.com", color=_MUTED, font_size=sp(12)))


# ---------------------------------------------------------------------------
# Help menu
# ---------------------------------------------------------------------------


class HelpMenuScreen(Screen):
    """Sub-menu: How to Use, The Science, About."""

    def go_howto(self):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "howto"

    def go_science(self):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "science"

    def go_about(self):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "about"


# ---------------------------------------------------------------------------
# Tests menu
# ---------------------------------------------------------------------------


class TestsMenuScreen(Screen):
    """Sub-menu: BDEFS, Stroop, View Results."""

    def go_bdefs(self):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "bdefs"

    def go_stroop(self):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "stroop"

    def go_results(self):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "results"


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

        content = BoxLayout(orientation="vertical", padding=10, spacing=10)
        content.add_widget(_make_label(msg, font_size=sp(13)))
        close = Button(text="Close", size_hint_y=None, height=dp(44))
        popup = Popup(title="Assessment Result", content=content, size_hint=(0.9, 0.65))
        close.bind(on_release=lambda _: popup.dismiss())
        content.add_widget(close)
        popup.open()

        self.manager.transition = SlideTransition(direction="right")
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
            self.manager.transition = SlideTransition(direction="right")
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

        if bdefs_results:
            try:
                radar_img = bdefs_radar(
                    highlight=None, past=bdefs_results,
                    title="Mean Executive Function Profile",
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
                    c.add_widget(_make_label(f"  {d}: {s}", font_size=sp(12), color=_MUTED))
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

        c.add_widget(Widget(size_hint_y=None, height=dp(20)))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class MomentumApp(App):
    """Kivy application entry point."""

    title = "Momentum"

    def build(self):
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
        sm.add_widget(StroopScreen(name="stroop"))
        sm.add_widget(ResultsScreen(name="results"))
        return sm

    def on_stop(self):
        home = self.root.get_screen("home")
        if home.conn:
            home.conn.close()


if __name__ == "__main__":
    MomentumApp().run()
