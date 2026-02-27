"""Momentum mobile app -- Kivy-based Android interface.

Reuses the core momentum modules (models, db, encouragement) with
a touch-friendly UI designed for phones.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the parent package is importable when running standalone
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import (
    BooleanProperty,
    ListProperty,
    NumericProperty,
    StringProperty,
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button

from momentum import db
from momentum.encouragement import get_break_message, get_nudge
from momentum.models import FocusSessionCreate, TaskCreate, TaskStatus

# ---------------------------------------------------------------------------
# Kivy UI definition (KV language)
# ---------------------------------------------------------------------------

KV = """
#:import get_color_from_hex kivy.utils.get_color_from_hex

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

<MomentumRoot>:
    orientation: 'vertical'
    padding: dp(12)
    spacing: dp(8)
    canvas.before:
        Color:
            rgba: get_color_from_hex('#2b2b2b')
        Rectangle:
            pos: self.pos
            size: self.size

    # --- Status bar ---
    Label:
        text: root.status_text
        size_hint_y: None
        height: dp(32)
        color: get_color_from_hex('#6a9fb5')
        font_size: sp(13)

    # --- Task list ---
    Label:
        text: 'Tasks'
        size_hint_y: None
        height: dp(28)
        color: get_color_from_hex('#6a9fb5')
        font_size: sp(18)
        bold: True
        halign: 'left'
        text_size: self.size

    ScrollView:
        size_hint_y: 0.4
        do_scroll_x: False
        BoxLayout:
            id: task_list
            orientation: 'vertical'
            size_hint_y: None
            height: self.minimum_height
            spacing: dp(2)

    BoxLayout:
        size_hint_y: None
        height: dp(44)
        spacing: dp(8)
        Button:
            text: 'Add task'
            background_color: get_color_from_hex('#6a9fb5')
            on_release: root.show_add_dialog()
        Button:
            text: 'Break down'
            background_color: get_color_from_hex('#5a8a5a')
            on_release: root.show_breakdown_dialog()

    # --- Timer ---
    Label:
        text: 'Timer'
        size_hint_y: None
        height: dp(28)
        color: get_color_from_hex('#6a9fb5')
        font_size: sp(18)
        bold: True
        halign: 'left'
        text_size: self.size

    Label:
        text: root.timer_display
        size_hint_y: None
        height: dp(56)
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

    # --- Nudge ---
    Label:
        text: root.nudge_text
        size_hint_y: None
        height: dp(56)
        color: get_color_from_hex('#b5b5b5')
        font_size: sp(13)
        italic: True
        text_size: self.width, None
        halign: 'center'

    Button:
        text: 'New encouragement'
        size_hint_y: None
        height: dp(40)
        background_color: get_color_from_hex('#5a5a7a')
        on_release: root.refresh_nudge()
"""


# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------


class TaskRow(BoxLayout):
    """A single task row in the list."""

    task_id = NumericProperty(0)
    icon = StringProperty("[ ]")
    title_text = StringProperty("")
    app_ref = None  # set by MomentumRoot

    def on_complete(self) -> None:
        if self.app_ref:
            self.app_ref.complete_task(self.task_id)


class MomentumRoot(BoxLayout):
    """Root widget for the mobile app."""

    status_text = StringProperty("Loading...")
    timer_display = StringProperty("00:00")
    timer_progress = NumericProperty(0)
    nudge_text = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conn = db.get_connection()
        self._timer_running: bool = False
        self._timer_seconds_left: int = 0
        self._timer_total: int = 0
        self._timer_task_id: int | None = None
        self._timer_is_break: bool = False
        self._timer_event = None
        self._selected_task_id: int | None = None

        self.nudge_text = get_nudge()
        Clock.schedule_once(lambda dt: self.refresh_all(), 0)

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def refresh_all(self) -> None:
        self.refresh_tasks()
        self.refresh_status()

    def refresh_tasks(self) -> None:
        task_list = self.ids.task_list
        task_list.clear_widgets()

        active = db.list_tasks(self.conn, status=TaskStatus.ACTIVE)
        pending = db.list_tasks(self.conn, status=TaskStatus.PENDING)

        self._selected_task_id = None

        for task in active + pending:
            row = TaskRow()
            row.task_id = task.id
            row.icon = "[~]" if task.status == TaskStatus.ACTIVE else "[ ]"
            prefix = "  " if task.is_subtask else ""
            row.title_text = f"{prefix}#{task.id} {task.title}"
            row.app_ref = self
            task_list.add_widget(row)

            if self._selected_task_id is None:
                self._selected_task_id = task.id

    def refresh_status(self) -> None:
        summary = db.get_status(self.conn)
        today = summary.today
        streak = summary.streak_days
        self.status_text = (
            f"Today: {today.tasks_completed} done, {today.focus_minutes}m focused  |  "
            f"Streak: {streak} day{'s' if streak != 1 else ''}"
        )

    # ------------------------------------------------------------------
    # Task actions
    # ------------------------------------------------------------------

    def show_add_dialog(self) -> None:
        content = BoxLayout(orientation="vertical", spacing=10, padding=10)
        text_input = TextInput(hint_text="What do you need to do?", multiline=False)
        content.add_widget(text_input)

        btn_row = BoxLayout(size_hint_y=None, height=44, spacing=8)

        popup = Popup(title="Add task", content=content, size_hint=(0.9, 0.3))

        def on_add(_):
            title = text_input.text.strip()
            if title:
                task_in = TaskCreate(title=title)
                db.add_task(self.conn, task_in)
                self.refresh_all()
            popup.dismiss()

        def on_cancel(_):
            popup.dismiss()

        add_btn = Button(text="Add")
        add_btn.bind(on_release=on_add)
        cancel_btn = Button(text="Cancel")
        cancel_btn.bind(on_release=on_cancel)

        btn_row.add_widget(add_btn)
        btn_row.add_widget(cancel_btn)
        content.add_widget(btn_row)

        popup.open()

    def show_breakdown_dialog(self) -> None:
        if self._selected_task_id is None:
            return

        task = db.get_task(self.conn, self._selected_task_id)
        if task is None:
            return

        content = BoxLayout(orientation="vertical", spacing=10, padding=10)
        content.add_widget(Label(text=f'Breaking down: "{task.title}"', font_size=14))
        text_input = TextInput(hint_text="Add a sub-step", multiline=False)
        content.add_widget(text_input)

        btn_row = BoxLayout(size_hint_y=None, height=44, spacing=8)

        popup = Popup(title="Break down", content=content, size_hint=(0.9, 0.35))

        def on_add(_):
            step = text_input.text.strip()
            if step:
                sub = TaskCreate(title=step, parent_id=task.id)
                db.add_task(self.conn, sub)
                text_input.text = ""
                self.refresh_tasks()

        def on_done(_):
            popup.dismiss()

        add_btn = Button(text="Add step")
        add_btn.bind(on_release=on_add)
        done_btn = Button(text="Done")
        done_btn.bind(on_release=on_done)

        btn_row.add_widget(add_btn)
        btn_row.add_widget(done_btn)
        content.add_widget(btn_row)

        popup.open()

    def complete_task(self, task_id: int) -> None:
        db.complete_task(self.conn, task_id)
        self.nudge_text = get_nudge()
        self.refresh_all()

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def start_focus(self) -> None:
        self._start_timer(15, is_break=False)

    def start_break(self) -> None:
        self._start_timer(5, is_break=True)

    def _start_timer(self, minutes: int, is_break: bool = False) -> None:
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

    def _tick(self, dt: float) -> None:
        if not self._timer_running:
            if self._timer_event:
                self._timer_event.cancel()
            return

        elapsed = self._timer_total - self._timer_seconds_left
        self.timer_progress = (elapsed / self._timer_total * 100) if self._timer_total > 0 else 0

        mins, secs = divmod(self._timer_seconds_left, 60)
        self.timer_display = f"{mins:02d}:{secs:02d}"

        if self._timer_seconds_left <= 0:
            self._timer_running = False
            if self._timer_event:
                self._timer_event.cancel()
            self._on_timer_complete()
            return

        self._timer_seconds_left -= 1

    def _on_timer_complete(self) -> None:
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

    def stop_timer(self) -> None:
        self._timer_running = False
        if self._timer_event:
            self._timer_event.cancel()
        self.timer_display = "00:00"
        self.timer_progress = 0

    # ------------------------------------------------------------------
    # Nudge
    # ------------------------------------------------------------------

    def refresh_nudge(self) -> None:
        self.nudge_text = get_nudge()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class MomentumApp(App):
    """Kivy application entry point."""

    title = "Momentum"

    def build(self):
        Builder.load_string(KV)
        return MomentumRoot()

    def on_stop(self):
        if hasattr(self.root, "conn"):
            self.root.conn.close()


if __name__ == "__main__":
    MomentumApp().run()
