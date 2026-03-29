"""Tkinter GUI dashboard -- same database and models as the CLI."""

from __future__ import annotations

import io
import logging
import random
import re
import threading
import tkinter as tk
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import messagebox, scrolledtext, simpledialog, ttk
from typing import TYPE_CHECKING, Optional

from PIL import Image, ImageDraw, ImageFont, ImageTk

from momentum import config as cfg
from momentum import db
from momentum.assessments import (
    BDEFS_INSTRUCTIONS,
    BDEFS_QUESTIONS,
    BDEFS_SCALE,
    BISBAS_INSTRUCTIONS,
    BISBAS_QUESTIONS,
    BISBAS_SCALE,
    RESULTS_GUIDE,
    STROOP_INSTRUCTIONS,
    StroopResult,
    bisbas_domain_advice,
    domain_advice,
    generate_stroop_trials,
    interpret_bdefs,
    interpret_bisbas,
    interpret_stroop,
    personalised_nudge,
    score_bdefs,
    score_bisbas,
    score_stroop,
)
from momentum.charts import bdefs_momentum_glow
from momentum.encouragement import get_break_message, get_nudge
from momentum.models import (
    AssessmentResult,
    AssessmentType,
    TaskStatus,
    ThemeMode,
    WindowPosition,
)
from momentum.services import (
    AssessmentService,
    PersonalisationService,
    SessionService,
    StatusService,
    TaskService,
)

if TYPE_CHECKING:
    from momentum.assessments import PersonalisationProfile

log = logging.getLogger(__name__)

# Small fallback list used when IMAGES.md is missing.
_FALLBACK_PHOTOS: list[str] = [
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
    md_path = Path(__file__).resolve().parent.parent / "IMAGES.md"
    if not md_path.exists():
        return _FALLBACK_PHOTOS
    ids: list[str] = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- photo-"):
            photo_id = stripped[2:].strip()
            if photo_id:
                ids.append(photo_id)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for pid in ids:
        if pid not in seen:
            seen.add(pid)
            unique.append(pid)
    return unique if unique else _FALLBACK_PHOTOS


_PEACEFUL_PHOTOS: list[str] = _load_photos()

_IMG_WIDTH: int = 500
_IMG_HEIGHT: int = 120


class MomentumApp:
    """Main GUI application window."""

    FOCUS_DEFAULT_MINUTES: int = 15
    BREAK_DEFAULT_MINUTES: int = 5

    def _font_size(self, base: int) -> int:
        """Scale base font size when large-text accessibility is enabled."""
        scale = 1.2 if self._config.accessibility_large_text else 1.0
        return max(8, int(base * scale))

    def _theme(self) -> dict[str, str]:
        """Resolve the active colour palette from config."""
        if self._config.theme_mode == ThemeMode.LIGHT:
            palette = {
                "bg": "#f5f6f8",
                "panel": "#ffffff",
                "fg": "#1f2933",
                "muted": "#5f6b76",
                "accent": "#2f6f8f",
                "timer": "#a86f00",
                "selection": "#9cc8e0",
                "progress_trough": "#d9e1e8",
            }
        else:
            palette = {
                "bg": "#2b2b2b",
                "panel": "#333333",
                "fg": "#e0e0e0",
                "muted": "#b5b5b5",
                "accent": "#6a9fb5",
                "timer": "#e8c547",
                "selection": "#6a9fb5",
                "progress_trough": "#444444",
            }

        if self._config.accessibility_high_contrast:
            if self._config.theme_mode == ThemeMode.LIGHT:
                palette.update(
                    {
                        "fg": "#000000",
                        "muted": "#222222",
                        "accent": "#005fcc",
                        "selection": "#79b0ff",
                    }
                )
            else:
                palette.update(
                    {
                        "fg": "#ffffff",
                        "muted": "#e6e6e6",
                        "accent": "#8cc9ff",
                        "selection": "#6dafff",
                    }
                )
        return palette

    def _input_palette(self) -> dict[str, str]:
        """Resolve colors for direct tk widgets (Entry/Listbox/Radiobutton)."""
        if self._config.theme_mode == ThemeMode.LIGHT:
            colors = {
                "input_bg": "#ffffff",
                "input_fg": self._palette["fg"],
                "select_bg": self._palette["selection"],
                "radio_select": "#d9e1e8",
                "link": "#1b5fa7",
            }
        else:
            colors = {
                "input_bg": "#333333",
                "input_fg": self._palette["fg"],
                "select_bg": self._palette["selection"],
                "radio_select": "#333333",
                "link": "#82b1ff",
            }
        if self._config.accessibility_high_contrast:
            colors["link"] = (
                "#004ecb" if self._config.theme_mode == ThemeMode.LIGHT else "#9cd1ff"
            )
        return colors

    def _apply_runtime_theme(self) -> None:
        """Apply palette changes to already-created top-level widgets."""
        self.root.configure(bg=self._palette["bg"])
        self._configure_styles()
        if hasattr(self, "_task_listbox"):
            inputs = self._input_palette()
            self._task_listbox.configure(
                bg=self._palette["panel"],
                fg=self._palette["fg"],
                selectbackground=inputs["select_bg"],
                selectforeground=self._palette["fg"],
                font=("sans-serif", self._font_size(10)),
            )
        if hasattr(self, "_image_label"):
            self._image_label.configure(bg=self._palette["bg"])

    def _refresh_banner(self, *, force_fetch: bool = False) -> None:
        """Refresh the top banner after appearance changes."""
        if self._config.accessibility_reduce_visual_load:
            rgb = (
                (226, 235, 240)
                if self._config.theme_mode == ThemeMode.LIGHT
                else (58, 90, 106)
            )
            fallback = Image.new("RGB", (_IMG_WIDTH, _IMG_HEIGHT), rgb)
            self._draw_title(fallback)
            self._set_image(fallback)
            return
        if force_fetch or self._photo_image is None:
            threading.Thread(target=self._fetch_image, daemon=True).start()

    def _personalisation_profile(self) -> PersonalisationProfile:
        """Get personalization defaults from the latest BIS/BAS result."""
        return PersonalisationService(self.conn).profile()

    def _task_service(self) -> TaskService:
        """Build the task workflow service for GUI interactions."""
        return TaskService(self.conn)

    def _assessment_service(self) -> AssessmentService:
        """Build the assessment service for GUI interactions."""
        return AssessmentService(self.conn)

    def _refresh_personalisation(self) -> None:
        """Refresh timer defaults and button labels from BIS/BAS profile."""
        profile = self._personalisation_profile()
        self.FOCUS_DEFAULT_MINUTES = profile.focus_minutes
        self.BREAK_DEFAULT_MINUTES = profile.break_minutes
        if hasattr(self, "_focus_button"):
            self._focus_button.configure(text=f"Focus {self.FOCUS_DEFAULT_MINUTES} min")
        if hasattr(self, "_break_button"):
            self._break_button.configure(text=f"Break {self.BREAK_DEFAULT_MINUTES} min")

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Momentum")
        self._apply_window_position(520, 720)
        self._config = cfg.load_config()
        self._palette = self._theme()
        self.root.configure(bg=self._palette["bg"])
        self._set_app_icon()

        self.conn = db.get_connection()
        self._timer_running: bool = False
        self._timer_seconds_left: int = 0
        self._timer_total: int = 0
        self._timer_task_id: Optional[int] = None
        self._timer_is_break: bool = False
        self._photo_image: Optional[ImageTk.PhotoImage] = None

        self._style = ttk.Style()
        self._style.theme_use("clam")
        self._configure_styles()
        self._build_ui()
        self._refresh_tasks()
        self._refresh_status()
        self._refresh_personalisation()
        self._refresh_banner(force_fetch=True)

    # ------------------------------------------------------------------
    # Window position
    # ------------------------------------------------------------------

    def _apply_window_position(self, w: int, h: int) -> None:
        """Set window geometry based on the configured position preference."""
        config = cfg.load_config()
        if config.window_position == WindowPosition.TOP_LEFT:
            self.root.geometry(f"{w}x{h}+0+0")
        else:
            # Centre on screen
            self.root.update_idletasks()
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            x = (sw - w) // 2
            y = (sh - h) // 2
            self.root.geometry(f"{w}x{h}+{x}+{y}")

    # ------------------------------------------------------------------
    # App icon
    # ------------------------------------------------------------------

    def _set_app_icon(self) -> None:
        """Generate a 64x64 blue 'M' icon and apply it to the window."""
        size = 64
        img = Image.new("RGBA", (size, size), (43, 43, 43, 255))
        draw = ImageDraw.Draw(img)
        # Rounded-rect background in accent blue
        draw.rounded_rectangle([2, 2, size - 3, size - 3], radius=12, fill="#6a9fb5")
        # Draw the "M"
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
        except OSError:
            font = ImageFont.load_default(size=42)
        bbox = draw.textbbox((0, 0), "M", font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - tw) // 2
        y = (size - th) // 2 - bbox[1]  # compensate for font ascent offset
        draw.text((x, y), "M", fill="white", font=font)
        self._icon_image = ImageTk.PhotoImage(img)
        self.root.wm_iconphoto(True, self._icon_image)

    def _configure_styles(self) -> None:
        """Configure ttk styles for the active theme and accessibility options."""
        bg = self._palette["bg"]
        fg = self._palette["fg"]
        accent = self._palette["accent"]

        self._style.configure("TFrame", background=bg)
        self._style.configure(
            "TLabel",
            background=bg,
            foreground=fg,
            font=("sans-serif", self._font_size(10)),
        )
        self._style.configure(
            "Title.TLabel",
            background=bg,
            foreground=accent,
            font=("sans-serif", self._font_size(14), "bold"),
        )
        self._style.configure(
            "Timer.TLabel",
            background=bg,
            foreground=self._palette["timer"],
            font=("monospace", self._font_size(28), "bold"),
        )
        self._style.configure(
            "Nudge.TLabel",
            background=bg,
            foreground=self._palette["muted"],
            font=("sans-serif", self._font_size(10), "italic"),
            wraplength=460,
        )
        self._style.configure("TButton", font=("sans-serif", self._font_size(9)))
        self._style.configure(
            "Accent.TButton", font=("sans-serif", self._font_size(9), "bold")
        )
        self._style.configure(
            "Horizontal.TProgressbar",
            troughcolor=self._palette["progress_trough"],
            background=accent,
        )

    def _build_ui(self) -> None:
        """Construct all UI elements."""
        pad = {"padx": 10, "pady": 5}
        panel_bg = self._palette["panel"]
        fg = self._palette["fg"]
        accent = self._palette["accent"]

        # --- Menu bar ---
        menubar = tk.Menu(
            self.root,
            bg=panel_bg,
            fg=fg,
            activebackground=accent,
            activeforeground=fg,
        )
        self.root.configure(menu=menubar)

        app_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=panel_bg,
            fg=fg,
            activebackground=accent,
            activeforeground=fg,
        )
        app_menu.add_command(label="Settings", command=self._on_settings)
        app_menu.add_separator()
        app_menu.add_command(label="Quit", command=self.root.destroy)
        menubar.add_cascade(label="Menu", menu=app_menu)

        help_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=panel_bg,
            fg=fg,
            activebackground=accent,
            activeforeground=fg,
        )
        help_menu.add_command(label="How to Use", command=self._on_help)
        help_menu.add_command(label="The Science", command=self._on_science)
        help_menu.add_command(label="About", command=self._on_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        tests_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=panel_bg,
            fg=fg,
            activebackground=accent,
            activeforeground=fg,
        )
        tests_menu.add_command(label="Take Self-Assessment", command=self._on_bdefs)
        tests_menu.add_command(label="Take BIS/BAS Profile", command=self._on_bisbas)
        tests_menu.add_command(label="Take Stroop Test", command=self._on_stroop)
        tests_menu.add_separator()
        tests_menu.add_command(label="View Results", command=self._on_view_results)
        menubar.add_cascade(label="Tests", menu=tests_menu)

        # --- Peaceful image banner ---
        self._image_label = tk.Label(
            self.root,
            bg=self._palette["bg"],
            height=_IMG_HEIGHT // 8,
        )
        self._image_label.pack(fill=tk.X, padx=10, pady=(6, 0))

        # --- Nudge ---
        nudge_frame = ttk.Frame(self.root)
        nudge_frame.pack(fill=tk.X, **pad)

        self._nudge_label = ttk.Label(
            nudge_frame,
            text=personalised_nudge(get_nudge(), self._personalisation_profile()),
            style="Nudge.TLabel",
        )
        self._nudge_label.pack(pady=(5, 3))

        ttk.Button(nudge_frame, text="New encouragement", command=self._on_nudge).pack()

        # --- Status bar ---
        self._status_frame = ttk.Frame(self.root)
        self._status_frame.pack(fill=tk.X, **pad)
        self._status_label = ttk.Label(self._status_frame, text="", style="TLabel")
        self._status_label.pack(side=tk.LEFT)

        # --- Task list ---
        task_frame = ttk.Frame(self.root)
        task_frame.pack(fill=tk.BOTH, expand=True, **pad)

        ttk.Label(task_frame, text="Tasks", style="Title.TLabel").pack(anchor=tk.W)

        list_container = ttk.Frame(task_frame)
        list_container.pack(fill=tk.BOTH, expand=True)

        self._task_listbox = tk.Listbox(
            list_container,
            bg=panel_bg,
            fg=fg,
            selectbackground=self._palette["selection"],
            selectforeground=fg,
            font=("sans-serif", self._font_size(10)),
            activestyle="none",
            borderwidth=0,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            list_container, orient=tk.VERTICAL, command=self._task_listbox.yview
        )
        self._task_listbox.configure(yscrollcommand=scrollbar.set)
        self._task_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._task_listbox.bind("<Double-1>", self._on_toggle_task)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Task buttons
        btn_frame = ttk.Frame(task_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(btn_frame, text="Add task", command=self._on_add_task).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="Complete", command=self._on_complete_task).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="Break down", command=self._on_break_down).pack(
            side=tk.LEFT, padx=2
        )

        self._show_completed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            btn_frame,
            text="Show completed",
            variable=self._show_completed_var,
            command=self._refresh_tasks,
        ).pack(side=tk.RIGHT, padx=2)

        # --- Timer ---
        timer_frame = ttk.Frame(self.root)
        timer_frame.pack(fill=tk.X, **pad)

        ttk.Label(timer_frame, text="Timer", style="Title.TLabel").pack(anchor=tk.W)

        self._timer_label = ttk.Label(timer_frame, text="00:00", style="Timer.TLabel")
        self._timer_label.pack(pady=(5, 0))

        self._timer_progress = ttk.Progressbar(
            timer_frame, orient=tk.HORIZONTAL, length=460, mode="determinate"
        )
        self._timer_progress.pack(pady=5)

        timer_btn_frame = ttk.Frame(timer_frame)
        timer_btn_frame.pack()
        self._focus_button = ttk.Button(
            timer_btn_frame,
            text=f"Focus {self.FOCUS_DEFAULT_MINUTES} min",
            command=self._on_focus,
            style="Accent.TButton",
        )
        self._focus_button.pack(side=tk.LEFT, padx=2)
        self._break_button = ttk.Button(
            timer_btn_frame,
            text=f"Break {self.BREAK_DEFAULT_MINUTES} min",
            command=self._on_break,
        )
        self._break_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(timer_btn_frame, text="Stop", command=self._on_stop_timer).pack(
            side=tk.LEFT, padx=2
        )

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def _refresh_tasks(self) -> None:
        """Reload the task list from the database."""
        self._task_listbox.delete(0, tk.END)
        self._task_ids: list[int] = []
        tasks = self._task_service()

        active = tasks.list_tasks(status=TaskStatus.ACTIVE)
        pending = tasks.list_tasks(status=TaskStatus.PENDING)

        for task in active + pending:
            icon = "[~]" if task.status == TaskStatus.ACTIVE else "[ ]"
            prefix = "    " if task.is_subtask else ""
            self._task_listbox.insert(
                tk.END, f"{prefix}{icon} #{task.id}  {task.title}"
            )
            self._task_ids.append(task.id)

        if self._show_completed_var.get():
            done = tasks.list_tasks(status=TaskStatus.DONE)
            for task in done:
                prefix = "    " if task.is_subtask else ""
                self._task_listbox.insert(
                    tk.END, f"{prefix}[x] #{task.id}  {task.title}"
                )
                self._task_listbox.itemconfig(tk.END, fg=self._palette["muted"])
                self._task_ids.append(task.id)

    def _refresh_status(self) -> None:
        """Update the status bar."""
        summary = StatusService(self.conn).summary()
        today = summary.today
        text = (
            f"Today: {today.tasks_completed} done, {today.focus_minutes} min focused  |  "
            f"Streak: {summary.streak_days} day{'s' if summary.streak_days != 1 else ''}"
        )
        self._status_label.configure(text=text)

    # ------------------------------------------------------------------
    # Task actions
    # ------------------------------------------------------------------

    def _selected_task_id(self) -> Optional[int]:
        """Get the task ID from the current listbox selection."""
        sel = self._task_listbox.curselection()
        if not sel:
            return None
        idx: int = sel[0]
        if idx < len(self._task_ids):
            return self._task_ids[idx]
        return None

    def _on_add_task(self) -> None:
        title = simpledialog.askstring(
            "Add task", "What do you need to do?", parent=self.root
        )
        if title and title.strip():
            self._task_service().add_task(title.strip())
            self._refresh_tasks()
            self._refresh_status()

    def _on_complete_task(self) -> None:
        task_id = self._selected_task_id()
        if task_id is None:
            messagebox.showinfo("Complete", "Select a task first.", parent=self.root)
            return
        self._task_service().complete_task(task_id)
        self._refresh_tasks()
        self._refresh_status()
        self._nudge_label.configure(
            text=personalised_nudge(get_nudge(), self._personalisation_profile())
        )

    def _on_toggle_task(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Double-click toggles a task between done and pending."""
        task_id = self._selected_task_id()
        if task_id is None:
            return
        tasks = self._task_service()
        task = tasks.get_task(task_id)
        if task is None:
            return
        if task.status == TaskStatus.DONE:
            tasks.reopen_task(task_id)
        else:
            tasks.complete_task(task_id)
            self._nudge_label.configure(
                text=personalised_nudge(get_nudge(), self._personalisation_profile())
            )
        self._refresh_tasks()
        self._refresh_status()

    def _on_break_down(self) -> None:
        task_id = self._selected_task_id()
        if task_id is None:
            messagebox.showinfo("Break down", "Select a task first.", parent=self.root)
            return
        step = simpledialog.askstring("Break down", "Add a sub-step:", parent=self.root)
        if step and step.strip():
            self._task_service().add_subtask(parent_id=task_id, title=step.strip())
            self._refresh_tasks()

    # ------------------------------------------------------------------
    # Timer actions
    # ------------------------------------------------------------------

    def _start_timer(self, minutes: int, is_break: bool = False) -> None:
        if self._timer_running:
            messagebox.showinfo(
                "Timer", "A timer is already running.", parent=self.root
            )
            return

        self._timer_running = True
        self._timer_is_break = is_break
        self._timer_total = minutes * 60
        self._timer_seconds_left = self._timer_total
        self._timer_progress["maximum"] = self._timer_total
        self._timer_progress["value"] = 0

        task_id = self._selected_task_id()
        if not is_break and task_id is not None:
            self._timer_task_id = task_id
            self._task_service().activate_task(task_id)
            self._refresh_tasks()
        else:
            self._timer_task_id = None

        self._tick()

    def _tick(self) -> None:
        if not self._timer_running:
            return

        elapsed = self._timer_total - self._timer_seconds_left
        self._timer_progress["value"] = elapsed

        mins, secs = divmod(self._timer_seconds_left, 60)
        self._timer_label.configure(text=f"{mins:02d}:{secs:02d}")

        if self._timer_seconds_left <= 0:
            self._timer_running = False
            self._on_timer_complete()
            return

        self._timer_seconds_left -= 1
        self.root.after(1000, self._tick)

    def _on_timer_complete(self) -> None:
        self.root.bell()
        if self._timer_is_break:
            self._nudge_label.configure(text=get_break_message())
            messagebox.showinfo(
                "Break over", "Break time is up. Ready to go again?", parent=self.root
            )
        else:
            # Log the session
            minutes = self._timer_total // 60
            SessionService(self.conn).log_focus(
                task_id=self._timer_task_id, duration_minutes=minutes
            )
            self._refresh_status()
            self._nudge_label.configure(
                text=personalised_nudge(get_nudge(), self._personalisation_profile())
            )
            messagebox.showinfo(
                "Focus complete",
                f"{minutes}-minute focus session logged.",
                parent=self.root,
            )

    def _on_focus(self) -> None:
        self._start_timer(self.FOCUS_DEFAULT_MINUTES, is_break=False)

    def _on_break(self) -> None:
        self._start_timer(self.BREAK_DEFAULT_MINUTES, is_break=True)

    def _on_stop_timer(self) -> None:
        self._timer_running = False
        self._timer_label.configure(text="00:00")
        self._timer_progress["value"] = 0

    # ------------------------------------------------------------------
    # Nudge
    # ------------------------------------------------------------------

    def _on_nudge(self) -> None:
        self._nudge_label.configure(
            text=personalised_nudge(get_nudge(), self._personalisation_profile())
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _on_settings(self) -> None:
        """Open the settings dialog."""
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("520x680")
        win.configure(bg=self._palette["bg"])
        win.transient(self.root)
        inputs = self._input_palette()
        win.grab_set()

        pad = {"padx": 12, "pady": 4}

        # Current DB path
        current = cfg.load_config()
        resolved = cfg.get_db_path()
        db_text = current.db_path if current.db_path else f"{resolved} (default)"

        ttk.Label(win, text="Database location", style="Title.TLabel").pack(
            anchor=tk.W, **pad
        )
        path_label = ttk.Label(win, text=db_text, style="TLabel", wraplength=400)
        path_label.pack(anchor=tk.W, padx=12)

        # Cloud sync buttons
        ttk.Label(win, text="Sync via cloud", style="Title.TLabel").pack(
            anchor=tk.W, padx=12, pady=(10, 4)
        )
        cloud_frame = ttk.Frame(win)
        cloud_frame.pack(fill=tk.X, padx=12)

        def _sync(provider: str) -> None:
            result = cfg.set_cloud_sync(provider)
            if result is None:
                messagebox.showwarning(
                    "Not found",
                    f"Could not find {provider} folder on this system.",
                    parent=win,
                )
                return
            path_label.configure(text=result.db_path)
            self._reconnect_db()
            messagebox.showinfo(
                "Sync configured", f"Database: {result.db_path}", parent=win
            )

        for provider in ("OneDrive", "Dropbox", "Google Drive"):
            key = provider.lower().replace(" ", "-")
            ttk.Button(cloud_frame, text=provider, command=lambda p=key: _sync(p)).pack(
                side=tk.LEFT, padx=2
            )

        # Custom path
        ttk.Label(win, text="Or set a custom path", style="Title.TLabel").pack(
            anchor=tk.W, padx=12, pady=(10, 4)
        )
        custom_frame = ttk.Frame(win)
        custom_frame.pack(fill=tk.X, padx=12)
        inputs = self._input_palette()

        path_entry = tk.Entry(
            custom_frame,
            bg=inputs["input_bg"],
            fg=inputs["input_fg"],
            insertbackground=inputs["input_fg"],
            font=("sans-serif", self._font_size(10)),
        )
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        def _set_custom() -> None:
            p = path_entry.get().strip()
            if p:
                result = cfg.set_db_path(p)
                path_label.configure(text=result.db_path)
                self._reconnect_db()
                messagebox.showinfo(
                    "Path set", f"Database: {result.db_path}", parent=win
                )

        ttk.Button(custom_frame, text="Set", command=_set_custom).pack(side=tk.LEFT)

        # --- Window position ---
        ttk.Label(win, text="Window start position", style="Title.TLabel").pack(
            anchor=tk.W, padx=12, pady=(10, 4)
        )
        pos_frame = ttk.Frame(win)
        pos_frame.pack(fill=tk.X, padx=12)

        pos_var = tk.StringVar(value=current.window_position.value)

        def _set_position() -> None:
            conf = cfg.load_config()
            conf.window_position = WindowPosition(pos_var.get())
            cfg.save_config(conf)

        for label, value in (
            ("Centre of screen", WindowPosition.CENTRE.value),
            ("Top-left corner", WindowPosition.TOP_LEFT.value),
        ):
            tk.Radiobutton(
                pos_frame,
                text=label,
                variable=pos_var,
                value=value,
                command=_set_position,
                bg=self._palette["bg"],
                fg=self._palette["fg"],
                selectcolor=inputs["radio_select"],
                activebackground=self._palette["bg"],
                activeforeground=self._palette["fg"],
                font=("sans-serif", self._font_size(10)),
            ).pack(anchor=tk.W)

        # --- Theme and accessibility ---
        ttk.Label(win, text="Appearance", style="Title.TLabel").pack(
            anchor=tk.W, padx=12, pady=(10, 4)
        )
        appearance_frame = ttk.Frame(win)
        appearance_frame.pack(fill=tk.X, padx=12)
        theme_var = tk.StringVar(value=current.theme_mode.value)

        def _reopen_settings() -> None:
            if win.winfo_exists():
                win.destroy()
            self.root.after_idle(self._on_settings)

        def _set_theme() -> None:
            cfg.set_theme_mode(theme_var.get())
            self._config = cfg.load_config()
            self._palette = self._theme()
            self._apply_runtime_theme()
            self._refresh_banner(force_fetch=True)
            self._refresh_tasks()
            self._refresh_status()
            self._refresh_personalisation()
            _reopen_settings()

        for label, value in (
            ("Dark", ThemeMode.DARK.value),
            ("Light", ThemeMode.LIGHT.value),
        ):
            tk.Radiobutton(
                appearance_frame,
                text=label,
                variable=theme_var,
                value=value,
                command=_set_theme,
                bg=self._palette["bg"],
                fg=self._palette["fg"],
                selectcolor=inputs["radio_select"],
                activebackground=self._palette["bg"],
                activeforeground=self._palette["fg"],
                font=("sans-serif", self._font_size(10)),
            ).pack(anchor=tk.W)

        ttk.Label(win, text="Accessibility", style="Title.TLabel").pack(
            anchor=tk.W, padx=12, pady=(8, 4)
        )
        access_frame = ttk.Frame(win)
        access_frame.pack(fill=tk.X, padx=12)

        large_text_var = tk.BooleanVar(value=current.accessibility_large_text)
        high_contrast_var = tk.BooleanVar(value=current.accessibility_high_contrast)
        reduce_visual_var = tk.BooleanVar(
            value=current.accessibility_reduce_visual_load
        )

        def _apply_accessibility() -> None:
            cfg.set_accessibility_options(
                large_text=large_text_var.get(),
                high_contrast=high_contrast_var.get(),
                reduce_visual_load=reduce_visual_var.get(),
            )
            self._config = cfg.load_config()
            self._palette = self._theme()
            self._apply_runtime_theme()
            self._refresh_tasks()
            self._refresh_status()
            self._refresh_personalisation()
            self._refresh_banner(force_fetch=True)
            _reopen_settings()

        for label, var in (
            ("Larger text", large_text_var),
            ("Higher contrast", high_contrast_var),
            ("Reduce visual load (simple banner)", reduce_visual_var),
        ):
            tk.Checkbutton(
                access_frame,
                text=label,
                variable=var,
                command=_apply_accessibility,
                bg=self._palette["bg"],
                fg=self._palette["fg"],
                activebackground=self._palette["bg"],
                activeforeground=self._palette["fg"],
                selectcolor=inputs["radio_select"],
                font=("sans-serif", self._font_size(10)),
            ).pack(anchor=tk.W)

        # Reset
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=12, pady=(10, 8))

        def _reset() -> None:
            cfg.reset_db_path()
            new_path = cfg.get_db_path()
            path_label.configure(text=f"{new_path} (default)")
            self._reconnect_db()

        ttk.Button(btn_frame, text="Reset to default", command=_reset).pack(
            side=tk.LEFT
        )
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side=tk.RIGHT)

        # --- Data management ---
        ttk.Label(win, text="Data Management", style="Title.TLabel").pack(
            anchor=tk.W, padx=12, pady=(10, 4)
        )
        data_frame = ttk.Frame(win)
        data_frame.pack(fill=tk.X, padx=12)

        def _delete_results() -> None:
            count = self._assessment_service().count_results()
            if count == 0:
                messagebox.showinfo("No data", "No assessment results.", parent=win)
                return
            if messagebox.askyesno(
                "Confirm",
                f"Delete all {count} assessment result{'s' if count != 1 else ''}?",
                parent=win,
            ):
                self._assessment_service().delete_all_results()
                messagebox.showinfo(
                    "Deleted", "All assessment results deleted.", parent=win
                )

        def _delete_tasks() -> None:
            count = len(self._task_service().list_tasks())
            if count == 0:
                messagebox.showinfo("No data", "No tasks.", parent=win)
                return
            if messagebox.askyesno(
                "Confirm",
                f"Delete all {count} task{'s' if count != 1 else ''} and focus sessions?",
                parent=win,
            ):
                self._task_service().delete_all_tasks()
                self._refresh_tasks()
                self._refresh_status()
                messagebox.showinfo(
                    "Deleted", "All tasks and sessions deleted.", parent=win
                )

        ttk.Button(
            data_frame, text="Delete all test results", command=_delete_results
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(data_frame, text="Delete all tasks", command=_delete_tasks).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(data_frame, text="Browse database", command=self._on_browse_db).pack(
            side=tk.LEFT, padx=2
        )

    def _on_browse_db(self) -> None:
        """Open a database browser window."""
        win = tk.Toplevel(self.root)
        win.title("Browse Database")
        win.geometry("640x500")
        win.configure(bg=self._palette["bg"])
        win.transient(self.root)
        inputs = self._input_palette()

        # Table selector
        sel_frame = ttk.Frame(win)
        sel_frame.pack(fill=tk.X, padx=8, pady=4)
        table_var = tk.StringVar(value="tasks")

        listbox = tk.Listbox(
            win,
            bg=inputs["input_bg"],
            fg=inputs["input_fg"],
            selectbackground=inputs["select_bg"],
            selectforeground=self._palette["fg"],
            font=("monospace", self._font_size(9)),
            activestyle="none",
            borderwidth=0,
            highlightthickness=0,
        )
        listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        row_ids: list[str] = []

        def _load() -> None:
            listbox.delete(0, tk.END)
            row_ids.clear()
            tbl = table_var.get()
            if tbl == "tasks":
                for t in self._task_service().list_tasks():
                    listbox.insert(tk.END, f"#{t.id}  [{t.status.value}]  {t.title}")
                    row_ids.append(str(t.id))
            elif tbl == "assessments":
                for r in self._assessment_service().list_results(limit=100):
                    listbox.insert(
                        tk.END,
                        f"#{r.id}  {r.assessment_type.value}  "
                        f"{r.score}/{r.max_score}  {r.taken_at:%Y-%m-%d %H:%M}",
                    )
                    row_ids.append(str(r.id))
            elif tbl == "sessions":
                for s in db.list_focus_sessions(self.conn, limit=100):
                    listbox.insert(
                        tk.END,
                        f"#{s.id}  {s.duration_minutes}min  task={s.task_id}  "
                        f"{s.completed_at:%Y-%m-%d %H:%M}",
                    )
                    row_ids.append(str(s.id))
            elif tbl == "daily_log":
                for lg in db.list_all_daily_logs(self.conn):
                    listbox.insert(
                        tk.END,
                        f"{lg.date}  tasks={lg.tasks_completed}  "
                        f"focus={lg.focus_minutes}min",
                    )
                    row_ids.append(str(lg.date))

        for tbl_name in ("tasks", "assessments", "sessions", "daily_log"):
            tk.Radiobutton(
                sel_frame,
                text=tbl_name,
                variable=table_var,
                value=tbl_name,
                command=_load,
                bg=self._palette["bg"],
                fg=self._palette["fg"],
                selectcolor=inputs["radio_select"],
                activebackground=self._palette["bg"],
                activeforeground=self._palette["fg"],
                font=("sans-serif", self._font_size(9)),
            ).pack(side=tk.LEFT, padx=4)

        def _delete_selected() -> None:
            sel = listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            eid = row_ids[idx]
            tbl = table_var.get()
            ok = False
            if tbl == "tasks":
                ok = db.delete_task(self.conn, int(eid))
            elif tbl == "assessments":
                ok = db.delete_assessment(self.conn, int(eid))
            elif tbl == "sessions":
                ok = db.delete_focus_session(self.conn, int(eid))
            elif tbl == "daily_log":
                ok = db.delete_daily_log(self.conn, eid)
            if ok:
                _load()
                self._refresh_tasks()
                self._refresh_status()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(btn_frame, text="Delete selected", command=_delete_selected).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="Refresh", command=_load).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side=tk.RIGHT)
        _load()

    def _reconnect_db(self) -> None:
        """Close current DB connection and open a new one at the configured path."""
        self.conn.close()
        self.conn = db.get_connection()
        self._refresh_tasks()
        self._refresh_status()

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _on_help(self) -> None:
        """Show the README in a scrollable window with rendered markdown."""
        readme_path = Path(__file__).resolve().parent.parent / "README.md"
        if readme_path.exists():
            content = readme_path.read_text(encoding="utf-8")
        else:
            content = (
                "# Momentum\n\n"
                "A gentle tool for executive dysfunction support.\n\n"
                "## Commands\n\n"
                "- **Add task** -- add something you need to do\n"
                "- **Complete** -- mark a selected task as done\n"
                "- **Break down** -- split a task into smaller steps\n"
            )

        win = tk.Toplevel(self.root)
        win.title("How to Use")
        win.geometry("600x540")
        win.configure(bg=self._palette["bg"])
        win.transient(self.root)
        inputs = self._input_palette()

        text = scrolledtext.ScrolledText(
            win,
            wrap=tk.WORD,
            bg=inputs["input_bg"],
            fg=inputs["input_fg"],
            font=("sans-serif", self._font_size(10)),
            borderwidth=0,
            highlightthickness=0,
            padx=16,
            pady=12,
        )
        text.pack(fill=tk.BOTH, expand=True)
        self._render_markdown(text, content)
        text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Markdown renderer
    # ------------------------------------------------------------------

    def _render_markdown(self, widget: scrolledtext.ScrolledText, md: str) -> None:
        """Insert *md* into *widget* with basic visual formatting."""
        inputs = self._input_palette()
        h_fg = self._palette["accent"]
        body_fg = self._palette["fg"]
        muted_fg = self._palette["muted"]
        code_bg = (
            self._palette["panel"]
            if self._config.theme_mode == ThemeMode.LIGHT
            else "#1e1e1e"
        )
        widget.tag_configure(
            "h1",
            font=("sans-serif", self._font_size(16), "bold"),
            foreground=h_fg,
            spacing3=4,
        )
        widget.tag_configure(
            "h2",
            font=("sans-serif", self._font_size(13), "bold"),
            foreground=h_fg,
            spacing1=10,
            spacing3=2,
        )
        widget.tag_configure(
            "h3",
            font=("sans-serif", self._font_size(11), "bold"),
            foreground=h_fg,
            spacing1=8,
            spacing3=2,
        )
        widget.tag_configure(
            "body",
            font=("sans-serif", self._font_size(10)),
            foreground=body_fg,
        )
        widget.tag_configure(
            "bullet",
            font=("sans-serif", self._font_size(10)),
            foreground=body_fg,
            lmargin1=16,
            lmargin2=28,
        )
        widget.tag_configure(
            "code_block",
            font=("monospace", self._font_size(9)),
            foreground=muted_fg,
            background=code_bg,
            lmargin1=16,
            lmargin2=16,
            rmargin=16,
            spacing1=4,
            spacing3=4,
        )
        widget.tag_configure(
            "table_row",
            font=("monospace", self._font_size(9)),
            foreground=body_fg,
            lmargin1=8,
            lmargin2=8,
        )
        widget.tag_configure(
            "bold",
            font=("sans-serif", self._font_size(10), "bold"),
            foreground=body_fg,
        )
        widget.tag_configure(
            "inline_code",
            font=("monospace", self._font_size(9)),
            foreground=muted_fg,
            background=code_bg,
        )

        link_count = 0

        def _make_link(url: str):
            return lambda _e: webbrowser.open(url)

        def _insert_inline(line: str, base_tag: str) -> None:
            """Insert a line handling **bold**, `code`, and [links](url)."""
            nonlocal link_count
            pattern = re.compile(r"(\*\*(.+?)\*\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\))")
            pos = 0
            for m in pattern.finditer(line):
                # Text before this match
                if m.start() > pos:
                    widget.insert(tk.END, line[pos : m.start()], base_tag)
                if m.group(2) is not None:  # **bold**
                    widget.insert(tk.END, m.group(2), "bold")
                elif m.group(3) is not None:  # `inline code`
                    widget.insert(tk.END, m.group(3), "inline_code")
                elif m.group(4) is not None:  # [text](url)
                    tag = f"md_link_{link_count}"
                    link_count += 1
                    widget.tag_configure(
                        tag,
                        font=("sans-serif", self._font_size(10)),
                        foreground=inputs["link"],
                        underline=True,
                    )
                    widget.tag_bind(tag, "<Button-1>", _make_link(m.group(5)))
                    widget.tag_bind(
                        tag, "<Enter>", lambda _e: widget.configure(cursor="hand2")
                    )
                    widget.tag_bind(
                        tag, "<Leave>", lambda _e: widget.configure(cursor="")
                    )
                    widget.insert(tk.END, m.group(4), tag)
                pos = m.end()
            if pos < len(line):
                widget.insert(tk.END, line[pos:], base_tag)

        in_code_block = False
        lines = md.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            # Code block fences
            if line.startswith("```"):
                if not in_code_block:
                    in_code_block = True
                    i += 1
                    continue
                else:
                    in_code_block = False
                    widget.insert(tk.END, "\n")
                    i += 1
                    continue

            if in_code_block:
                widget.insert(tk.END, line + "\n", "code_block")
                i += 1
                continue

            # Blank line
            if not line.strip():
                widget.insert(tk.END, "\n")
                i += 1
                continue

            # Headings
            if line.startswith("### "):
                widget.insert(tk.END, line[4:] + "\n", "h3")
                i += 1
                continue
            if line.startswith("## "):
                widget.insert(tk.END, line[3:] + "\n", "h2")
                i += 1
                continue
            if line.startswith("# "):
                widget.insert(tk.END, line[2:] + "\n", "h1")
                i += 1
                continue

            # Table separator rows (|---|---|
            if re.match(r"^\|[-\s|:]+\|$", line):
                i += 1
                continue

            # Table rows
            if line.startswith("|") and line.endswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                row = "  ".join(
                    f"{c:<30}" if ci == 0 else c for ci, c in enumerate(cells)
                )
                _insert_inline(row + "\n", "table_row")
                i += 1
                continue

            # Bullet list
            if line.startswith("- ") or line.startswith("* "):
                _insert_inline(
                    "  "
                    + line[:2].replace("-", "--").replace("*", "--")
                    + " "
                    + line[2:]
                    + "\n",
                    "bullet",
                )
                i += 1
                continue

            # Regular paragraph line
            _insert_inline(line + "\n", "body")
            i += 1

    def _on_science(self) -> None:
        """Show the scientific rationale loaded from SCIENCE.md."""
        science_path = Path(__file__).resolve().parent.parent / "SCIENCE.md"
        if science_path.exists():
            content = science_path.read_text(encoding="utf-8")
        else:
            content = "# The Science Behind Momentum\n\nScience document not found."

        win = tk.Toplevel(self.root)
        win.title("The Science Behind Momentum")
        win.geometry("620x640")
        win.configure(bg=self._palette["bg"])
        win.transient(self.root)
        inputs = self._input_palette()

        text = scrolledtext.ScrolledText(
            win,
            wrap=tk.WORD,
            bg=inputs["input_bg"],
            fg=inputs["input_fg"],
            font=("sans-serif", self._font_size(10)),
            borderwidth=0,
            highlightthickness=0,
            padx=16,
            pady=12,
        )
        text.pack(fill=tk.BOTH, expand=True)
        self._render_markdown(text, content)
        text.configure(state=tk.DISABLED)

    def _on_about(self) -> None:
        """Show a brief about dialog."""
        messagebox.showinfo(
            "About Momentum",
            "Momentum v0.1.0\n\n"
            "A gentle tool to help people with executive\n"
            "dysfunction get back on track, one small step\n"
            "at a time.\n\n"
            "Created by Robin \u00d6berg\n"
            "Data Scientist, MSc Social Anthropology,\n"
            "MSc Applied Cultural Analysis.\n\n"
            "Copyright \u00a9 2026 Robin \u00d6berg.\n"
            "Licensed under the MIT License.\n\n"
            "https://github.com/yidaki53/momentum",
            parent=self.root,
        )

    # ------------------------------------------------------------------
    # Self-assessment tests
    # ------------------------------------------------------------------

    def _on_bdefs(self) -> None:
        """Run the BDEFS self-assessment in a dialog."""
        # Show instruction page first
        if not messagebox.askokcancel(
            "BDEFS Self-Assessment",
            BDEFS_INSTRUCTIONS + "\n\nPress OK to begin.",
            parent=self.root,
        ):
            return

        win = tk.Toplevel(self.root)
        win.title("Executive Function Self-Assessment")
        win.geometry("560x520")
        win.configure(bg=self._palette["bg"])
        win.transient(self.root)
        win.grab_set()
        inputs = self._input_palette()

        canvas = tk.Canvas(win, bg=self._palette["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(win, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)

        inner.bind(
            "<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(
            inner,
            text="Rate each statement (1 = Never ... 4 = Very Often)",
            style="Title.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=12, pady=(8, 4))

        vars_map: dict[str, list[tk.IntVar]] = {}
        row = 1
        for domain, questions in BDEFS_QUESTIONS.items():
            ttk.Label(
                inner,
                text=domain,
                style="Title.TLabel",
            ).grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=12, pady=(10, 2))
            row += 1
            domain_vars: list[tk.IntVar] = []
            for q in questions:
                ttk.Label(inner, text=q, style="TLabel", wraplength=380).grid(
                    row=row,
                    column=0,
                    sticky=tk.W,
                    padx=(20, 4),
                    pady=2,
                )
                var = tk.IntVar(value=1)
                domain_vars.append(var)
                opt_frame = ttk.Frame(inner)
                opt_frame.grid(row=row, column=1, sticky=tk.W, padx=4, pady=2)
                for val, label in BDEFS_SCALE.items():
                    tk.Radiobutton(
                        opt_frame,
                        text=str(val),
                        variable=var,
                        value=val,
                        bg=self._palette["bg"],
                        fg=self._palette["fg"],
                        selectcolor=inputs["radio_select"],
                        activebackground=self._palette["bg"],
                        activeforeground=self._palette["fg"],
                        font=("sans-serif", self._font_size(9)),
                    ).pack(side=tk.LEFT)
                row += 1
            vars_map[domain] = domain_vars

        def _submit() -> None:
            answers = {d: [v.get() for v in vs] for d, vs in vars_map.items()}
            create_model = score_bdefs(answers)
            saved = self._assessment_service().save_result(create_model)
            win.destroy()
            self._show_bdefs_result(saved)

        ttk.Button(inner, text="Submit", command=_submit, style="Accent.TButton").grid(
            row=row,
            column=0,
            columnspan=2,
            pady=12,
        )

    def _on_bisbas(self) -> None:
        """Run the BIS/BAS motivational profile in a dialog."""
        if not messagebox.askokcancel(
            "BIS/BAS Motivational Profile",
            BISBAS_INSTRUCTIONS + "\n\nPress OK to begin.",
            parent=self.root,
        ):
            return

        win = tk.Toplevel(self.root)
        win.title("BIS/BAS Motivational Profile")
        win.geometry("600x560")
        win.configure(bg=self._palette["bg"])
        win.transient(self.root)
        win.grab_set()

        canvas = tk.Canvas(win, bg=self._palette["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(win, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind(
            "<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(
            inner,
            text="Rate each statement (1 = very false ... 4 = very true)",
            style="Title.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=12, pady=(8, 4))

        vars_map: dict[str, list[tk.IntVar]] = {}
        row = 1
        for domain, questions in BISBAS_QUESTIONS.items():
            ttk.Label(inner, text=domain, style="Title.TLabel").grid(
                row=row,
                column=0,
                columnspan=2,
                sticky=tk.W,
                padx=12,
                pady=(10, 2),
            )
            row += 1
            domain_vars: list[tk.IntVar] = []
            for q in questions:
                ttk.Label(inner, text=q, style="TLabel", wraplength=420).grid(
                    row=row,
                    column=0,
                    sticky=tk.W,
                    padx=(20, 4),
                    pady=2,
                )
                var = tk.IntVar(value=2)
                domain_vars.append(var)
                opt_frame = ttk.Frame(inner)
                opt_frame.grid(row=row, column=1, sticky=tk.W, padx=4, pady=2)
                for val in BISBAS_SCALE:
                    tk.Radiobutton(
                        opt_frame,
                        text=str(val),
                        variable=var,
                        value=val,
                        bg=self._palette["bg"],
                        fg=self._palette["fg"],
                        selectcolor=self._palette["panel"],
                        activebackground=self._palette["bg"],
                        activeforeground=self._palette["fg"],
                        font=("sans-serif", self._font_size(9)),
                    ).pack(side=tk.LEFT)
                row += 1
            vars_map[domain] = domain_vars

        def _submit() -> None:
            answers = {d: [v.get() for v in vs] for d, vs in vars_map.items()}
            create_model = score_bisbas(answers)
            saved = self._assessment_service().save_result(create_model)
            self._refresh_personalisation()
            win.destroy()

            lines: list[str] = [f"Total score: {saved.score}/{saved.max_score}", ""]
            for d, s in saved.domain_scores.items():
                max_domain = len(BISBAS_QUESTIONS.get(d, [])) * 4
                lines.append(f"{d}: {s}/{max_domain if max_domain else 1}")
                advice = bisbas_domain_advice(d, s, max_domain if max_domain else 1)
                if advice:
                    lines.append(f"  - {advice}")
            lines.append("")
            lines.append(
                interpret_bisbas(saved.score, saved.max_score, saved.domain_scores)
            )
            lines.append("")
            lines.append(
                f"Personalized defaults applied: focus {self.FOCUS_DEFAULT_MINUTES} min, "
                f"break {self.BREAK_DEFAULT_MINUTES} min."
            )
            messagebox.showinfo("BIS/BAS Result", "\n".join(lines), parent=self.root)

        ttk.Button(inner, text="Submit", command=_submit, style="Accent.TButton").grid(
            row=row,
            column=0,
            columnspan=2,
            pady=12,
        )

    def _on_stroop(self) -> None:
        """Run the Stroop colour-word test in a dialog."""
        # Show instruction page first
        if not messagebox.askokcancel(
            "Stroop Colour-Word Test",
            STROOP_INSTRUCTIONS + "\n\nPress OK to begin.",
            parent=self.root,
        ):
            return

        import time as _time

        trials = generate_stroop_trials()
        state = {"idx": 0, "correct": 0, "total_time": 0.0, "t0": 0.0, "per_trial": []}

        win = tk.Toplevel(self.root)
        win.title("Stroop Colour-Word Test")
        win.geometry("400x300")
        win.configure(bg=self._palette["bg"])
        win.transient(self.root)
        win.grab_set()
        inputs = self._input_palette()

        ttk.Label(
            win,
            text="Type the COLOUR of the text, not the word.",
            style="TLabel",
            wraplength=360,
        ).pack(padx=12, pady=(12, 4))

        progress_label = ttk.Label(win, text="", style="TLabel")
        progress_label.pack()

        colour_map = {
            "red": "#e06060",
            "green": "#60c060",
            "blue": "#6090e0",
            "yellow": "#e0d060",
        }

        word_label = tk.Label(
            win,
            text="",
            font=("sans-serif", self._font_size(32), "bold"),
            bg=self._palette["bg"],
        )
        word_label.pack(pady=16)

        entry_var = tk.StringVar()
        entry = tk.Entry(
            win,
            textvariable=entry_var,
            bg=inputs["input_bg"],
            fg=inputs["input_fg"],
            insertbackground=inputs["input_fg"],
            font=("sans-serif", self._font_size(14)),
            justify=tk.CENTER,
        )
        entry.pack(padx=40, fill=tk.X)
        entry.focus_set()

        feedback_label = ttk.Label(win, text="", style="TLabel")
        feedback_label.pack(pady=4)

        def _show_trial() -> None:
            idx = state["idx"]
            if idx >= len(trials):
                _finish()
                return
            trial = trials[idx]
            word_label.configure(
                text=trial.word.upper(), fg=colour_map[trial.ink_colour]
            )
            progress_label.configure(text=f"Trial {idx + 1} of {len(trials)}")
            entry_var.set("")
            feedback_label.configure(text="")
            state["t0"] = _time.monotonic()

        def _on_submit(_event=None) -> None:
            elapsed = _time.monotonic() - state["t0"]
            answer = entry_var.get().strip().lower()
            if not answer:
                return
            trial = trials[state["idx"]]
            is_correct = answer == trial.ink_colour
            state["per_trial"].append((is_correct, elapsed))
            state["total_time"] += elapsed
            if is_correct:
                state["correct"] += 1
                feedback_label.configure(text="Correct!", foreground="#60c060")
            else:
                feedback_label.configure(
                    text=f"The colour was {trial.ink_colour}.",
                    foreground="#e06060",
                )
            state["idx"] += 1
            win.after(600, _show_trial)

        entry.bind("<Return>", _on_submit)

        def _finish() -> None:
            result = StroopResult(
                trials=len(trials),
                correct=state["correct"],
                total_time_s=state["total_time"],
                per_trial=state["per_trial"],
            )
            create_model = score_stroop(result)
            saved = self._assessment_service().save_result(create_model)
            avg_ms = int(result.avg_time_s * 1000)
            msg = (
                f"Score: {saved.score}/{saved.max_score}\n"
                f"Average response: {avg_ms} ms\n\n"
                f"{interpret_stroop(saved.score, saved.max_score, avg_ms)}"
            )
            win.destroy()
            messagebox.showinfo("Stroop Test Result", msg, parent=self.root)

        _show_trial()

    # --- shared result display helpers -----------------------------------

    def _show_bdefs_result(self, saved: "AssessmentResult") -> None:
        """Open a results window with a motivating chart and score breakdown."""

        all_bdefs = self._assessment_service().list_results(
            assessment_type=AssessmentType.BDEFS, limit=50
        )
        previous = next((r for r in all_bdefs if r.id != saved.id), None)

        rwin = tk.Toplevel(self.root)
        rwin.title("Assessment Result")
        rwin.geometry("620x780")
        rwin.configure(bg=self._palette["bg"])
        rwin.transient(self.root)

        chart_img = bdefs_momentum_glow(
            latest=saved,
            previous=previous,
            title="Momentum Reserve Snapshot",
        )
        chart_tk = ImageTk.PhotoImage(chart_img)
        chart_label = tk.Label(rwin, image=chart_tk, bg=self._palette["bg"])  # type: ignore[arg-type]
        chart_label.image = chart_tk  # prevent GC
        chart_label.pack(padx=8, pady=(8, 0))

        # Score text
        score_frame = ttk.Frame(rwin)
        score_frame.pack(fill=tk.X, padx=16, pady=(4, 2))
        ttk.Label(
            score_frame,
            text=f"Total score: {saved.score}/{saved.max_score}",
            style="Title.TLabel",
        ).pack(anchor=tk.W)
        for d, s in saved.domain_scores.items():
            n_qs = len(BDEFS_QUESTIONS[d])
            ttk.Label(score_frame, text=f"  {d}: {s}/{n_qs * 4}", style="TLabel").pack(
                anchor=tk.W
            )
            advice = domain_advice(d, s, n_qs * 4)
            if advice:
                ttk.Label(
                    score_frame,
                    text=f"    {advice}",
                    style="Nudge.TLabel",
                    wraplength=560,
                ).pack(anchor=tk.W)

        ttk.Label(
            rwin,
            text=interpret_bdefs(saved.score, saved.max_score),
            style="Nudge.TLabel",
        ).pack(padx=16, pady=(6, 4))

        ttk.Button(rwin, text="Close", command=rwin.destroy).pack(pady=(4, 10))

    def _on_view_results(self) -> None:
        """Display past assessment results with charts in a scrollable window."""
        results = self._assessment_service().list_results(limit=50)

        win = tk.Toplevel(self.root)
        win.title("Assessment Results")
        win.geometry("680x780")
        win.configure(bg=self._palette["bg"])
        win.transient(self.root)
        inputs = self._input_palette()

        # Scrollable canvas for the whole window
        outer_canvas = tk.Canvas(win, bg=self._palette["bg"], highlightthickness=0)
        v_scroll = ttk.Scrollbar(win, orient=tk.VERTICAL, command=outer_canvas.yview)
        content = ttk.Frame(outer_canvas)
        content.bind(
            "<Configure>",
            lambda _e: outer_canvas.configure(scrollregion=outer_canvas.bbox("all")),
        )
        outer_canvas.create_window((0, 0), window=content, anchor="nw")
        outer_canvas.configure(yscrollcommand=v_scroll.set)
        outer_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Keep references to Tk images to prevent garbage collection.
        self._results_images: list[ImageTk.PhotoImage] = []

        bdefs_results = [
            r for r in results if r.assessment_type == AssessmentType.BDEFS
        ]

        # Guide text
        guide_label = ttk.Label(
            content,
            text=RESULTS_GUIDE,
            style="Nudge.TLabel",
            wraplength=620,
        )
        guide_label.pack(padx=16, pady=(8, 4), anchor=tk.W)

        if not results:
            ttk.Label(content, text="No assessment results yet.", style="TLabel").pack(
                padx=16,
                pady=12,
                anchor=tk.W,
            )
            ttk.Label(
                content,
                text="Take a test from the Tests menu to get started.",
                style="Nudge.TLabel",
            ).pack(padx=16, anchor=tk.W)
        else:
            # --- BDEFS momentum glow chart (latest vs previous) ---
            if bdefs_results:
                latest_r = bdefs_results[0]  # most recent first
                prev_r = bdefs_results[1] if len(bdefs_results) > 1 else None
                glow_img = bdefs_momentum_glow(
                    latest=latest_r,
                    previous=prev_r,
                    title="Latest Momentum Reserve Profile",
                )
                glow_tk = ImageTk.PhotoImage(glow_img)
                self._results_images.append(glow_tk)
                tk.Label(content, image=glow_tk, bg=self._palette["bg"]).pack(
                    padx=8, pady=(8, 0)
                )

            # --- Textual list of all results ---
            text = scrolledtext.ScrolledText(
                content,
                wrap=tk.WORD,
                bg=inputs["input_bg"],
                fg=inputs["input_fg"],
                font=("sans-serif", self._font_size(10)),
                borderwidth=0,
                highlightthickness=0,
                padx=16,
                pady=12,
                height=14,
            )
            text.pack(fill=tk.BOTH, expand=True, padx=4)

            text.tag_configure(
                "heading",
                font=("sans-serif", self._font_size(12), "bold"),
                foreground=self._palette["accent"],
            )
            text.tag_configure(
                "body",
                font=("sans-serif", self._font_size(10)),
                foreground=self._palette["fg"],
            )
            text.tag_configure(
                "interp",
                font=("sans-serif", self._font_size(10), "italic"),
                foreground=self._palette["muted"],
            )

            for r in results:
                taken = r.taken_at.strftime("%Y-%m-%d %H:%M")
                text.insert(
                    tk.END,
                    f"{r.assessment_type.value.upper()}  --  {taken}\n",
                    "heading",
                )
                text.insert(tk.END, f"  Score: {r.score}/{r.max_score}\n", "body")
                if r.assessment_type == AssessmentType.BDEFS:
                    for d, s in r.domain_scores.items():
                        text.insert(tk.END, f"    {d}: {s}\n", "body")
                    text.insert(
                        tk.END, f"  {interpret_bdefs(r.score, r.max_score)}\n", "interp"
                    )
                elif r.assessment_type == AssessmentType.STROOP:
                    avg_ms = r.domain_scores.get("avg_time_ms", 0)
                    text.insert(tk.END, f"  Avg response: {avg_ms} ms\n", "body")
                    text.insert(
                        tk.END,
                        f"  {interpret_stroop(r.score, r.max_score, avg_ms)}\n",
                        "interp",
                    )
                elif r.assessment_type == AssessmentType.BISBAS:
                    for d, s in r.domain_scores.items():
                        max_domain = len(BISBAS_QUESTIONS.get(d, [])) * 4
                        text.insert(
                            tk.END,
                            f"    {d}: {s}/{max_domain if max_domain else 1}\n",
                            "body",
                        )
                    text.insert(
                        tk.END,
                        f"  {interpret_bisbas(r.score, r.max_score, r.domain_scores)}\n",
                        "interp",
                    )
                text.insert(tk.END, "\n")

            text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Peaceful image
    # ------------------------------------------------------------------

    def _fetch_image(self) -> None:
        """Download a random peaceful photo in a background thread.

        Retries up to 5 times with different photos if a fetch fails.
        """
        attempts = min(5, len(_PEACEFUL_PHOTOS))
        tried: set[str] = set()
        for _ in range(attempts):
            photo_id = random.choice(_PEACEFUL_PHOTOS)
            if photo_id in tried:
                continue
            tried.add(photo_id)
            url = (
                f"https://images.unsplash.com/{photo_id}"
                f"?w={_IMG_WIDTH}&h={_IMG_HEIGHT}&fit=crop&crop=center&auto=format&q=80"
            )
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Momentum/0.1"}
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = resp.read()
                image = Image.open(io.BytesIO(data))
                image = image.resize((_IMG_WIDTH, _IMG_HEIGHT), Image.LANCZOS)
                self._draw_title(image)
                self.root.after(0, self._set_image, image)
                return
            except Exception:
                log.debug(
                    "Image fetch failed for %s; retrying.", photo_id, exc_info=True
                )
        log.warning("Could not fetch any peaceful image after %d attempts.", attempts)
        # Fallback: solid-colour banner with title text
        fallback = Image.new("RGB", (_IMG_WIDTH, _IMG_HEIGHT), (58, 90, 106))
        self._draw_title(fallback)
        self.root.after(0, self._set_image, fallback)

    @staticmethod
    def _draw_title(image: Image.Image) -> None:
        """Draw 'Momentum' centred on the banner with a drop shadow."""
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
        except OSError:
            font = ImageFont.load_default(size=32)
        text = "Momentum"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (image.width - tw) // 2
        y = (image.height - th) // 2
        # drop shadow
        draw.text((x + 2, y + 2), text, fill=(0, 0, 0, 180), font=font)
        # main text
        draw.text((x, y), text, fill="white", font=font)

    def _set_image(self, image: Image.Image) -> None:
        """Display the fetched image in the banner label (main thread)."""
        self._photo_image = ImageTk.PhotoImage(image)
        self._image_label.configure(image=self._photo_image, height=_IMG_HEIGHT)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the tkinter main loop."""
        self.root.mainloop()
        self.conn.close()


def run_gui() -> None:
    """Entry point for the GUI (called from CLI)."""
    app = MomentumApp()
    app.run()
