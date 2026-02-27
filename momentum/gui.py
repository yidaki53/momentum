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
from typing import Optional

from PIL import Image, ImageDraw, ImageFont, ImageTk

from momentum import config as cfg
from momentum import db
from momentum.assessments import (
    BDEFS_QUESTIONS,
    BDEFS_SCALE,
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
    WindowPosition,
)

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

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Momentum")
        self._apply_window_position(520, 720)
        self.root.configure(bg="#2b2b2b")
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
        threading.Thread(target=self._fetch_image, daemon=True).start()

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
        """Configure ttk styles for a dark, calming theme."""
        bg = "#2b2b2b"
        fg = "#e0e0e0"
        accent = "#6a9fb5"

        self._style.configure("TFrame", background=bg)
        self._style.configure(
            "TLabel", background=bg, foreground=fg, font=("sans-serif", 10)
        )
        self._style.configure(
            "Title.TLabel",
            background=bg,
            foreground=accent,
            font=("sans-serif", 14, "bold"),
        )
        self._style.configure(
            "Timer.TLabel",
            background=bg,
            foreground="#e8c547",
            font=("monospace", 28, "bold"),
        )
        self._style.configure(
            "Nudge.TLabel",
            background=bg,
            foreground="#b5b5b5",
            font=("sans-serif", 10, "italic"),
            wraplength=460,
        )
        self._style.configure("TButton", font=("sans-serif", 9))
        self._style.configure("Accent.TButton", font=("sans-serif", 9, "bold"))
        self._style.configure(
            "Horizontal.TProgressbar", troughcolor="#444", background=accent
        )

    def _build_ui(self) -> None:
        """Construct all UI elements."""
        pad = {"padx": 10, "pady": 5}

        # --- Menu bar ---
        menubar = tk.Menu(
            self.root,
            bg="#333",
            fg="#e0e0e0",
            activebackground="#6a9fb5",
            activeforeground="#fff",
        )
        self.root.configure(menu=menubar)

        app_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg="#333",
            fg="#e0e0e0",
            activebackground="#6a9fb5",
            activeforeground="#fff",
        )
        app_menu.add_command(label="Settings", command=self._on_settings)
        app_menu.add_separator()
        app_menu.add_command(label="Quit", command=self.root.destroy)
        menubar.add_cascade(label="Menu", menu=app_menu)

        help_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg="#333",
            fg="#e0e0e0",
            activebackground="#6a9fb5",
            activeforeground="#fff",
        )
        help_menu.add_command(label="How to Use", command=self._on_help)
        help_menu.add_command(label="The Science", command=self._on_science)
        help_menu.add_command(label="About", command=self._on_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        tests_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg="#333",
            fg="#e0e0e0",
            activebackground="#6a9fb5",
            activeforeground="#fff",
        )
        tests_menu.add_command(label="Take Self-Assessment", command=self._on_bdefs)
        tests_menu.add_command(label="Take Stroop Test", command=self._on_stroop)
        tests_menu.add_separator()
        tests_menu.add_command(label="View Results", command=self._on_view_results)
        menubar.add_cascade(label="Tests", menu=tests_menu)

        # --- Peaceful image banner ---
        self._image_label = tk.Label(
            self.root,
            bg="#2b2b2b",
            height=_IMG_HEIGHT // 8,
        )
        self._image_label.pack(fill=tk.X, padx=10, pady=(6, 0))

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
            bg="#333",
            fg="#e0e0e0",
            selectbackground="#6a9fb5",
            selectforeground="#fff",
            font=("sans-serif", 10),
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
        ttk.Button(
            timer_btn_frame,
            text="Focus 15 min",
            command=self._on_focus,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(timer_btn_frame, text="Break 5 min", command=self._on_break).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(timer_btn_frame, text="Stop", command=self._on_stop_timer).pack(
            side=tk.LEFT, padx=2
        )

        # --- Nudge ---
        nudge_frame = ttk.Frame(self.root)
        nudge_frame.pack(fill=tk.X, **pad)

        self._nudge_label = ttk.Label(
            nudge_frame, text=get_nudge(), style="Nudge.TLabel"
        )
        self._nudge_label.pack(pady=5)

        ttk.Button(nudge_frame, text="New encouragement", command=self._on_nudge).pack()

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def _refresh_tasks(self) -> None:
        """Reload the task list from the database."""
        self._task_listbox.delete(0, tk.END)
        self._task_ids: list[int] = []

        active = db.list_tasks(self.conn, status=TaskStatus.ACTIVE)
        pending = db.list_tasks(self.conn, status=TaskStatus.PENDING)
        done = db.list_tasks(self.conn, status=TaskStatus.DONE)

        for task in active + pending:
            icon = "[~]" if task.status == TaskStatus.ACTIVE else "[ ]"
            prefix = "    " if task.is_subtask else ""
            self._task_listbox.insert(
                tk.END, f"{prefix}{icon} #{task.id}  {task.title}"
            )
            self._task_ids.append(task.id)

        for task in done:
            prefix = "    " if task.is_subtask else ""
            self._task_listbox.insert(tk.END, f"{prefix}[x] #{task.id}  {task.title}")
            self._task_listbox.itemconfig(tk.END, fg="#777777")
            self._task_ids.append(task.id)

    def _refresh_status(self) -> None:
        """Update the status bar."""
        summary = db.get_status(self.conn)
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
            task_in = TaskCreate(title=title.strip())
            db.add_task(self.conn, task_in)
            self._refresh_tasks()
            self._refresh_status()

    def _on_complete_task(self) -> None:
        task_id = self._selected_task_id()
        if task_id is None:
            messagebox.showinfo("Complete", "Select a task first.", parent=self.root)
            return
        db.complete_task(self.conn, task_id)
        self._refresh_tasks()
        self._refresh_status()
        self._nudge_label.configure(text=get_nudge())

    def _on_toggle_task(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Double-click toggles a task between done and pending."""
        task_id = self._selected_task_id()
        if task_id is None:
            return
        task = db.get_task(self.conn, task_id)
        if task is None:
            return
        if task.status == TaskStatus.DONE:
            db.uncomplete_task(self.conn, task_id)
        else:
            db.complete_task(self.conn, task_id)
            self._nudge_label.configure(text=get_nudge())
        self._refresh_tasks()
        self._refresh_status()

    def _on_break_down(self) -> None:
        task_id = self._selected_task_id()
        if task_id is None:
            messagebox.showinfo("Break down", "Select a task first.", parent=self.root)
            return
        step = simpledialog.askstring("Break down", "Add a sub-step:", parent=self.root)
        if step and step.strip():
            sub = TaskCreate(title=step.strip(), parent_id=task_id)
            db.add_task(self.conn, sub)
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
            db.set_task_active(self.conn, task_id)
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
            session_in = FocusSessionCreate(
                task_id=self._timer_task_id, duration_minutes=minutes
            )
            db.log_focus_session(self.conn, session_in)
            self._refresh_status()
            self._nudge_label.configure(text=get_nudge())
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
        self._nudge_label.configure(text=get_nudge())

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _on_settings(self) -> None:
        """Open the settings dialog."""
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("440x480")
        win.configure(bg="#2b2b2b")
        win.transient(self.root)
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

        path_entry = tk.Entry(
            custom_frame,
            bg="#333",
            fg="#e0e0e0",
            insertbackground="#e0e0e0",
            font=("sans-serif", 10),
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
                bg="#2b2b2b",
                fg="#e0e0e0",
                selectcolor="#333",
                activebackground="#2b2b2b",
                activeforeground="#e0e0e0",
                font=("sans-serif", 10),
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
        win.configure(bg="#2b2b2b")
        win.transient(self.root)

        text = scrolledtext.ScrolledText(
            win,
            wrap=tk.WORD,
            bg="#2b2b2b",
            fg="#e0e0e0",
            font=("sans-serif", 10),
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
        widget.tag_configure(
            "h1", font=("sans-serif", 16, "bold"), foreground="#6a9fb5", spacing3=4
        )
        widget.tag_configure(
            "h2",
            font=("sans-serif", 13, "bold"),
            foreground="#6a9fb5",
            spacing1=10,
            spacing3=2,
        )
        widget.tag_configure(
            "h3",
            font=("sans-serif", 11, "bold"),
            foreground="#6a9fb5",
            spacing1=8,
            spacing3=2,
        )
        widget.tag_configure("body", font=("sans-serif", 10), foreground="#e0e0e0")
        widget.tag_configure(
            "bullet",
            font=("sans-serif", 10),
            foreground="#e0e0e0",
            lmargin1=16,
            lmargin2=28,
        )
        widget.tag_configure(
            "code_block",
            font=("monospace", 9),
            foreground="#c5c8c6",
            background="#1e1e1e",
            lmargin1=16,
            lmargin2=16,
            rmargin=16,
            spacing1=4,
            spacing3=4,
        )
        widget.tag_configure(
            "table_row",
            font=("monospace", 9),
            foreground="#e0e0e0",
            lmargin1=8,
            lmargin2=8,
        )
        widget.tag_configure(
            "bold", font=("sans-serif", 10, "bold"), foreground="#e0e0e0"
        )
        widget.tag_configure(
            "inline_code",
            font=("monospace", 9),
            foreground="#c5c8c6",
            background="#1e1e1e",
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
                        font=("sans-serif", 10),
                        foreground="#82b1ff",
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
        win.configure(bg="#2b2b2b")
        win.transient(self.root)

        text = scrolledtext.ScrolledText(
            win,
            wrap=tk.WORD,
            bg="#2b2b2b",
            fg="#e0e0e0",
            font=("sans-serif", 10),
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
            "at a time.",
            parent=self.root,
        )

    # ------------------------------------------------------------------
    # Self-assessment tests
    # ------------------------------------------------------------------

    def _on_bdefs(self) -> None:
        """Run the BDEFS self-assessment in a dialog."""
        win = tk.Toplevel(self.root)
        win.title("Executive Function Self-Assessment")
        win.geometry("560x520")
        win.configure(bg="#2b2b2b")
        win.transient(self.root)
        win.grab_set()

        canvas = tk.Canvas(win, bg="#2b2b2b", highlightthickness=0)
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
                        bg="#2b2b2b",
                        fg="#e0e0e0",
                        selectcolor="#333",
                        activebackground="#2b2b2b",
                        activeforeground="#e0e0e0",
                        font=("sans-serif", 9),
                    ).pack(side=tk.LEFT)
                row += 1
            vars_map[domain] = domain_vars

        def _submit() -> None:
            answers = {d: [v.get() for v in vs] for d, vs in vars_map.items()}
            create_model = score_bdefs(answers)
            saved = db.save_assessment(self.conn, create_model)
            win.destroy()
            self._show_bdefs_result(saved)

        ttk.Button(inner, text="Submit", command=_submit, style="Accent.TButton").grid(
            row=row,
            column=0,
            columnspan=2,
            pady=12,
        )

    def _on_stroop(self) -> None:
        """Run the Stroop colour-word test in a dialog."""
        import time as _time

        trials = generate_stroop_trials()
        state = {"idx": 0, "correct": 0, "total_time": 0.0, "t0": 0.0, "per_trial": []}

        win = tk.Toplevel(self.root)
        win.title("Stroop Colour-Word Test")
        win.geometry("400x300")
        win.configure(bg="#2b2b2b")
        win.transient(self.root)
        win.grab_set()

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
            font=("sans-serif", 36, "bold"),
            bg="#2b2b2b",
        )
        word_label.pack(pady=16)

        entry_var = tk.StringVar()
        entry = tk.Entry(
            win,
            textvariable=entry_var,
            bg="#333",
            fg="#e0e0e0",
            insertbackground="#e0e0e0",
            font=("sans-serif", 14),
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
            saved = db.save_assessment(self.conn, create_model)
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
        """Open a results window with radar chart and score breakdown."""

        past_results = [
            r
            for r in db.list_assessments(
                self.conn, assessment_type=AssessmentType.BDEFS, limit=50
            )
            if r.id != saved.id
        ]

        rwin = tk.Toplevel(self.root)
        rwin.title("Assessment Result")
        rwin.geometry("620x680")
        rwin.configure(bg="#2b2b2b")
        rwin.transient(self.root)

        # Radar chart
        radar_img = bdefs_radar(highlight=saved, past=past_results)
        radar_tk = ImageTk.PhotoImage(radar_img)
        radar_label = tk.Label(rwin, image=radar_tk, bg="#2b2b2b")  # type: ignore[arg-type]
        radar_label.image = radar_tk  # prevent GC
        radar_label.pack(padx=8, pady=(8, 0))

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

        ttk.Label(
            rwin,
            text=interpret_bdefs(saved.score, saved.max_score),
            style="Nudge.TLabel",
        ).pack(padx=16, pady=(6, 4))

        ttk.Button(rwin, text="Close", command=rwin.destroy).pack(pady=(4, 10))

    def _on_view_results(self) -> None:
        """Display past assessment results with charts in a scrollable window."""
        results = db.list_assessments(self.conn, limit=50)

        win = tk.Toplevel(self.root)
        win.title("Assessment Results")
        win.geometry("680x780")
        win.configure(bg="#2b2b2b")
        win.transient(self.root)

        # Scrollable canvas for the whole window
        outer_canvas = tk.Canvas(win, bg="#2b2b2b", highlightthickness=0)
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
            # --- BDEFS radar chart (mean in blue, individual greyed) ---
            if bdefs_results:
                radar_img = bdefs_radar(
                    highlight=None,
                    past=bdefs_results,
                    title="Mean Executive Function Profile",
                )
                radar_tk = ImageTk.PhotoImage(radar_img)
                self._results_images.append(radar_tk)
                tk.Label(content, image=radar_tk, bg="#2b2b2b").pack(
                    padx=8, pady=(8, 0)
                )

            # --- BDEFS timeseries (if 2+ results) ---
            ts_img = bdefs_timeseries(results)
            if ts_img is not None:
                ts_tk = ImageTk.PhotoImage(ts_img)
                self._results_images.append(ts_tk)
                tk.Label(content, image=ts_tk, bg="#2b2b2b").pack(padx=8, pady=(4, 8))

            # --- Textual list of all results ---
            text = scrolledtext.ScrolledText(
                content,
                wrap=tk.WORD,
                bg="#2b2b2b",
                fg="#e0e0e0",
                font=("sans-serif", 10),
                borderwidth=0,
                highlightthickness=0,
                padx=16,
                pady=12,
                height=14,
            )
            text.pack(fill=tk.BOTH, expand=True, padx=4)

            text.tag_configure(
                "heading", font=("sans-serif", 12, "bold"), foreground="#6a9fb5"
            )
            text.tag_configure("body", font=("sans-serif", 10), foreground="#e0e0e0")
            text.tag_configure(
                "interp", font=("sans-serif", 10, "italic"), foreground="#b5b5b5"
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
