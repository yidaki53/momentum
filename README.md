# Momentum

A gentle CLI and GUI tool to help people with executive dysfunction get back on track.

## Design Principles

- **Minimal friction** -- every interaction requires as little effort as possible. If the tool itself demands executive function to use, it defeats the purpose.
- **Non-judgmental tone** -- no guilt-tripping for missed tasks or inactivity. Gentle, matter-of-fact language.
- **Small wins** -- emphasis on micro-steps and visible progress.
- **Grounded in evidence** -- features map to strategies from CBT and executive dysfunction management literature (task decomposition, timed work intervals, break reminders, self-compassion prompts).

## Installation

```bash
make install
```

Or manually:

```bash
poetry install
```

## Quick Start

```bash
# The gentlest way to begin -- suggests a small step
momentum start

# Add a task
momentum add "Write introduction"

# Break a big task into smaller steps
momentum break-down 1

# Mark a task complete
momentum done 1

# See your tasks
momentum list

# Start a 15-minute focus session
momentum focus

# Focus on a specific task for 20 minutes
momentum focus --task 1 --minutes 20

# Take a break
momentum take-break

# See how your day is going
momentum status

# Get a gentle encouragement message
momentum nudge

# Open the GUI dashboard
momentum gui

# Enable autostart on login
momentum autostart --enable

# Sync your data via OneDrive (also: dropbox, google-drive)
momentum config --sync onedrive

# Show current database location
momentum config --show
```

## Commands

| Command | Description |
|---------|-------------|
| `start` | A gentle way to begin. Suggests a small step to get going. |
| `add` | Add a new task. |
| `break-down` | Break a task into smaller sub-steps (interactive). |
| `done` | Mark a task as done. |
| `list` | List your tasks. Use `--all` to include completed. |
| `focus` | Start a focus timer (default 15 min). |
| `take-break` | Take a break (default 5 min). |
| `status` | See tasks completed, focus time, and streak. |
| `nudge` | Get a gentle encouragement message. |
| `gui` | Open the GUI dashboard. |
| `config` | Configure database location / cloud sync. |
| `autostart` | Manage autostart on login (`--enable`, `--disable`, `--status`). |

## GUI

`momentum gui` (or `make gui`) opens a tkinter window with:

- Task list with add/complete/break-down controls
- Focus timer with start/stop
- Status summary (today's progress, streak)
- Encouragement button
- **Menu bar**: Menu (Settings, Quit) and Help (How to Use, About)

The GUI shares the same database as the CLI -- you can use both interchangeably.

## Cloud Sync

Sync your tasks and progress across devices by pointing the database at a cloud-synced folder:

```bash
# Auto-detect a cloud provider folder
momentum config --sync onedrive
momentum config --sync dropbox
momentum config --sync google-drive

# Or set any custom path
momentum config --db-path ~/my/sync/folder/momentum.db

# Check current setting
momentum config --show

# Reset to default local storage
momentum config --reset
```

This is also available in the GUI via Menu > Settings.

## Data

By default, data is stored in `~/.local/share/momentum/momentum.db` (SQLite). This can be changed with `momentum config`. Tasks carry over between days -- there is no guilt for unfinished items.

## Mobile (Android)

A Kivy-based mobile app is in `mobile/`. To build an APK:

```bash
make mobile-deps    # install buildozer + kivy
make mobile-apk     # compile APK (requires Android SDK/NDK)
```

See the [Buildozer docs](https://buildozer.readthedocs.io/en/latest/installation.html) for SDK/NDK setup.

## Make Targets

```bash
make help           # show all targets
make install        # install momentum
make install-dev    # install with dev dependencies
make test           # run tests
make lint           # run ruff linter
make typecheck      # run mypy
make gui            # launch the GUI
make clean          # remove caches and build artefacts
```

## Running Tests

```bash
make test
# or: poetry run pytest tests/ -v
```

## Architecture

- `models.py` -- Pydantic models (single source of truth for all types)
- `db.py` -- SQLite layer (all functions accept/return Pydantic models)
- `config.py` -- App configuration and cloud sync path management
- `cli.py` -- Typer CLI commands
- `gui.py` -- Tkinter GUI (same database, same models)
- `timer.py` -- Focus/break countdown logic
- `encouragement.py` -- Curated CBT/self-compassion message bank
- `display.py` -- Rich terminal formatting
- `autostart.py` -- Systemd/XDG autostart management
- `mobile/main.py` -- Kivy mobile app (Android)
