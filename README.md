# Momentum

A gentle CLI and GUI tool to help people with executive dysfunction get back on track.

## Design Principles

- **Minimal friction** -- every interaction requires as little effort as possible. If the tool itself demands executive function to use, it defeats the purpose.
- **Non-judgmental tone** -- no guilt-tripping for missed tasks or inactivity. Gentle, matter-of-fact language.
- **Small wins** -- emphasis on micro-steps and visible progress.
- **Grounded in evidence** -- features map to strategies from CBT and executive dysfunction management literature (task decomposition, timed work intervals, break reminders, self-compassion prompts).

## Installation

### Debian/Ubuntu (.deb)

Download the `.deb` from the latest [GitHub release](../../releases) and install:

```bash
sudo dpkg -i momentum_*_amd64.deb
```

This installs the `momentum` binary to `/usr/local/bin` and adds a desktop menu entry.

### From source

```bash
make install
# or: poetry install
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
| `test` | Take a self-assessment (BDEFS or `--stroop`). |
| `test-results` | View past assessment results. |
| `autostart` | Manage autostart on login (`--enable`, `--disable`, `--status`). |

## Self-Assessment Tests

Momentum includes two evidence-based self-assessment tools:

- **BDEFS** -- A brief executive-function questionnaire covering time management, organisation, self-restraint, self-motivation, and emotion regulation. Results are visualised as a radar chart and tracked over time with a trend line.
- **Stroop** -- A timed colour-word test measuring inhibitory control. Accuracy and response times are recorded.

Both tests include instruction pages before starting. Past results can be viewed with charts and interpretations.

```bash
momentum test              # BDEFS self-assessment
momentum test --stroop     # Stroop colour-word test
momentum test-results      # View past results
```

## GUI

`momentum gui` (or `make gui`) opens a tkinter window with:

- Task list with add/complete/break-down controls
- Focus timer with start/stop
- Status summary (today's progress, streak)
- Encouragement button
- **Menu bar**: Menu (Settings, Quit), Help (How to Use, The Science, About), and Tests (BDEFS, Stroop, View Results)

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

A full-featured Kivy-based Android app is in `mobile/`. It mirrors all desktop GUI features: task management, focus timer, BDEFS and Stroop assessments with charts, settings, and help pages.

Pre-built APKs are attached to [GitHub releases](../../releases). To install, download `momentum-android.apk` and sideload it.

To build locally:

```bash
make mobile-deps    # install buildozer + kivy
make mobile-apk     # compile APK (requires Android SDK/NDK + Java 17)
```

The APK is also built automatically by GitHub Actions CI on every push to master.

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
- `charts.py` -- Matplotlib radar and timeseries charts with trend lines
- `mobile/main.py` -- Kivy mobile app (Android)
