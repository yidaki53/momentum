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
# Switch timer flow between manual and automatic focus-break cycling
momentum config --cycle-mode auto

# Save an ACT journal check-in
momentum journal

# Review recent ACT journal entries
momentum journal --list

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
| `focus` | Start a focus timer (default 15 min, personalised from latest BIS/BAS + BDEFS + Stroop signals). |
| `take-break` | Take a break (default 5 min, personalised from latest BIS/BAS + BDEFS + Stroop signals). |
| `status` | See tasks completed, focus time, and streak. |
| `nudge` | Get a gentle encouragement message. |
| `journal` | Create or review structured ACT journal entries (`--list`). |
| `gui` | Open the GUI dashboard. |
| `config` | Configure database location / cloud sync / timer cycle mode (`--cycle-mode manual|auto`). |
| `test` | Take a self-assessment (BDEFS, `--stroop`, or `--bisbas`). |
| `test-results` | View past assessment results. |
| `about` | Show copyright, license, and author information. |
| `science` | Display the science behind Momentum (SCIENCE.md). |
| `delete-results` | Delete all saved assessment results. |
| `delete-tasks` | Delete all tasks and focus sessions. |
| `browse-db` | Browse database tables and list entries. |
| `delete-entry` | Delete a specific entry by table and ID. |
| `autostart` | Manage autostart on login (`--enable`, `--disable`, `--status`). |

## Self-Assessment Tests

Momentum includes three evidence-based self-assessment tools:

- **BDEFS** -- A brief executive-function questionnaire covering time management, organisation, self-restraint, self-motivation, and emotion regulation. Results are visualised as a radar chart and tracked over time with a trend line.
- **Stroop** -- A timed colour-word test measuring inhibitory control. Accuracy and response times are recorded.
- **BIS/BAS** -- A motivational-style profile (Behavioural Inhibition / Behavioural Activation) shown as a bar chart with reference lines (guidance anchors, not diagnostic cutoffs), plus tailored encouragement and practical tips.

Personalization now blends recent BIS/BAS, BDEFS, and Stroop data to tune focus length, break length, and nudge style. All tests include instruction pages before starting, and past results can be viewed with charts and interpretations.

```bash
momentum test              # BDEFS self-assessment
momentum test --stroop     # Stroop colour-word test
momentum test --bisbas     # BIS/BAS motivational profile
momentum test-results      # View past results
```

## GUI

`momentum gui` (or `make gui`) opens a tkinter window with:

- Task list with add/complete/break-down controls
- Focus timer with start/stop and manual/auto cycle mode
- Status summary (today's progress, streak)
- Encouragement button
- ACT journal check-in + ACT journal history
- **Menu bar**: Menu (Settings, Quit), Help (How to Use, The Science, About), and Tests (BDEFS, BIS/BAS, Stroop, View Results)
- BIS/BAS result windows and history charts with reference-line disclaimer + bespoke guidance text
- Appearance settings for dark/light mode and accessibility options (larger text, higher contrast, reduced visual load)

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

A full-featured Kivy-based Android app is in `mobile/`. It mirrors desktop capabilities with touch-focused navigation and includes:

- Bottom navigation toolbar (Home, Settings, Help, Tests)
- Task management + focus timer + manual/auto cycle mode + personalised nudges
- ACT journaling (quick check-in + history)
- BDEFS, BIS/BAS, and Stroop assessments with result history
- BIS/BAS bar chart results with reference-line disclaimer and bespoke guidance
- Light/dark theme and accessibility options (larger text, higher contrast, reduced visual load)
- Shared local data model with CLI and desktop GUI

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
- `assessments.py` -- BDEFS/BIS-BAS/Stroop scoring, interpretation, domain-specific advice, and personalization helpers
- `charts.py` -- Matplotlib charts (BDEFS radar/timeseries and BIS/BAS profile bars with reference lines)
- `mobile/main.py` -- Kivy mobile app (Android)
