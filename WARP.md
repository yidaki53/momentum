# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Maintaining This Document

**IMPORTANT**: This WARP.md file should be kept up to date as the project evolves. When you learn important patterns, workflows, or solutions to problems:

1. **Document learnings immediately**: Add new insights, debugging solutions, or workflow improvements to the appropriate section
2. **Update after significant changes**: When adding new scripts, changing architecture, or establishing new patterns, update the relevant sections
3. **Keep it practical**: Focus on information that will help WARP (or future developers) work more effectively with this codebase
4. **Maintain structure**: Add new sections as needed but preserve the existing organization

Examples of what to document:
- New modules and their purpose
- Solutions to common errors or debugging challenges
- New dependencies or setup requirements
- Changes to the data model or database schema
- Important configuration changes
- Workflow improvements or new commands

## Critical Rules

**NEVER commit or push to git.** The user handles all git operations manually. Do not run `git commit`, `git push`, or any commands that modify the git history.

**Use poetry, not uv.** This repo is managed with Poetry — run all Python commands via `poetry run ...` (e.g. `poetry run pytest`, `poetry run ruff check`, `poetry run mypy`). Do not install or use `uv` here; the pre-commit hooks, Makefile, and CI all use poetry, and the project venv lives at `.venv` (a Poetry virtualenv symlink).

## Project Metadata

### Author / Contact
- Author: Robin Oberg
- Email: robinoberg@live.com

## Project Overview

Momentum is a CLI and GUI tool designed to help people with executive dysfunction (stemming from depression or similar conditions) get back on track. It provides gentle task management, focus timers with breaks, self-assessment tests (BDEFS and Stroop), encouragement messages grounded in CBT and self-compassion principles, and data visualisation of assessment progress over time.

### Core Features
- **Task management**: Add, list, break down, and complete tasks
- **Focus timer**: Pomodoro-style sessions with configurable durations
- **Self-assessments**: BDEFS executive function questionnaire and Stroop colour-word test, with instruction pages and results guide
- **Encouragement**: 750+ curated messages from a user-editable markdown file
- **GUI dashboard**: Tkinter-based dark-themed GUI with nature banner images, timers, task list, and assessment charts
- **Data visualisation**: Radar/spider charts and timeseries plots with trend line for BDEFS results
- **Cloud sync**: Configure DB storage in OneDrive, Dropbox, or Google Drive folders
- **Autostart**: Systemd user service and XDG autostart desktop entry
- **Android app**: Full-featured Kivy-based mobile app with all desktop features (buildozer APK)

## Repository Architecture

```
momentum/
├── momentum/                    # Main Python package
│   ├── __init__.py
│   ├── cli.py                   # Typer CLI entry point (all commands)
│   ├── gui.py                   # Tkinter GUI dashboard
│   ├── db.py                    # SQLite database layer
│   ├── models.py                # Pydantic models (single source of truth)
│   ├── assessments.py           # BDEFS and Stroop test logic
│   ├── charts.py                # Matplotlib radar and timeseries charts
│   ├── config.py                # App config (DB path, cloud sync, window position)
│   ├── display.py               # Rich terminal formatting helpers
│   ├── encouragement.py         # Loads messages from ENCOURAGEMENTS.md
│   ├── timer.py                 # Focus/break countdown logic
│   └── autostart.py             # Systemd + XDG autostart management
├── tests/                       # Test suite (115 tests, 79% coverage)
│   ├── test_assessments.py
│   ├── test_autostart.py
│   ├── test_charts.py
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_db.py
│   ├── test_models.py
│   └── test_timer.py
├── mobile/                      # Kivy mobile app scaffold
├── .github/workflows/ci.yml     # GitHub Actions: test + build binary
├── ENCOURAGEMENTS.md            # 750+ user-editable encouragement messages
├── IMAGES.md                    # 218 Unsplash photo IDs for GUI banner
├── Makefile                     # install, test, lint, gui, dist, clean
├── pyproject.toml               # Poetry config, deps, coverage settings
└── .gitattributes               # Git LFS tracking for dist/momentum
```

## Development Setup

### Environment Management
```bash
# Install dependencies
poetry install --with dev

# Or run commands directly
poetry run momentum --help
```

### Makefile Targets
```bash
make install       # Install into Poetry virtualenv
make install-dev   # Install with dev dependencies
make test          # Run tests with verbose output
make lint          # Run ruff linter
make typecheck     # Run mypy type checking
make gui           # Launch the GUI
make dist          # Build standalone binary with PyInstaller
make clean         # Remove caches and build artefacts
make help          # Show all targets
```

### Core CLI Commands
```bash
poetry run momentum add "Write introduction"
poetry run momentum list
poetry run momentum done 1
poetry run momentum focus --minutes 15 --task 1
poetry run momentum take-break
poetry run momentum status
poetry run momentum nudge
poetry run momentum start            # Guided gentle start
poetry run momentum test             # BDEFS self-assessment
poetry run momentum test --stroop    # Stroop colour-word test
poetry run momentum test-results     # View past assessment results
poetry run momentum config --show    # Show current config
poetry run momentum config --sync onedrive
poetry run momentum autostart --enable
poetry run momentum gui              # Launch GUI dashboard
```

## Testing

### Running Tests
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=momentum --cov-report=term-missing -q

# Run with coverage threshold enforcement
poetry run pytest --cov=momentum --cov-fail-under=70 -q

# Run specific test file
poetry run pytest tests/test_assessments.py -v
```

### Coverage Configuration
- `gui.py` is excluded from coverage measurement (standard for tkinter code)
- Minimum threshold: 70% (configured in `pyproject.toml` under `[tool.coverage]`)
- Current coverage: ~79% across 115 tests

### Code Quality
```bash
poetry run ruff check momentum/ tests/
poetry run mypy momentum/
```

## Database

### Schema (SQLite)
Four tables: `tasks`, `focus_sessions`, `daily_log`, `assessments`.

### Storage Location
- Default: `~/.local/share/momentum/momentum.db`
- Cloud sync: Configurable to OneDrive/Dropbox/Google Drive via `momentum config --sync <provider>`
- Custom path: `momentum config --db-path /path/to/file.db`
- Config stored at: `~/.config/momentum/config.json`

## User-Editable Content

### ENCOURAGEMENTS.md
- 750+ bullet-point messages organised by theme (CBT, self-compassion, executive dysfunction, etc.)
- Loaded at import time by `encouragement.py` via `_load_messages()`
- Falls back to 11 hardcoded messages if file is missing
- Format: `- Message text here`

### IMAGES.md
- 218 Unsplash photo IDs across 12 nature categories (lakes, forests, beaches, etc.)
- Loaded by `gui.py` via `_load_photos()` with deduplication
- Falls back to 7 hardcoded IDs if file is missing
- Format: `- photo-XXXXXXXXXX-XXXXXXXXXXXX`
- `_fetch_image()` retries up to 5 different photos on HTTP failure

## GUI Details

### Window Sizes
- Main window: 520x720
- BDEFS result window: 620x680
- View Results window: 680x780
- Banner image: 500x120

### Theme
Dark theme with palette: background `#2b2b2b`, text `#e0e0e0`, accent `#6a9fb5`, timer `#e8c547`.

### Charts (charts.py)
- `bdefs_radar()`: 540x440 spider chart of BDEFS domain scores
- `bdefs_timeseries()`: 560x240 line chart of score over time with linear trend line (numpy polyfit)
- Both use matplotlib Agg backend with the dark GUI palette

## Build and Distribution

### PyInstaller Binary
```bash
make dist
# Produces: dist/momentum (standalone Linux binary)
```
- Bundles ENCOURAGEMENTS.md and IMAGES.md as data files
- Binary tracked by Git LFS via `.gitattributes`

### GitHub Actions CI (.github/workflows/ci.yml)
- **On push/PR to master**: lint + test with 70% coverage gate
- **On push to master only**: build PyInstaller binaries (Linux/macOS/Windows) + Android APK, create GitHub Release with all artifacts
- APK build requires Java 17, Cython 0.29.36, and buildozer
- Uses `softprops/action-gh-release@v2` for release creation

## Configuration

### pyproject.toml Key Settings
- Python: >=3.11,<4.0
- Dependencies: typer (with extras), rich, pydantic, Pillow, matplotlib
- Dev dependencies: pytest, pytest-cov, ruff, mypy, pyinstaller
- Coverage: omits `gui.py`, fail_under=70

## Troubleshooting

### Poetry Environment Issues
```bash
poetry env remove --all
poetry install
poetry env info
```

### Unsplash Image 404s
Photo IDs in IMAGES.md can go stale. The `_fetch_image()` method retries up to 5 random photos. To scan for broken IDs:
```bash
while IFS= read -r line; do
  id=$(echo "$line" | sed 's/^- //')
  [[ "$id" == photo-* ]] || continue
  code=$(curl -s -o /dev/null -w "%{http_code}" "https://images.unsplash.com/$id?w=500&h=120&fit=crop")
  [ "$code" != "200" ] && echo "BAD $id ($code)"
done < IMAGES.md
```

### GUI Won't Launch
- Ensure tkinter is installed: `sudo apt install python3-tk`
- Ensure Pillow is installed: `poetry install`
- Check for display: GUI requires a graphical session (X11/Wayland)

## Recent Learnings

### Unsplash Photo ID Cleanup (2026-02-27)
- 51 out of 269 original photo IDs were returning HTTP 404
- Removed all bad IDs, leaving 218 valid IDs in IMAGES.md
- Also removed one bad ID from the `_FALLBACK_PHOTOS` list in gui.py
- Added retry logic to `_fetch_image()` to handle future stale IDs gracefully

### Coverage Improvement (2026-02-27)
- Went from 22% to 79% coverage by adding tests for assessments, charts, config, autostart, and more CLI/DB commands
- GUI module excluded from coverage (tkinter is impractical to unit test)

### CI: LFS bandwidth budget fix (2026-07-25)
- The `test` job failed at `actions/checkout@v5` with `This repository exceeded its LFS budget` because checkout had `lfs: true` and the only LFS-tracked file (`dist/momentum`) was re-fetched every run.
- Fix: set `lfs: false` on the `test` and `build` (matrix) checkouts. CI rebuilds `dist/momentum` via PyInstaller each run, so no job needs the committed LFS object.

### CI: APK build -- numpy v-prefix checkout fix (2026-07-25)
- Root cause: `build-apk` failed with `error: pathspec '1.26.4' did not match any file(s) known to git` during `git checkout 1.26.4`. numpy git tags are v-prefixed (`v1.26.4`), but the `numpy==1.26.4` requirements pin made p4a run `git checkout 1.26.4`.
- Key discovery: buildozer **clones python-for-android fresh from GitHub** into `/tmp/buildozer-build/android/platform/python-for-android/` and ignores both the pip-installed p4a and `buildozer.spec`'s `p4a.source` (a site-packages path is not a git repo, so buildozer falls back to cloning). The earlier "Patch p4a numpy recipe" step patched the pip-installed p4a that was never used, and was also a no-op (that recipe's version was already `v2.3.0`). The `numpy==1.26.4` pin overrides the recipe's `version` field, so patching the recipe version never affected the checkout tag.
- Fix (primary): pin `numpy==v1.26.4` in `mobile/buildozer.spec` so p4a runs `git checkout v1.26.4` directly.
- Fix (fallback): the "Build APK" step captures pass 1's log; if it shows a numpy version/checkout error (`pathspec` / `did not match` / `InvalidVersion`), it patches the **cloned** p4a's numpy recipe `version` to `v1.26.4` via `sed` and unpins `numpy` in `buildozer.spec` so pass 2 uses the recipe version. Removed the useless pip-installed patch step and the `p4a.source` CI override.
- Two-pass buildozer retained for Cython injection (hostpython3's isolated site-packages only exists after pass 1 builds hostpython3).

### CI: APK build -- pydantic-core maturin ANDROID_API_LEVEL fix (2026-07-25)
- Root cause: `build-apk` failed at the pydantic-core recipe with `maturin failed: Failed to determine Android API level. Please set the ANDROID_API_LEVEL environment variable.` The Rust crate itself compiled fine (`Finished release profile [optimized]`); only the wheel platform-tag step failed. pydantic-core has no published Android wheel on PyPI, so p4a cross-compiles it from source via `python -m build` -> maturin. p4a exports `ANDROIDAPI`/`NDKAPI` but **not** `ANDROID_API_LEVEL`, which is the specific var maturin reads to tag the wheel `android_26_arm`.
- Red herring: initially suspected a pydantic-core version regression and pinned `pydantic==2.12.5,pydantic-core==2.41.5` (the 2026-02-28-era pair). The pin was a no-op -- the failure was identical on 2.41.4 and 2.41.5. The pins are kept for reproducible mobile builds but were not the fix.
- Fix: set `ANDROID_API_LEVEL: "26"` in the `env:` of both the "Build APK" and "Build AAB" steps in `.github/workflows/ci.yml`. Value 26 matches `android.minapi = 26` / `--ndk-api=26` / the `android_26_arm` platform tag p4a expects. The env var propagates into maturin's isolated build venv.

### Android APK Build (2026-02-28)
- Successfully built 80MB APK with numpy/matplotlib for chart support
- **Critical Cython fix**: python-for-android sets `PYTHONNOUSERSITE=1` and overrides `PYTHONPATH` to only `.../hostpython3/native-build/Lib/site-packages/`. Cython must be installed into that exact directory: `hostpython3 -m pip install --target=".../native-build/Lib/site-packages" Cython==0.29.36`
- Cython 3.x is incompatible with numpy 1.22.3 (the p4a recipe version) -- must use Cython 0.29.36
- Build requires Java 17 (`JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64`)
- Spaces in project path break python-for-android; `build_dir` must point to a path without spaces
- `config.py` detects Android via `ANDROID_ARGUMENT` env var and uses `ANDROID_PRIVATE` for writable data dir
- Mobile symlinks: `mobile/momentum -> ../momentum`, plus ENCOURAGEMENTS.md, SCIENCE.md, README.md, IMAGES.md

### Test Instructions and Results Guide (2026-02-28)
- Added BDEFS_INSTRUCTIONS, STROOP_INSTRUCTIONS, and RESULTS_GUIDE constants in `assessments.py`
- Shared across CLI, GUI, and mobile -- single source of truth for instruction text
- GUI shows instruction dialog (OK/Cancel) before starting each test
- Mobile shows instruction popup (Stroop) or inline text (BDEFS) before the test
- Results screens include a guide explaining how to interpret charts and scores

### Timeseries Trend Line (2026-02-28)
- Added linear regression (numpy polyfit) trend line to `bdefs_timeseries()` chart
- Dashed grey line with "Trend" legend label

## Mobile App (Android)

### Architecture
- `mobile/main.py`: Single-file Kivy app with 10 screens via ScreenManager
- Screens: Home, Settings, HelpMenu, HowTo, Science, About, TestsMenu, BDEFS, Stroop, Results
- Toolbar with Home/Settings/Help/Tests buttons on every screen
- Reuses core `momentum/` package via symlink

### Building Locally
```bash
# Requires buildozer venv at ~/.buildozer-venv with system-site-packages
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export TMPDIR=/media/robin/persistence/tmp
cd mobile
/home/robin/.buildozer-venv/bin/buildozer android debug
```

### Presplash & Icon
- `icon.png`: 512x512 blue rounded-rect "M" generated by PIL
- `presplash.png`: 480x800 dark-themed loading screen with M logo and "Momentum" text
- Both configured in `buildozer.spec`

## Citations and References
- Rule: User uses poetry to run python CLI commands.
- Rule: Do not add any emojis in responses or communication as they are considered unprofessional.

<citations>
  <document>
      <document_type>RULE</document_type>
      <document_id>BHurptshQeZDUZ5WW8LovY</document_id>
  </document>
  <document>
      <document_type>RULE</document_type>
      <document_id>dNYglGzBX1l800iPUpmTfa</document_id>
  </document>
</citations>
