"""Startup data recovery and manual backups.

Guards against the two ways user data has historically gone "missing":

1. An update changes the storage layout and the new version creates a fresh
   empty database before anyone notices the old one — the old file then sits
   orphaned forever because migration only copies when the target is absent.
2. A config reset (corrupt/unparsable config.json) loses a custom ``db_path``
   while the real database remains untouched at the old path.

:func:`recover_default_db` runs once per process, only for the *default*
database path, and only when that database has no tasks and no assessments:
it then scans known legacy locations plus a depth-limited walk of the
app-private tree and adopts the richest Momentum database it finds.

This module deliberately uses raw ``sqlite3`` (never ``momentum.db``) so it
can be imported from ``momentum.db`` without an import cycle.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Optional

from momentum import config as cfg

log = logging.getLogger(__name__)

# Recursion cap for the app-private tree walk (data dirs are tiny; the cap
# only exists so a pathological tree can never stall startup).
_MAX_SCAN_DEPTH = 8

# Directory names never worth descending into (caches/models/logs contain no
# user databases and may be large).
_SKIP_DIR_NAMES = {
    "models",
    "cache",
    "__pycache__",
    "log",
    "logs",
    "tmp",
    ".git",
}

_recovery_done = False


def reset_recovery_state() -> None:
    """Testing hook: allow the once-per-process guard to run again."""
    global _recovery_done
    _recovery_done = False


def count_user_rows(db_path: Path) -> Optional[tuple[int, int]]:
    """Return ``(tasks, assessments)`` for an on-disk DB, or None if unusable.

    Read-only open: never creates or modifies the file it inspects.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except (sqlite3.Error, OSError):
        return None
    try:
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        except sqlite3.Error:
            return None
        if "tasks" not in tables and "assessments" not in tables:
            return None
        tasks = assessments = 0
        if "tasks" in tables:
            tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        if "assessments" in tables:
            assessments = conn.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
        return int(tasks), int(assessments)
    except (sqlite3.Error, IndexError, TypeError):
        return None
    finally:
        conn.close()


def _deep_scan(root: Path) -> list[Path]:
    """Depth-limited walk under *root* collecting ``momentum.db`` files."""
    found: list[Path] = []
    if not root.is_dir():
        return found
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        if len(current.parts) - root_depth >= _MAX_SCAN_DEPTH:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames if d not in _SKIP_DIR_NAMES and not d.startswith(".")
        ]
        if "momentum.db" in filenames:
            found.append(current / "momentum.db")
    return found


def _known_layouts() -> list[Path]:
    """Fixed, non-recursive candidate locations (cloud sync, custom layouts)."""
    candidates: list[Path] = []
    # Cloud-sync layout produced by config.set_cloud_sync(): <folder>/momentum/.
    for paths in cfg._CLOUD_PRESETS.values():
        for root in paths:
            candidates.append(root / "momentum" / "momentum.db")
    # Historic layouts relative to the app-data root's parent (e.g. ancient
    # builds that stored the DB directly under the files dir, or used the
    # desktop-style ~/.local/share/momentum path inside the sandbox). Fixed
    # paths only — the parent itself is never deep-walked.
    parent = cfg.app_data_root().parent
    for relative in (
        ("momentum.db",),
        ("db", "momentum.db"),
        ("data", "db", "momentum.db"),
        (".local", "share", "momentum", "momentum.db"),
    ):
        candidates.append(parent.joinpath(*relative))
    # Android external files dir (storage scope for API 29+ devices).
    if cfg._is_android():
        try:
            from jnius import autoclass  # type: ignore[import-not-found]

            activity = autoclass("org.kivy.android.PythonActivity").mActivity
            external = activity.getExternalFilesDir(None) if activity else None
            if external is not None:
                candidates.append(
                    Path(external.getAbsolutePath()) / "data" / "db" / "momentum.db"
                )
        except Exception:
            pass
    return candidates


def find_candidate_dbs(current: Path) -> list[Path]:
    """Locate Momentum databases other than *current*, richest data first."""
    try:
        current_key = str(current.resolve())
    except OSError:
        current_key = str(current)

    roots: list[Path] = [cfg.app_data_root()]
    if cfg._is_android():
        roots.extend(cfg._android_legacy_data_dirs())
    else:
        roots.append(cfg._DB_DIR)

    seen: set[str] = set()
    candidates: list[Path] = []
    for root in roots:
        for db_file in _deep_scan(root):
            try:
                key = str(db_file.resolve())
            except OSError:
                key = str(db_file)
            if key == current_key or key in seen or not db_file.is_file():
                continue
            seen.add(key)
            candidates.append(db_file)
    for db_file in _known_layouts():
        try:
            key = str(db_file.resolve())
        except OSError:
            key = str(db_file)
        if key == current_key or key in seen or not db_file.is_file():
            continue
        seen.add(key)
        candidates.append(db_file)

    def _richness(db_file: Path) -> tuple[int, int]:
        counts = count_user_rows(db_file)
        return counts if counts else (0, 0)

    candidates.sort(key=_richness, reverse=True)
    return candidates


def recover_default_db(default_path: Path) -> bool:
    """Adopt the richest orphaned DB into *default_path* when it is empty.

    Returns True when a recovery copy was performed. Safe to call often:
    runs its scan at most once per process, and never touches a default
    database that already contains user rows.
    """
    global _recovery_done
    if _recovery_done:
        return False
    _recovery_done = True
    # Tests (and support tooling) can opt out entirely; see tests/conftest.py.
    if os.environ.get("MOMENTUM_DISABLE_RECOVERY"):
        return False

    if default_path.exists():
        counts = count_user_rows(default_path)
        if counts is None:
            # Existing file is unreadable/not a DB: leave diagnosis to sqlite
            # on first real open rather than overwriting evidence.
            log.warning(
                "Default DB at %s exists but is unreadable; not recovering",
                default_path,
            )
            return False
        if counts[0] > 0 or counts[1] > 0:
            return False  # current DB has user data — nothing to do

    for candidate in find_candidate_dbs(default_path):
        cand_counts = count_user_rows(candidate)
        if not cand_counts or (cand_counts[0] == 0 and cand_counts[1] == 0):
            continue
        # Adopt: preserve whatever empty/unreadable file currently sits at the
        # target, then copy the candidate (plus WAL sidecars) into place.
        default_path.parent.mkdir(parents=True, exist_ok=True)
        if default_path.exists():
            aside = default_path.with_name(default_path.name + ".empty.bak")
            try:
                shutil.move(str(default_path), str(aside))
            except OSError:
                default_path.unlink(missing_ok=True)
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(candidate) + suffix)
            if src.exists():
                shutil.copy2(src, Path(str(default_path) + suffix))
        log.warning(
            "Recovered user data: adopted %s (%d tasks, %d assessments) into %s",
            candidate,
            cand_counts[0],
            cand_counts[1],
            default_path,
        )
        return True
    return False


def export_backup(dest_dir: Path) -> Path:
    """Copy the current database + config into *dest_dir* with a timestamp.

    Returns the path of the copied database file.
    """
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest_dir.mkdir(parents=True, exist_ok=True)
    db_src = cfg.get_db_path()
    dest_db = dest_dir / f"momentum-backup-{stamp}.db"
    if db_src.exists():
        # Consistent copy even if the DB is mid-write.
        src_conn = sqlite3.connect(str(db_src))
        try:
            dest_conn = sqlite3.connect(str(dest_db))
            try:
                src_conn.backup(dest_conn)
            finally:
                dest_conn.close()
        finally:
            src_conn.close()
    else:
        dest_db.touch()
    if cfg._CONFIG_FILE.exists():
        shutil.copy2(cfg._CONFIG_FILE, dest_dir / f"config-backup-{stamp}.json")
    log.info("Exported backup to %s", dest_db)
    return dest_db


def restore_backup(backup_db: Path) -> Path:
    """Restore *backup_db* over the current default database.

    The existing database (if any) is preserved as ``momentum.db.pre-restore``
    first. The caller must reopen any live connection afterwards.
    """
    counts = count_user_rows(backup_db)
    if counts is None:
        raise ValueError(f"Not a valid Momentum database: {backup_db}")
    target = cfg.get_db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        aside = target.with_name(target.name + ".pre-restore")
        shutil.copy2(target, aside)
    shutil.copy2(backup_db, target)
    log.warning(
        "Restored backup %s over %s (%d tasks, %d assessments)",
        backup_db,
        target,
        counts[0],
        counts[1],
    )
    return target
