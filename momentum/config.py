"""Application configuration management."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from momentum.models import AppConfig, ThemeMode, TimerCycleMode


def _is_android() -> bool:
    """Return True when running inside an Android (p4a) environment."""
    return "ANDROID_ARGUMENT" in os.environ or hasattr(sys, "getandroidapilevel")


def _android_data_dir() -> Path:
    """Return the stable writable app-private directory on Android."""
    try:
        from jnius import autoclass

        python_activity = autoclass("org.kivy.android.PythonActivity")
        activity = python_activity.mActivity
        if activity is not None:
            files_dir = activity.getFilesDir()
            if files_dir is not None:
                return Path(files_dir.getAbsolutePath())
    except Exception:
        pass

    for var in ("ANDROID_PRIVATE", "ANDROID_APP_PATH"):
        val = os.environ.get(var)
        if val:
            return Path(val)
    return Path(".")  # last resort


def _android_legacy_data_dirs() -> list[Path]:
    """Return legacy Android data roots that may contain older config or DB files."""
    candidates: list[Path] = []
    for var in ("ANDROID_PRIVATE", "ANDROID_APP_PATH"):
        val = os.environ.get(var)
        if val:
            candidates.append(Path(val) / "data")
    candidates.append(Path(".") / "data")

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


if _is_android():
    _DATA_DIR = _android_data_dir() / "data"
    _CONFIG_DIR = _DATA_DIR / "config"
    _DB_DIR = _DATA_DIR / "db"
    _LEGACY_CONFIG_FILES = [
        legacy_dir / "config" / "config.json"
        for legacy_dir in _android_legacy_data_dirs()
    ]
    _LEGACY_DB_FILES = [
        legacy_dir / "db" / "momentum.db" for legacy_dir in _android_legacy_data_dirs()
    ]
else:
    _CONFIG_DIR = Path.home() / ".config" / "momentum"
    _DB_DIR = Path.home() / ".local" / "share" / "momentum"
    _LEGACY_CONFIG_FILES: list[Path] = []
    _LEGACY_DB_FILES: list[Path] = []

_CONFIG_FILE = _CONFIG_DIR / "config.json"


def _migrate_legacy_file(target: Path, legacy_candidates: list[Path]) -> None:
    """Copy a legacy file forward to the current storage location when needed."""
    if target.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    for candidate in legacy_candidates:
        if not candidate.exists():
            continue
        try:
            if candidate.resolve() == target.resolve():
                continue
        except OSError:
            pass
        shutil.copy2(candidate, target)
        return


# Well-known cloud sync directories (checked in order)
_CLOUD_PRESETS: dict[str, list[Path]] = {
    "onedrive": [
        Path.home() / "OneDrive",
        Path.home() / "onedrive",
    ],
    "dropbox": [
        Path.home() / "Dropbox",
        Path.home() / "dropbox",
    ],
    "google-drive": [
        Path.home() / "Google Drive",
        Path.home() / "google-drive",
    ],
}

_CLOUD_PROVIDER_ALIASES: dict[str, str] = {
    "one-drive": "onedrive",
    "one drive": "onedrive",
    "google drive": "google-drive",
    "google_drive": "google-drive",
    "googledrive": "google-drive",
    "gdrive": "google-drive",
    "drop box": "dropbox",
}

_ANDROID_CLOUD_PRESETS: dict[str, list[Path]] = {
    "onedrive": [
        Path("OneDrive"),
        Path("Android") / "media" / "com.microsoft.skydrive",
    ],
    "dropbox": [Path("Dropbox")],
    "google-drive": [Path("Google Drive"), Path("google-drive")],
}


def _canonical_cloud_provider(provider: str) -> str:
    normalized = provider.strip().lower().replace("_", "-")
    return _CLOUD_PROVIDER_ALIASES.get(normalized, normalized)


def _cloud_search_roots() -> list[Path]:
    roots: list[Path] = [Path.home()]
    if _is_android():
        for var in ("EXTERNAL_STORAGE", "SECONDARY_STORAGE", "ANDROID_STORAGE"):
            raw = os.environ.get(var, "")
            for entry in raw.split(os.pathsep):
                if entry:
                    roots.append(Path(entry))
        roots.extend(
            [
                Path("/storage/emulated/0"),
                Path("/storage/self/primary"),
                Path("/sdcard"),
            ]
        )

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def _cloud_candidates(provider: str) -> list[Path]:
    canonical = _canonical_cloud_provider(provider)
    candidates = list(_CLOUD_PRESETS.get(canonical, []))
    if _is_android():
        for root in _cloud_search_roots():
            for relative in _ANDROID_CLOUD_PRESETS.get(canonical, []):
                candidates.append(root / relative)
    return candidates


def load_config() -> AppConfig:
    """Load config from disk, returning defaults if none exists."""
    _migrate_legacy_file(_CONFIG_FILE, _LEGACY_CONFIG_FILES)
    if _CONFIG_FILE.exists():
        try:
            data = json.loads(_CONFIG_FILE.read_text())
            return AppConfig(**data)
        except (json.JSONDecodeError, Exception):
            pass
    return AppConfig()


def save_config(config: AppConfig) -> Path:
    """Write config to disk. Returns the config file path."""
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if hasattr(config, "model_dump_json"):
            payload = config.model_dump_json(indent=2)
        elif hasattr(config, "json"):
            payload = config.json(indent=2)
        else:
            payload = json.dumps(config.dict(), indent=2)
        _CONFIG_FILE.write_text(payload, encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Could not write config file at {_CONFIG_FILE}") from exc
    return _CONFIG_FILE


def get_db_path() -> Path:
    """Resolve the database path from config (or default)."""
    config = load_config()
    if config.db_path is not None:
        p = Path(config.db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    # Default
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    default_path = _DB_DIR / "momentum.db"
    _migrate_legacy_file(default_path, _LEGACY_DB_FILES)
    return default_path


def set_db_path(path: str) -> AppConfig:
    """Set a custom database path and save config."""
    resolved = Path(path).expanduser().resolve()
    # Ensure it ends with a filename
    if resolved.is_dir():
        resolved = resolved / "momentum.db"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    config = load_config()
    config.db_path = str(resolved)
    save_config(config)
    return config


def set_timer_cycle_mode(mode: str) -> AppConfig:
    """Persist timer cycle mode preference."""
    config = load_config()
    try:
        config.timer_cycle_mode = TimerCycleMode(mode)
    except ValueError as exc:
        raise ValueError(
            f"Invalid timer cycle mode '{mode}'. Expected one of: "
            f"{', '.join(m.value for m in TimerCycleMode)}"
        ) from exc
    save_config(config)
    return config


def detect_cloud_folder(provider: str) -> Optional[Path]:
    """Try to find a cloud sync folder for the given provider."""
    candidates = _cloud_candidates(provider)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def set_cloud_sync(provider: str) -> Optional[AppConfig]:
    """Configure the DB to live inside a cloud provider's sync folder.

    Returns the config if successful, None if the folder wasn't found.
    """
    folder = detect_cloud_folder(provider)
    if folder is None:
        return None
    db_dir = folder / "momentum"
    db_dir.mkdir(parents=True, exist_ok=True)
    return set_db_path(str(db_dir / "momentum.db"))


def reset_db_path() -> AppConfig:
    """Reset to the default local database path."""
    config = load_config()
    config.db_path = None
    save_config(config)
    return config


def set_theme_mode(mode: str) -> AppConfig:
    """Persist visual theme mode."""
    config = load_config()
    try:
        config.theme_mode = ThemeMode(mode)
    except ValueError as exc:
        raise ValueError(
            f"Invalid theme mode '{mode}'. Expected one of: "
            f"{', '.join(m.value for m in ThemeMode)}"
        ) from exc
    save_config(config)
    return config


def set_accessibility_options(
    *,
    large_text: Optional[bool] = None,
    high_contrast: Optional[bool] = None,
    reduce_visual_load: Optional[bool] = None,
) -> AppConfig:
    """Persist accessibility options."""
    config = load_config()
    if large_text is not None:
        config.accessibility_large_text = large_text
    if high_contrast is not None:
        config.accessibility_high_contrast = high_contrast
    if reduce_visual_load is not None:
        config.accessibility_reduce_visual_load = reduce_visual_load
    save_config(config)
    return config


def set_check_updates_at_startup(enabled: bool) -> AppConfig:
    """Persist update check preference."""
    config = load_config()
    config.check_updates_at_startup = enabled
    save_config(config)
    return config
