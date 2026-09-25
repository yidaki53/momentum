"""Application configuration management."""

from __future__ import annotations

import dataclasses
import enum
import json
import logging
import os
import re
import shutil
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from momentum.models import AppConfig, ThemeMode, TimerCycleMode, WindowPosition

log = logging.getLogger(__name__)


def _is_android() -> bool:
    """Return True when running inside an Android (p4a) environment."""
    return "ANDROID_ARGUMENT" in os.environ or hasattr(sys, "getandroidapilevel")


def _android_data_dir() -> Path:
    """Return the stable writable app-private directory on Android."""
    try:
        from jnius import autoclass  # type: ignore[import-not-found]

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


def app_data_root() -> Path:
    """Return the root directory for all Momentum user data on this platform.

    On Android this is the app-private ``getFilesDir()`` (preserved across app
    updates and uninstallable-only-by-uninstall), so the DB, config, and
    downloaded models all live here. On desktop it is the XDG-style
    ``~/.local/share/momentum`` location (``_DB_DIR``'s parent) so behaviour is
    unchanged. Callers that need a sub-tree (e.g. the LLM model cache) should use
    ``app_data_root() / "<subdir>"``.
    """
    if _is_android():
        return _android_data_dir() / "data"
    return Path.home() / ".local" / "share" / "momentum"


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


_LEGACY_CONFIG_FILES: list[Path]
_LEGACY_DB_FILES: list[Path]

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
    _LEGACY_CONFIG_FILES = []
    _LEGACY_DB_FILES = []

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


# Per-field expected types for lenient config loading. Anything not listed
# here falls through to a best-effort assignment guarded by AppConfig's own
# __post_init__ coercion.
_ENUM_FIELDS = {
    "window_position": WindowPosition,
    "theme_mode": ThemeMode,
    "timer_cycle_mode": TimerCycleMode,
}
_BOOL_FIELDS = {
    "accessibility_large_text",
    "accessibility_high_contrast",
    "accessibility_reduce_visual_load",
    "check_updates_at_startup",
    "show_llm_welcome",
    "llm_enabled",
}
_INT_FIELDS = {"last_update_check_unix"}
_STR_FIELDS = {"llm_model"}
_OPTIONAL_STR_FIELDS = {"db_path"}


def _backup_corrupt_config() -> None:
    """Preserve an unreadable config next to the original (config.json.bak).

    The old loader silently discarded the file's *contents* on any parse
    error, which could throw away a custom ``db_path`` and make the app open
    a fresh empty database while the user's real data sat untouched at the
    old path. The .bak copy guarantees the raw bytes always survive.
    """
    try:
        bak = _CONFIG_FILE.with_name(_CONFIG_FILE.name + ".bak")
        if not bak.exists():
            shutil.copy2(_CONFIG_FILE, bak)
            log.warning("Config unreadable; preserved original at %s", bak)
    except OSError:
        log.warning("Could not back up corrupt config at %s", _CONFIG_FILE)


def _salvage_db_path(raw: str) -> Optional[str]:
    """Extract ``db_path`` from raw (possibly corrupt) config text.

    Last-resort regex salvage so a hard JSON parse failure still cannot lose
    the pointer to the user's database.
    """
    m = re.search(r'"db_path"\s*:\s*(null|"((?:[^"\\]|\\.)*)")', raw)
    if not m:
        return None
    if m.group(1) == "null":
        return None
    try:
        value = json.loads('"' + m.group(2) + '"')
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, str) and value else None


def _load_config_lenient(data: dict[str, Any]) -> AppConfig:
    """Build an AppConfig keeping every valid field, dropping only bad ones.

    ``AppConfig(**data)`` raises ``TypeError`` on unknown keys and
    ``ValueError`` on invalid enum values — previously either turned into a
    *total* config reset. Now one bad field only loses that field.
    """
    known = {f.name for f in dataclasses.fields(AppConfig)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            continue
        if key in _ENUM_FIELDS:
            enum_cls = _ENUM_FIELDS[key]
            if isinstance(value, enum_cls):
                kwargs[key] = value
            else:
                try:
                    kwargs[key] = enum_cls(value)
                except (ValueError, KeyError, TypeError):
                    log.warning(
                        "Config field %s has invalid value %r; using default",
                        key,
                        value,
                    )
        elif key in _BOOL_FIELDS:
            if isinstance(value, bool):
                kwargs[key] = value
            else:
                log.warning("Config field %s is not a bool; using default", key)
        elif key in _INT_FIELDS:
            if isinstance(value, int) and not isinstance(value, bool):
                kwargs[key] = value
        elif key in _STR_FIELDS:
            if isinstance(value, str):
                kwargs[key] = value
        elif key in _OPTIONAL_STR_FIELDS:
            if value is None or isinstance(value, str):
                kwargs[key] = value
        else:
            kwargs[key] = value
    return AppConfig(**kwargs)


def load_config() -> AppConfig:
    """Load config from disk, returning defaults if none exists.

    Never silently discards a partially-valid file: unknown keys and invalid
    values fall back to defaults field-by-field, and a totally unreadable
    file is preserved as ``config.json.bak`` while any salvageable
    ``db_path`` is still honoured.
    """
    _migrate_legacy_file(_CONFIG_FILE, _LEGACY_CONFIG_FILES)
    if not _CONFIG_FILE.exists():
        return AppConfig()
    try:
        raw = _CONFIG_FILE.read_text(encoding="utf-8")
    except OSError:
        return AppConfig()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise json.JSONDecodeError("config root is not an object", "<config>", 0)
        return _load_config_lenient(data)
    except json.JSONDecodeError:
        _backup_corrupt_config()
        salvaged = _salvage_db_path(raw)
        if salvaged:
            log.warning("Recovered db_path from corrupt config: %s", salvaged)
            return AppConfig(db_path=salvaged)
        return AppConfig()


def save_config(config: AppConfig) -> Path:
    """Write config to disk. Returns the config file path."""
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        def _json_default(obj: object) -> object:
            if isinstance(obj, enum.Enum):
                return obj.value
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serialisable"
            )

        payload = json.dumps(
            dataclasses.asdict(config), indent=2, default=_json_default
        )
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


def set_llm_enabled(enabled: bool) -> AppConfig:
    """Persist whether the optional AI Coach is enabled."""
    config = load_config()
    config.llm_enabled = bool(enabled)
    save_config(config)
    return config
