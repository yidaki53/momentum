"""Application configuration management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from momentum.models import AppConfig

_CONFIG_DIR = Path.home() / ".config" / "momentum"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

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


def load_config() -> AppConfig:
    """Load config from disk, returning defaults if none exists."""
    if _CONFIG_FILE.exists():
        try:
            data = json.loads(_CONFIG_FILE.read_text())
            return AppConfig(**data)
        except (json.JSONDecodeError, Exception):
            pass
    return AppConfig()


def save_config(config: AppConfig) -> Path:
    """Write config to disk. Returns the config file path."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(config.model_dump_json(indent=2))
    return _CONFIG_FILE


def get_db_path() -> Path:
    """Resolve the database path from config (or default)."""
    config = load_config()
    if config.db_path is not None:
        p = Path(config.db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    # Default
    default_dir = Path.home() / ".local" / "share" / "momentum"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir / "momentum.db"


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


def detect_cloud_folder(provider: str) -> Optional[Path]:
    """Try to find a cloud sync folder for the given provider."""
    candidates = _CLOUD_PRESETS.get(provider.lower(), [])
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
