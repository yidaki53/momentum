"""Tests for the config module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from momentum.config import (
    _CONFIG_DIR,
    _CONFIG_FILE,
    detect_cloud_folder,
    get_db_path,
    load_config,
    reset_db_path,
    save_config,
    set_cloud_sync,
    set_db_path,
)
from momentum.models import AppConfig, WindowPosition


def _patch_config_paths(tmp_path: Path):
    """Return a context manager that redirects config dir/file to tmp_path."""
    cfg_dir = tmp_path / "config"
    cfg_file = cfg_dir / "config.json"
    return (
        patch("momentum.config._CONFIG_DIR", cfg_dir),
        patch("momentum.config._CONFIG_FILE", cfg_file),
    )


class TestLoadSaveConfig:
    def test_load_default_when_missing(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        with p1, p2:
            config = load_config()
            assert config.db_path is None
            assert config.window_position == WindowPosition.CENTRE

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        with p1, p2:
            cfg = AppConfig(db_path="/tmp/test.db", window_position=WindowPosition.TOP_LEFT)
            path = save_config(cfg)
            assert path.exists()

            loaded = load_config()
            assert loaded.db_path == "/tmp/test.db"
            assert loaded.window_position == WindowPosition.TOP_LEFT

    def test_load_handles_corrupt_file(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        with p1, p2:
            cfg_dir = tmp_path / "config"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "config.json").write_text("not valid json{{{")
            config = load_config()
            assert config.db_path is None  # falls back to default


class TestDbPath:
    def test_default_path(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        with p1, p2:
            path = get_db_path()
            assert path.name == "momentum.db"

    def test_set_db_path(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        with p1, p2:
            custom = tmp_path / "custom" / "my.db"
            cfg = set_db_path(str(custom))
            assert cfg.db_path == str(custom)
            assert get_db_path() == custom

    def test_set_db_path_directory(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        with p1, p2:
            d = tmp_path / "somedir"
            d.mkdir()
            cfg = set_db_path(str(d))
            assert cfg.db_path is not None
            assert cfg.db_path.endswith("momentum.db")

    def test_reset_db_path(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        with p1, p2:
            set_db_path(str(tmp_path / "custom.db"))
            cfg = reset_db_path()
            assert cfg.db_path is None


class TestCloudSync:
    def test_detect_cloud_folder_found(self, tmp_path: Path) -> None:
        od = tmp_path / "OneDrive"
        od.mkdir()
        presets = {"onedrive": [od]}
        with patch("momentum.config._CLOUD_PRESETS", presets):
            assert detect_cloud_folder("onedrive") == od

    def test_detect_cloud_folder_not_found(self) -> None:
        with patch("momentum.config._CLOUD_PRESETS", {"onedrive": [Path("/nonexistent/path")]}):
            assert detect_cloud_folder("onedrive") is None

    def test_detect_unknown_provider(self) -> None:
        assert detect_cloud_folder("icloud") is None

    def test_set_cloud_sync_success(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        od = tmp_path / "OneDrive"
        od.mkdir()
        presets = {"onedrive": [od]}
        with p1, p2, patch("momentum.config._CLOUD_PRESETS", presets):
            cfg = set_cloud_sync("onedrive")
            assert cfg is not None
            assert "OneDrive" in cfg.db_path  # type: ignore[operator]

    def test_set_cloud_sync_not_found(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        with p1, p2, patch("momentum.config._CLOUD_PRESETS", {"onedrive": [Path("/no")]}):
            assert set_cloud_sync("onedrive") is None
