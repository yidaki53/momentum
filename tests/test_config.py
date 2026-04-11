"""Tests for the config module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from momentum.config import (
    detect_cloud_folder,
    get_db_path,
    load_config,
    reset_db_path,
    save_config,
    set_accessibility_options,
    set_check_updates_at_startup,
    set_cloud_sync,
    set_db_path,
    set_theme_mode,
    set_timer_cycle_mode,
)
from momentum.models import AppConfig, ThemeMode, TimerCycleMode, WindowPosition


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
            cfg = AppConfig(
                db_path="/tmp/test.db", window_position=WindowPosition.TOP_LEFT
            )
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

    def test_load_migrates_legacy_android_config(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        legacy_file = tmp_path / "legacy" / "config" / "config.json"
        legacy_file.parent.mkdir(parents=True, exist_ok=True)
        legacy_file.write_text('{"check_updates_at_startup": false}', encoding="utf-8")
        with p1, p2, patch("momentum.config._LEGACY_CONFIG_FILES", [legacy_file]):
            config = load_config()
            assert config.check_updates_at_startup is False
            assert (tmp_path / "config" / "config.json").exists()


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

    def test_default_path_migrates_legacy_android_db(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        db_dir = tmp_path / "db"
        legacy_db = tmp_path / "legacy" / "db" / "momentum.db"
        legacy_db.parent.mkdir(parents=True, exist_ok=True)
        legacy_db.write_text("legacy-db", encoding="utf-8")
        with (
            p1,
            p2,
            patch("momentum.config._DB_DIR", db_dir),
            patch("momentum.config._LEGACY_DB_FILES", [legacy_db]),
        ):
            path = get_db_path()
            assert path == db_dir / "momentum.db"
            assert path.read_text(encoding="utf-8") == "legacy-db"


class TestCloudSync:
    def test_detect_cloud_folder_found(self, tmp_path: Path) -> None:
        od = tmp_path / "OneDrive"
        od.mkdir()
        presets = {"onedrive": [od]}
        with patch("momentum.config._CLOUD_PRESETS", presets):
            assert detect_cloud_folder("onedrive") == od

    def test_detect_cloud_folder_not_found(self) -> None:
        with patch(
            "momentum.config._CLOUD_PRESETS", {"onedrive": [Path("/nonexistent/path")]}
        ):
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
        with (
            p1,
            p2,
            patch("momentum.config._CLOUD_PRESETS", {"onedrive": [Path("/no")]}),
        ):
            assert set_cloud_sync("onedrive") is None


class TestSettingsPersistence:
    def test_set_theme_mode_persists(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        with p1, p2:
            cfg = set_theme_mode(ThemeMode.LIGHT.value)

            assert cfg.theme_mode == ThemeMode.LIGHT
            assert load_config().theme_mode == ThemeMode.LIGHT

    def test_set_theme_mode_rejects_invalid_value(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        with p1, p2:
            try:
                set_theme_mode("sepia")
            except ValueError as exc:
                assert "Invalid theme mode 'sepia'" in str(exc)
            else:
                raise AssertionError("Expected invalid theme mode to raise ValueError")

    def test_set_timer_cycle_mode_persists(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        with p1, p2:
            cfg = set_timer_cycle_mode(TimerCycleMode.AUTO.value)

            assert cfg.timer_cycle_mode == TimerCycleMode.AUTO
            assert load_config().timer_cycle_mode == TimerCycleMode.AUTO

    def test_set_timer_cycle_mode_rejects_invalid_value(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        with p1, p2:
            try:
                set_timer_cycle_mode("chaos")
            except ValueError as exc:
                assert "Invalid timer cycle mode 'chaos'" in str(exc)
            else:
                raise AssertionError(
                    "Expected invalid timer cycle mode to raise ValueError"
                )

    def test_set_accessibility_options_persists_all_flags(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        with p1, p2:
            cfg = set_accessibility_options(
                large_text=True,
                high_contrast=True,
                reduce_visual_load=True,
            )

            assert cfg.accessibility_large_text is True
            assert cfg.accessibility_high_contrast is True
            assert cfg.accessibility_reduce_visual_load is True

            loaded = load_config()
            assert loaded.accessibility_large_text is True
            assert loaded.accessibility_high_contrast is True
            assert loaded.accessibility_reduce_visual_load is True

    def test_set_check_updates_at_startup_persists(self, tmp_path: Path) -> None:
        p1, p2 = _patch_config_paths(tmp_path)
        with p1, p2:
            cfg = set_check_updates_at_startup(False)

            assert cfg.check_updates_at_startup is False
            assert load_config().check_updates_at_startup is False
