"""Tests for the autostart module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from momentum.autostart import (
    _desktop_entry_path,
    _service_path,
    _systemd_dir,
    _xdg_autostart_dir,
    disable_autostart,
    enable_autostart,
    get_autostart_status,
)


class TestPaths:
    def test_systemd_dir(self) -> None:
        d = _systemd_dir()
        assert d.parts[-3:] == (".config", "systemd", "user")

    def test_xdg_autostart_dir(self) -> None:
        d = _xdg_autostart_dir()
        assert d.parts[-2:] == (".config", "autostart")

    def test_service_path(self) -> None:
        p = _service_path()
        assert p.name == "momentum-gui.service"

    def test_desktop_entry_path(self) -> None:
        p = _desktop_entry_path()
        assert p.name == "momentum-gui.desktop"


class TestEnableAutostart:
    def test_no_momentum_binary(self) -> None:
        with patch("momentum.autostart._find_momentum_bin", return_value=None):
            status = enable_autostart()
            assert not status.systemd_enabled
            assert not status.xdg_enabled

    def test_enable_success(self, tmp_path: Path) -> None:
        svc_dir = tmp_path / "systemd" / "user"
        xdg_dir = tmp_path / "autostart"
        with (
            patch("momentum.autostart._find_momentum_bin", return_value="/usr/bin/momentum"),
            patch("momentum.autostart._systemd_dir", return_value=svc_dir),
            patch("momentum.autostart._xdg_autostart_dir", return_value=xdg_dir),
            patch("momentum.autostart._service_path", return_value=svc_dir / "momentum-gui.service"),
            patch("momentum.autostart._desktop_entry_path", return_value=xdg_dir / "momentum-gui.desktop"),
            patch("subprocess.run") as mock_run,
        ):
            status = enable_autostart()
            assert status.systemd_enabled
            assert status.xdg_enabled
            assert status.service_path is not None
            assert status.desktop_entry_path is not None
            # Verify systemd commands were called
            assert mock_run.call_count == 2  # daemon-reload + enable

    def test_enable_systemd_failure(self, tmp_path: Path) -> None:
        svc_dir = tmp_path / "systemd" / "user"
        xdg_dir = tmp_path / "autostart"
        with (
            patch("momentum.autostart._find_momentum_bin", return_value="/usr/bin/momentum"),
            patch("momentum.autostart._systemd_dir", return_value=svc_dir),
            patch("momentum.autostart._xdg_autostart_dir", return_value=xdg_dir),
            patch("momentum.autostart._service_path", return_value=svc_dir / "momentum-gui.service"),
            patch("momentum.autostart._desktop_entry_path", return_value=xdg_dir / "momentum-gui.desktop"),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            status = enable_autostart()
            assert not status.systemd_enabled
            # XDG should still succeed
            assert status.xdg_enabled


class TestDisableAutostart:
    def test_disable_clean(self, tmp_path: Path) -> None:
        svc = tmp_path / "momentum-gui.service"
        entry = tmp_path / "momentum-gui.desktop"
        svc.write_text("test")
        entry.write_text("test")
        with (
            patch("momentum.autostart._service_path", return_value=svc),
            patch("momentum.autostart._desktop_entry_path", return_value=entry),
            patch("subprocess.run"),
        ):
            disable_autostart()
            assert not svc.exists()
            assert not entry.exists()

    def test_disable_no_files(self, tmp_path: Path) -> None:
        svc = tmp_path / "momentum-gui.service"
        entry = tmp_path / "momentum-gui.desktop"
        with (
            patch("momentum.autostart._service_path", return_value=svc),
            patch("momentum.autostart._desktop_entry_path", return_value=entry),
            patch("subprocess.run"),
        ):
            disable_autostart()  # should not raise

    def test_disable_systemctl_missing(self, tmp_path: Path) -> None:
        svc = tmp_path / "momentum-gui.service"
        entry = tmp_path / "momentum-gui.desktop"
        with (
            patch("momentum.autostart._service_path", return_value=svc),
            patch("momentum.autostart._desktop_entry_path", return_value=entry),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            disable_autostart()  # should not raise


class TestGetAutostartStatus:
    def test_nothing_installed(self, tmp_path: Path) -> None:
        svc = tmp_path / "momentum-gui.service"
        entry = tmp_path / "momentum-gui.desktop"
        with (
            patch("momentum.autostart._service_path", return_value=svc),
            patch("momentum.autostart._desktop_entry_path", return_value=entry),
        ):
            status = get_autostart_status()
            assert not status.systemd_enabled
            assert not status.xdg_enabled

    def test_service_enabled(self, tmp_path: Path) -> None:
        svc = tmp_path / "momentum-gui.service"
        svc.write_text("[Unit]\n")
        entry = tmp_path / "momentum-gui.desktop"
        entry.write_text("[Desktop Entry]\n")
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        with (
            patch("momentum.autostart._service_path", return_value=svc),
            patch("momentum.autostart._desktop_entry_path", return_value=entry),
            patch("subprocess.run", return_value=mock_proc),
        ):
            status = get_autostart_status()
            assert status.systemd_enabled
            assert status.xdg_enabled

    def test_service_present_but_disabled(self, tmp_path: Path) -> None:
        svc = tmp_path / "momentum-gui.service"
        svc.write_text("[Unit]\n")
        entry = tmp_path / "momentum-gui.desktop"
        mock_proc = MagicMock()
        mock_proc.returncode = 1  # not enabled
        with (
            patch("momentum.autostart._service_path", return_value=svc),
            patch("momentum.autostart._desktop_entry_path", return_value=entry),
            patch("subprocess.run", return_value=mock_proc),
        ):
            status = get_autostart_status()
            assert not status.systemd_enabled

    def test_systemctl_not_found(self, tmp_path: Path) -> None:
        svc = tmp_path / "momentum-gui.service"
        svc.write_text("[Unit]\n")
        entry = tmp_path / "momentum-gui.desktop"
        with (
            patch("momentum.autostart._service_path", return_value=svc),
            patch("momentum.autostart._desktop_entry_path", return_value=entry),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            status = get_autostart_status()
            assert not status.systemd_enabled
