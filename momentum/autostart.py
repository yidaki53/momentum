"""Autostart management: systemd user service + XDG autostart desktop entry."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from momentum.models import AutostartStatus

_SERVICE_NAME = "momentum-gui.service"
_DESKTOP_ENTRY_NAME = "momentum-gui.desktop"


def _systemd_dir() -> Path:
    """Return the systemd user unit directory."""
    return Path.home() / ".config" / "systemd" / "user"


def _xdg_autostart_dir() -> Path:
    """Return the XDG autostart directory."""
    return Path.home() / ".config" / "autostart"


def _find_momentum_bin() -> Optional[str]:
    """Locate the momentum executable."""
    return shutil.which("momentum")


def _service_path() -> Path:
    return _systemd_dir() / _SERVICE_NAME


def _desktop_entry_path() -> Path:
    return _xdg_autostart_dir() / _DESKTOP_ENTRY_NAME


def _write_systemd_service(bin_path: str) -> Path:
    """Write the systemd user service file."""
    path = _service_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""\
[Unit]
Description=Momentum GUI - Executive Dysfunction Support
After=graphical-session.target

[Service]
Type=simple
ExecStart={bin_path} gui
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""
    path.write_text(content)
    return path


def _write_desktop_entry(bin_path: str) -> Path:
    """Write the XDG autostart desktop entry."""
    path = _desktop_entry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""\
[Desktop Entry]
Type=Application
Name=Momentum
Comment=Executive dysfunction support tool
Exec={bin_path} gui
Terminal=false
Categories=Utility;
X-GNOME-Autostart-enabled=true
"""
    path.write_text(content)
    return path


def enable_autostart() -> AutostartStatus:
    """Install and enable systemd service + XDG autostart entry."""
    bin_path = _find_momentum_bin()
    if bin_path is None:
        return AutostartStatus()

    result = AutostartStatus()

    # Systemd
    try:
        svc_path = _write_systemd_service(bin_path)
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["systemctl", "--user", "enable", _SERVICE_NAME],
            check=True,
            capture_output=True,
        )
        result.systemd_enabled = True
        result.service_path = str(svc_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # XDG
    try:
        entry_path = _write_desktop_entry(bin_path)
        result.xdg_enabled = True
        result.desktop_entry_path = str(entry_path)
    except OSError:
        pass

    return result


def disable_autostart() -> None:
    """Remove the systemd service and XDG autostart entry."""
    # Systemd
    try:
        subprocess.run(
            ["systemctl", "--user", "disable", _SERVICE_NAME],
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        pass

    svc = _service_path()
    if svc.exists():
        svc.unlink()

    try:
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        pass

    # XDG
    entry = _desktop_entry_path()
    if entry.exists():
        entry.unlink()


def get_autostart_status() -> AutostartStatus:
    """Check the current state of autostart configuration."""
    result = AutostartStatus()

    svc = _service_path()
    if svc.exists():
        result.service_path = str(svc)
        try:
            proc = subprocess.run(
                ["systemctl", "--user", "is-enabled", _SERVICE_NAME],
                capture_output=True,
                text=True,
            )
            result.systemd_enabled = proc.returncode == 0
        except FileNotFoundError:
            pass

    entry = _desktop_entry_path()
    if entry.exists():
        result.xdg_enabled = True
        result.desktop_entry_path = str(entry)

    return result
