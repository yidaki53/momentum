"""In-place self-update of a frozen Momentum binary.

This module downloads the correct platform asset for the latest GitHub
release, swaps it in for the running executable, and relaunches the app.
It is deliberately separate from ``update_check.py`` so the detection
layer stays pure (network metadata only) and this layer owns the
filesystem + process side effects.

Non-frozen (development ``poetry run``) builds, non-writable install
paths (e.g. a system ``.deb`` owned by root), and missing platform
assets all fall back to ``FALLBACK_NOTIFY`` -- the caller opens the
releases page in a browser instead of attempting a self-replace.
"""

from __future__ import annotations

import enum
import os
import platform
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from momentum.ui.update_check import (
    ReleaseInfo,
    certifi_ssl_context,
    is_certificate_verification_error,
)

ProgressCallback = Callable[[int, int], None]
# (downloaded_bytes, total_bytes); total_bytes is -1 when unknown.

# Map platform.system() -> release asset filename (matches ci.yml artifacts).
_ASSET_NAMES: dict[str, str] = {
    "Linux": "momentum-linux",
    "Darwin": "momentum-macos",
    "Windows": "momentum-windows.exe",
}


class SelfUpdateStatus(str, enum.Enum):
    """Outcome of a self-update attempt."""

    UPDATED = "updated"
    FALLBACK_NOTIFY = "fallback_notify"
    ERROR = "error"


@dataclass(frozen=True)
class SelfUpdateResult:
    """Structured result returned by ``perform_self_update``."""

    status: SelfUpdateStatus
    message: str = ""
    new_version: str = ""
    releases_url: str = ""


def is_frozen_build() -> bool:
    """Return True when running inside a PyInstaller-frozen binary."""
    return bool(getattr(sys, "frozen", False))


def current_executable_path() -> Optional[str]:
    """Return the path of the running executable, or None when not frozen."""
    if not is_frozen_build():
        return None
    return sys.executable


def select_asset_for_current_platform(
    assets: list[tuple[str, str]],
) -> Optional[str]:
    """Return the download URL for the asset matching this platform, or None."""
    target = _ASSET_NAMES.get(platform.system())
    if target is None:
        return None
    for name, url in assets:
        if name == target:
            return url
    return None


def _stream_to_file(
    request: urllib.request.Request,
    dest: str,
    timeout: float,
    ssl_context: Optional[ssl.SSLContext],
    progress_callback: Optional[ProgressCallback],
) -> None:
    with urllib.request.urlopen(
        request, timeout=timeout, context=ssl_context
    ) as response:
        total = int(response.headers.get("Content-Length") or -1)
        with open(dest, "wb") as fh:
            downloaded = 0
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if progress_callback is not None:
                    progress_callback(downloaded, total)


def download_asset(
    url: str,
    dest: str,
    progress_callback: Optional[ProgressCallback] = None,
    timeout: float = 60.0,
) -> None:
    """Stream a release asset to *dest*, retrying with certifi on cert errors."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "Momentum self-update"}
    )
    try:
        _stream_to_file(request, dest, timeout, None, progress_callback)
    except urllib.error.URLError as exc:
        if not is_certificate_verification_error(exc):
            raise
        certifi_context = certifi_ssl_context()
        if certifi_context is None:
            raise
        _stream_to_file(request, dest, timeout, certifi_context, progress_callback)


def cleanup_old_binary() -> None:
    """Remove a leftover ``<exe>.old`` from a previous Windows self-update."""
    if not is_frozen_build() or platform.system() != "Windows":
        return
    old_path = sys.executable + ".old"
    try:
        if os.path.exists(old_path):
            os.remove(old_path)
    except OSError:
        # The old binary may still be locked if the previous process has not
        # fully exited; it will be cleaned up on a later run.
        pass


def replace_and_relaunch(
    new_path: str,
    current_exe: str,
    restart_args: Optional[list[str]] = None,
) -> subprocess.Popen[bytes]:
    """Swap *new_path* into *current_exe* and launch the new binary.

    On Unix, ``os.replace`` atomically swaps the file while the running
    process keeps the old inode. On Windows the running exe is locked, so
    the current binary is renamed aside to ``<exe>.old`` (cleaned up on the
    next start by ``cleanup_old_binary``) before the new one is moved in.
    """
    args = restart_args or []
    if platform.system() == "Windows":
        old_path = current_exe + ".old"
        try:
            if os.path.exists(old_path):
                os.remove(old_path)
        except OSError:
            pass
        os.rename(current_exe, old_path)
        os.rename(new_path, current_exe)
    else:
        # os.replace requires same-filesystem; callers download *new_path*
        # into the same directory as *current_exe* to satisfy this.
        os.replace(new_path, current_exe)

    return subprocess.Popen([current_exe, *args])


def perform_self_update(
    release_info: ReleaseInfo,
    restart_args: Optional[list[str]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> SelfUpdateResult:
    """Detect, download, replace, and relaunch in one step.

    Returns ``UPDATED`` when the new binary was swapped in and relaunched,
    ``FALLBACK_NOTIFY`` when a self-replace is not possible (non-frozen
    build, non-writable path, no matching asset) and the caller should
    instead open the releases page, or ``ERROR`` on a download/replace
    failure.
    """
    fallback = SelfUpdateResult(
        status=SelfUpdateStatus.FALLBACK_NOTIFY,
        new_version=release_info.version,
        releases_url=release_info.url,
    )

    if not is_frozen_build():
        return fallback

    current_exe = current_executable_path()
    if current_exe is None or not os.access(current_exe, os.W_OK):
        return fallback

    asset_url = select_asset_for_current_platform(release_info.assets)
    if asset_url is None:
        return fallback

    # Download into the same directory as the target so os.replace/rename
    # stay on one filesystem (a cross-device rename raises OSError).
    dest = os.path.join(os.path.dirname(current_exe), ".momentum.update.download")
    try:
        try:
            download_asset(asset_url, dest, progress_callback)
        except Exception as exc:
            return SelfUpdateResult(
                status=SelfUpdateStatus.ERROR,
                message=f"Download failed: {exc}",
                new_version=release_info.version,
                releases_url=release_info.url,
            )
        try:
            replace_and_relaunch(dest, current_exe, restart_args)
        except OSError as exc:
            return SelfUpdateResult(
                status=SelfUpdateStatus.ERROR,
                message=f"Replace failed: {exc}",
                new_version=release_info.version,
                releases_url=release_info.url,
            )
        return SelfUpdateResult(
            status=SelfUpdateStatus.UPDATED,
            new_version=release_info.version,
            releases_url=release_info.url,
        )
    finally:
        # On success the temp file has been moved away by replace/rename;
        # this only cleans up leftovers from a failed attempt.
        try:
            if os.path.exists(dest):
                os.remove(dest)
        except OSError:
            pass


__all__ = [
    "ProgressCallback",
    "ReleaseInfo",
    "SelfUpdateResult",
    "SelfUpdateStatus",
    "cleanup_old_binary",
    "current_executable_path",
    "download_asset",
    "is_frozen_build",
    "perform_self_update",
    "replace_and_relaunch",
    "select_asset_for_current_platform",
]
