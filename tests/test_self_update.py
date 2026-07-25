"""Tests for the in-place self-update module.

No test ever performs a real download, file replace, or process relaunch:
``os.replace``/``os.rename``/``subprocess.Popen`` and the network call are
monkeypatched so the platform-specific side effects stay hermetic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from momentum.ui import self_update
from momentum.ui.self_update import (
    ReleaseInfo,
    SelfUpdateStatus,
)


def _release(version: str = "0.5.0") -> ReleaseInfo:
    return ReleaseInfo(
        version=version,
        url="https://example.com/release",
        assets=[("momentum-linux", "https://example.com/momentum-linux")],
    )


class TestIsFrozenBuild:
    def test_false_without_frozen_attr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delattr(sys, "frozen", raising=False)
        assert self_update.is_frozen_build() is False

    def test_true_with_frozen_attr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        assert self_update.is_frozen_build() is True


class TestCurrentExecutablePath:
    def test_none_when_not_frozen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delattr(sys, "frozen", raising=False)
        assert self_update.current_executable_path() is None

    def test_executable_when_frozen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        assert self_update.current_executable_path() == sys.executable


class TestSelectAsset:
    def test_linux_asset_selected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(self_update.platform, "system", lambda: "Linux")
        assets = [
            ("momentum-linux", "https://example.com/linux"),
            ("momentum-macos", "https://example.com/macos"),
        ]
        assert (
            self_update.select_asset_for_current_platform(assets)
            == "https://example.com/linux"
        )

    def test_unknown_platform_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(self_update.platform, "system", lambda: "Plan9")
        assert self_update.select_asset_for_current_platform([]) is None

    def test_missing_asset_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(self_update.platform, "system", lambda: "Linux")
        assert (
            self_update.select_asset_for_current_platform(
                [("momentum-macos", "https://example.com/macos")]
            )
            is None
        )


class TestReplaceAndRelaunch:
    def test_unix_uses_os_replace_and_popen(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        new_path = tmp_path / "new"
        new_path.write_text("new")
        exe = tmp_path / "momentum"
        exe.write_text("old")

        monkeypatch.setattr(self_update.platform, "system", lambda: "Linux")
        replace_calls: list[tuple[str, str]] = []

        def fake_replace(src: str, dst: str) -> None:
            replace_calls.append((src, dst))
            Path(dst).write_text(Path(src).read_text())

        monkeypatch.setattr(self_update.os, "replace", fake_replace)
        popen = MagicMock()
        monkeypatch.setattr(self_update.subprocess, "Popen", lambda args: popen)

        result = self_update.replace_and_relaunch(
            str(new_path), str(exe), restart_args=["gui"]
        )

        assert result is popen
        assert replace_calls == [(str(new_path), str(exe))]


class TestPerformSelfUpdate:
    def test_fallback_when_not_frozen(self) -> None:
        # Dev/poetry runs are never frozen, so this exercises the real guard.
        result = self_update.perform_self_update(_release())
        assert result.status is SelfUpdateStatus.FALLBACK_NOTIFY
        assert result.new_version == "0.5.0"

    def test_updated_when_frozen(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exe = tmp_path / "momentum"
        exe.write_text("old")

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(self_update, "current_executable_path", lambda: str(exe))
        monkeypatch.setattr(
            self_update,
            "select_asset_for_current_platform",
            lambda assets: "https://example.com/linux",
        )

        def fake_download(
            url: str, dest: str, progress_callback=None, timeout: float = 60.0
        ) -> None:
            Path(dest).write_text("new binary")
            if progress_callback is not None:
                progress_callback(4, 4)

        monkeypatch.setattr(self_update, "download_asset", fake_download)
        popen = MagicMock()
        monkeypatch.setattr(
            self_update,
            "replace_and_relaunch",
            lambda new_path, current_exe, restart_args=None: popen,
        )

        result = self_update.perform_self_update(_release(), restart_args=["gui"])

        assert result.status is SelfUpdateStatus.UPDATED
        assert result.new_version == "0.5.0"

    def test_error_on_download_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exe = tmp_path / "momentum"
        exe.write_text("old")

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(self_update, "current_executable_path", lambda: str(exe))
        monkeypatch.setattr(
            self_update,
            "select_asset_for_current_platform",
            lambda assets: "https://example.com/linux",
        )

        def bad_download(
            url: str, dest: str, progress_callback=None, timeout: float = 60.0
        ) -> None:
            raise OSError("network down")

        monkeypatch.setattr(self_update, "download_asset", bad_download)

        result = self_update.perform_self_update(_release())

        assert result.status is SelfUpdateStatus.ERROR
        assert "Download failed" in result.message

    def test_fallback_when_no_matching_asset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exe = tmp_path / "momentum"
        exe.write_text("old")

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(self_update, "current_executable_path", lambda: str(exe))
        # No asset for the current platform.
        monkeypatch.setattr(
            self_update, "select_asset_for_current_platform", lambda assets: None
        )

        result = self_update.perform_self_update(_release())

        assert result.status is SelfUpdateStatus.FALLBACK_NOTIFY
