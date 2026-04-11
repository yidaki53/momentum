from __future__ import annotations

from momentum.ui.update_check import (
    compare_versions,
    is_update_available,
    normalize_version,
)


def test_normalize_version_accepts_tag_prefix() -> None:
    assert normalize_version("v1.2.3") == "1.2.3"


def test_compare_versions_orders_semver_values() -> None:
    assert compare_versions("0.3.1", "0.4.0") == -1
    assert compare_versions("0.4.0", "0.3.1") == 1
    assert compare_versions("0.4.0", "0.4.0") == 0


def test_is_update_available_uses_semver_ordering() -> None:
    assert is_update_available("0.3.1", "0.4.0") is True
    assert is_update_available("0.4.0", "0.4.0") is False
