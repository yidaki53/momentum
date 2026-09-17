import pytest

from momentum.ui import update_check
from momentum.ui.update_check import fetch_latest_release, is_update_available


@pytest.mark.parametrize(
    ("current", "latest", "expected"),
    [
        ("0.4.0-build.69", "v0.4.0-build.70", True),
        ("0.4.0-build.70", "0.4.0-build.70", False),
        ("0.4.0-build.71", "0.4.0-build.70", False),
        ("0.4.0-build.9", "0.4.0-build.10", True),
        ("0.4.0", "0.4.0-build.70", True),
        ("0.5.0", "0.4.0-build.999", False),
    ],
)
def test_ci_build_ordering(current, latest, expected) -> None:
    assert is_update_available(current, latest) is expected


def test_release_keeps_build_number(monkeypatch) -> None:
    monkeypatch.setattr(
        update_check,
        "_read_release_payload",
        lambda *args, **kwargs: {"tag_name": "v0.4.0-build.70"},
    )
    assert fetch_latest_release().version == "0.4.0-build.70"
