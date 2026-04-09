"""Non-Kivy guardrail tests for mobile home UI behavior contracts."""

from __future__ import annotations

from pathlib import Path


def _mobile_main_source() -> str:
    root = Path(__file__).resolve().parent.parent
    return (root / "mobile" / "main.py").read_text(encoding="utf-8")


def test_section_markers_are_ascii_only() -> None:
    src = _mobile_main_source()
    assert "▼" not in src
    assert "▶" not in src
    assert "('+ '" in src or "'+ '" in src
    assert "('- '" in src or "'- '" in src


def test_collapsible_section_bodies_are_height_driven() -> None:
    src = _mobile_main_source()
    assert "height: self.minimum_height if root.tasks_expanded else dp(0)" in src
    assert "height: self.minimum_height if root.timer_expanded else dp(0)" in src
    assert "height: self.minimum_height if root.journal_expanded else dp(0)" in src
    assert "disabled: not root.tasks_expanded" in src
    assert "disabled: not root.timer_expanded" in src
    assert "disabled: not root.journal_expanded" in src


def test_home_screen_exposes_accordion_toggle_handlers() -> None:
    src = _mobile_main_source()
    assert "def toggle_tasks_section(self) -> None:" in src
    assert "def toggle_timer_section(self) -> None:" in src
    assert "def toggle_journal_section(self) -> None:" in src
    assert 'self._toggle_section("tasks")' in src
    assert 'self._toggle_section("timer")' in src
    assert 'self._toggle_section("journal")' in src
