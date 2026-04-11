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
    assert "↔" not in src
    assert "('+ '" in src or "'+ '" in src
    assert "('- '" in src or "'- '" in src


def test_collapsible_section_bodies_are_height_driven() -> None:
    src = _mobile_main_source()
    assert "height: self.minimum_height if root.tasks_expanded else dp(0)" in src
    assert "height: self.minimum_height if root.timer_expanded else dp(0)" in src
    assert (
        "height: self.minimum_height if root.journal_expanded and root.act_controls_visible else dp(0)"
        in src
    )
    assert "height: dp(180) if root.tasks_expanded else dp(0)" in src
    assert "height: dp(44) if root.tasks_expanded else dp(0)" in src
    assert "height: dp(50) if root.timer_expanded else dp(0)" in src
    assert "height: dp(8) if root.timer_expanded else dp(0)" in src
    assert "height: dp(48) if root.timer_expanded else dp(0)" in src
    assert (
        "height: dp(40) if root.journal_expanded and root.act_controls_visible else dp(0)"
        in src
    )
    assert "disabled: not root.tasks_expanded" in src
    assert "disabled: not root.timer_expanded" in src
    assert "disabled: not (root.journal_expanded and root.act_controls_visible)" in src


def test_collapsed_section_children_are_removed_from_touch_flow() -> None:
    src = _mobile_main_source()
    assert "opacity: 1 if root.tasks_expanded else 0" in src
    assert "opacity: 1 if root.timer_expanded else 0" in src
    assert "opacity: 1 if root.act_controls_visible else 0" in src
    assert "disabled: not (root.journal_expanded and root.act_controls_visible)" in src


def test_encouragement_is_outside_collapsible_sections() -> None:
    src = _mobile_main_source()
    assert "text: root.nudge_text" in src
    assert "text: 'New encouragement'" not in src
    assert "def refresh_nudge(self):" not in src
    assert src.index("text: root.nudge_text") > src.index(
        "on_release: root.open_act_history()"
    )
    assert src.index("text: root.nudge_text") < src.index("Toolbar:")


def test_home_screen_exposes_accordion_toggle_handlers() -> None:
    src = _mobile_main_source()
    assert "def toggle_tasks_section(self) -> None:" in src
    assert "def toggle_timer_section(self) -> None:" in src
    assert "def toggle_journal_section(self) -> None:" in src
    assert 'self._toggle_section("tasks")' in src
    assert 'self._toggle_section("timer")' in src
    assert 'self._toggle_section("journal")' in src


def test_act_section_is_labeled_and_gated_by_profile_threshold() -> None:
    src = _mobile_main_source()
    assert "'ACT - ' + root.journal_summary" in src
    assert "height: dp(42) if root.act_controls_visible else dp(0)" in src
    assert "if not self.act_controls_visible:" in src
    assert "Acceptance and Commitment Therapy" in src
    assert 'title="ACT Momentum Reset"' in src


def test_update_check_and_timer_cycle_copy_are_mobile_safe() -> None:
    src = _mobile_main_source()
    assert "TODO: Implement actual update checking" not in src
    assert "Auto focus/break" in src
    assert 'text="ⓘ"' not in src
    assert 'text="Info"' in src
    assert 'text="Save reset"' in src


def test_stroop_uses_multiple_choice_buttons() -> None:
    src = _mobile_main_source()
    assert "Tap the INK COLOUR of the text, not the word." in src
    assert "id: stroop_input" not in src
    assert "on_release: root.answer_option(self.text)" in src
