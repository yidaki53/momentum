"""Tests for mobile timer state synchronization and home section behavior."""

from __future__ import annotations

import pytest

pytest.importorskip("kivy")

from mobile.main import KV, HomeScreen, MomentumApp


class _FakeHome:
    def __init__(self) -> None:
        self.pause_called = 0
        self.stop_called = 0

    def toggle_timer_pause_resume(self) -> None:
        self.pause_called += 1

    def stop_timer(self) -> None:
        self.stop_called += 1


class _FakeRoot:
    def __init__(self, home: _FakeHome | None) -> None:
        self._home = home

    def has_screen(self, name: str) -> bool:
        return name == "home" and self._home is not None

    def get_screen(self, _name: str) -> _FakeHome:
        if self._home is None:
            raise KeyError("home")
        return self._home


def test_active_timer_state_updates_and_clears() -> None:
    app = MomentumApp()

    app.update_active_timer_state(
        mode="Focus",
        display="12:34",
        progress=42.0,
        paused=False,
    )

    assert app.timer_active is True
    assert app.active_timer_label == "Focus"
    assert app.active_timer_display == "12:34"
    assert app.active_timer_progress == 42.0
    assert app.active_timer_control_text == "Pause"

    app.update_active_timer_state(
        mode="Break",
        display="03:21",
        progress=75.5,
        paused=True,
    )
    assert app.active_timer_label == "Break (paused)"
    assert app.active_timer_control_text == "Resume"

    app.clear_active_timer_state()
    assert app.timer_active is False
    assert app.active_timer_display == "00:00"
    assert app.active_timer_progress == 0


def test_banner_controls_delegate_to_home_screen() -> None:
    app = MomentumApp()
    home = _FakeHome()
    app.root = _FakeRoot(home)

    app.toggle_timer_pause_resume()
    app.stop_active_timer()

    assert home.pause_called == 1
    assert home.stop_called == 1


def test_home_sections_use_accordion_behavior() -> None:
    home = HomeScreen()
    assert home.tasks_expanded is True
    assert home.timer_expanded is False
    assert home.journal_expanded is False

    home.toggle_timer_section()
    assert home.tasks_expanded is False
    assert home.timer_expanded is True
    assert home.journal_expanded is False

    home.toggle_journal_section()
    assert home.tasks_expanded is False
    assert home.timer_expanded is False
    assert home.journal_expanded is True

    # Toggling the open section closes it, leaving all collapsed.
    home.toggle_journal_section()
    assert home.tasks_expanded is False
    assert home.timer_expanded is False
    assert home.journal_expanded is False


def test_start_timer_expands_timer_section() -> None:
    home = HomeScreen()
    home._start_timer(minutes=1, is_break=False, task_id=None)
    try:
        assert home.tasks_expanded is False
        assert home.timer_expanded is True
        assert home.journal_expanded is False
        assert home.timer_display == "01:00"
    finally:
        home.stop_timer()


def test_home_section_markers_stay_ascii() -> None:
    # Keep section markers font-safe across platforms.
    assert "▼" not in KV
    assert "▶" not in KV
    assert "'+ '" in KV
    assert "'- '" in KV
