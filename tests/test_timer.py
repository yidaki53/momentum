"""Tests for the timer module."""

from __future__ import annotations

from unittest.mock import patch

from momentum.models import TimerConfig
from momentum.timer import run_break, run_focus, run_timer


class TestTimerConfig:
    def test_focus_config(self) -> None:
        config = TimerConfig(minutes=15, label="Focus")
        assert config.minutes == 15
        assert not config.is_break

    def test_break_config(self) -> None:
        config = TimerConfig(minutes=5, label="Break", is_break=True)
        assert config.is_break


class TestRunTimer:
    @patch("momentum.timer.time.sleep")
    def test_completes(self, mock_sleep) -> None:
        """Timer with 1 minute should call sleep 60 times and return True."""
        config = TimerConfig(minutes=1, label="Test")
        result = run_timer(config)
        assert result is True
        assert mock_sleep.call_count == 60

    @patch("momentum.timer.time.sleep", side_effect=KeyboardInterrupt)
    def test_interrupted(self, mock_sleep) -> None:
        """Ctrl-C should return False."""
        config = TimerConfig(minutes=1, label="Test")
        result = run_timer(config)
        assert result is False


class TestConvenienceFunctions:
    @patch("momentum.timer.time.sleep")
    def test_run_focus(self, mock_sleep) -> None:
        result = run_focus(minutes=1)
        assert result is True

    @patch("momentum.timer.time.sleep")
    def test_run_break(self, mock_sleep) -> None:
        result = run_break(minutes=1)
        assert result is True
