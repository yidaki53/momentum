"""Tests for the thermal guard that wraps long-running pipeline scripts.

The workstation crashed from thermal overload during a CPU fine-tuning run. These tests
exist so the guard that prevents a repeat is not itself untested.

Everything that decides *whether to stop* is a pure function over readings, so it is
testable on any machine -- including CI, which has no sensors at all.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "coach-distillation" / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location(
        "cd_guard", SCRIPTS / "thermal_guard.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["cd_guard"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def guard():
    return _load()


# --- thresholds -------------------------------------------------------------


def test_readings_are_graded_against_the_stop_threshold(guard):
    limits = {"cpu_package": (80.0, 90.0)}
    assert guard.classify("cpu_package", 40.0, limits) == "ok"
    assert guard.classify("cpu_package", 79.9, limits) == "ok"
    assert guard.classify("cpu_package", 80.0, limits) == "warn"
    assert guard.classify("cpu_package", 89.9, limits) == "warn"
    assert guard.classify("cpu_package", 90.0, limits) == "stop"
    assert guard.classify("cpu_package", 120.0, limits) == "stop"


def test_an_unrated_sensor_gets_a_coarse_backstop_not_a_free_pass(guard):
    """A sensor we have no rating for is still watched, just conservatively.

    A machine whose sensor layout we do not recognise must still be able to run the
    pipeline -- so an ordinary reading must not stop anything. But a chip at 150C is an
    emergency whatever its rating says, so the fallback has to catch that too.
    """
    assert guard.classify("mystery_chip", 45.0, {"cpu_package": (80.0, 90.0)}) == "ok"
    assert (
        guard.classify("mystery_chip", 105.0, {"cpu_package": (80.0, 90.0)}) == "warn"
    )
    assert (
        guard.classify("mystery_chip", 150.0, {"cpu_package": (80.0, 90.0)}) == "stop"
    )


def test_decide_stops_on_any_sensor_over_its_stop_threshold(guard):
    limits = {"cpu_package": (80.0, 90.0), "gpu": (80.0, 85.0)}
    action, offender = guard.decide(
        [guard.Reading("cpu_package", 70.0), guard.Reading("gpu", 86.0)], limits
    )
    assert action == "stop"
    assert offender is not None and offender.key == "gpu"


def test_decide_warns_without_stopping_on_a_warning_reading(guard):
    """A brief spike under load is normal; killing on it would make the guard useless."""
    limits = {"cpu_package": (80.0, 90.0)}
    action, offender = guard.decide([guard.Reading("cpu_package", 84.0)], limits)
    assert action == "warn"
    assert offender is not None


def test_decide_runs_when_everything_is_cool(guard):
    action, offender = guard.decide(
        [guard.Reading("cpu_package", 55.0), guard.Reading("gpu", 50.0)]
    )
    assert action == "run"
    assert offender is None


def test_empty_sample_is_not_a_stop(guard):
    """No readable sensors must not be mistaken for an overheat."""
    assert guard.decide([]) == ("run", None)


def test_every_tracked_sensor_has_a_threshold(guard):
    """A sensor we sample must have a limit, or sampling it is pointless."""
    for key in ("cpu_package", "cpu_core", "gpu", "nvme"):
        assert key in guard.DEFAULT_LIMITS
        warn, stop = guard.DEFAULT_LIMITS[key]
        assert warn < stop


# --- sensors parsing --------------------------------------------------------


def test_sensors_json_is_parsed_by_feature_name(guard):
    """The real `sensors -j` shape: features keyed by name, values as tempN_input."""
    payload = json.dumps(
        {
            "iwlwifi_1-virtual-0": {
                "Adapter": "Virtual device",
                "temp1": {"temp1_input": 54.0},
            },
            "coretemp-isa-0000": {
                "Adapter": "ISA adapter",
                "Package id 0": {"temp1_input": 62.0, "temp1_max": 100.0},
                "Core 0": {"temp2_input": 55.0},
                "Core 1": {"temp3_input": 81.0},
            },
            "nvme-pci-e100": {
                "Adapter": "PCI adapter",
                "Composite": {"temp1_input": 51.8, "temp1_max": 82.8},
            },
        }
    )
    readings = {r.key: r.celsius for r in guard.parse_sensors_json(payload)}

    assert readings["cpu_package"] == 62.0
    # The hottest core is what throttles the package, so that is what we record.
    assert readings["cpu_core"] == 81.0
    assert readings["nvme"] == 51.8
    # The wifi radio is not a temperature we guard against.
    assert "wifi" not in readings


def test_sensors_json_tolerates_leading_diagnostics(guard):
    """lm-sensors can print an ERROR line before the JSON on a partially readable board."""
    payload = (
        "ERROR: Can't get value of subfeature temp1_max_alarm: Can't read\n"
        + json.dumps({"coretemp-isa-0000": {"Package id 0": {"temp1_input": 70.0}}})
    )
    readings = guard.parse_sensors_json(payload)
    assert any(r.key == "cpu_package" and r.celsius == 70.0 for r in readings)


def test_unparseable_sensor_output_yields_nothing_rather_than_a_wrong_reading(guard):
    assert guard.parse_sensors_json("not json at all") == []


# --- supervision ------------------------------------------------------------


def test_supervise_returns_the_commands_own_exit_code(guard):
    """A guard that rewrote exit codes would break every caller that checks one."""
    assert (
        guard.supervise(
            [sys.executable, "-c", "raise SystemExit(7)"],
            interval=1,
            dry_run=False,
        )
        == 7
    )


def test_dry_run_starts_nothing(guard, capsys):
    assert guard.supervise(["echo", "should not run"], dry_run=True) == 0
    assert "should not run" in capsys.readouterr().out


def test_missing_lm_sensors_reports_clearly(guard, monkeypatch):
    """No sensors means no protection, and the guard must say so rather than imply safety."""
    monkeypatch.setattr(
        guard.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(guard, "read_gpu", lambda: [])
    assert guard.sample() == []

    monkeypatch.setattr(guard, "read_sensors", lambda: [])
    monkeypatch.setattr(guard, "read_gpu", lambda: [])
    assert guard.sample() == []


def test_the_guard_survives_killing_its_child(guard):
    """Regression: the child must be in its own process group.

    The first version spawned the child in the guard's own group, so the kill that ended
    the overheating run also ended the guard -- the protection was destroying the thing
    doing the protecting. This asserts the isolation directly.
    """
    import inspect

    source = inspect.getsource(guard.supervise)
    assert "start_new_session=True" in source, (
        "the child must be started in its own session, or _terminate's killpg takes "
        "the guard down with the child"
    )


def test_a_hot_run_is_stopped_and_reported_as_124(guard, monkeypatch):
    """The overheat path, end to end, with the thresholds forced down.

    Tightening the limits exercises the real kill path on hardware that is not actually
    overheating, which is the only way to test it in CI.
    """
    hot = [guard.Reading("cpu_package", 91.0) for _ in range(4)]
    monkeypatch.setattr(guard, "sample", lambda: hot)

    result = guard.supervise(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        interval=1,
        patience=2,
        limits={"cpu_package": (80.0, 90.0)},
    )
    assert result == 124


def test_a_cool_run_is_left_to_finish(guard, monkeypatch, tmp_path):
    marker = tmp_path / "finished"
    monkeypatch.setattr(
        guard,
        "sample",
        lambda: [guard.Reading("cpu_package", 55.0), guard.Reading("gpu", 50.0)],
    )
    result = guard.supervise(
        [
            sys.executable,
            "-c",
            f"import pathlib; pathlib.Path({str(marker)!r}).write_text('done')",
        ],
        interval=1,
        patience=2,
    )
    assert result == 0
    assert marker.exists()
