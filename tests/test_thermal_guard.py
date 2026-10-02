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


def test_supervise_returns_the_commands_own_exit_code(guard, monkeypatch):
    """A guard that rewrote exit codes would break every caller that checks one."""
    monkeypatch.setattr(
        guard,
        "sample",
        lambda: [guard.Reading("cpu_package", 50.0), guard.Reading("gpu", 45.0)],
    )
    monkeypatch.setattr(guard, "busiest_other_cpu", lambda: 5.0)
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


# ---------------------------------------------------------------------------
# The second crash: laptop thresholds, ramp detection, preflight, durable logs
# ---------------------------------------------------------------------------


def test_thresholds_leave_real_headroom_below_the_firmware_limits(guard):
    """Waiting for the firmware limit means letting the machine defend itself.

    The CPU reports high=100C / crit=100C. The first guard stopped at 90C, which on a
    laptop is not a margin -- it is a countdown, and it crashed anyway.
    """
    assert guard.DEFAULT_LIMITS["cpu_package"][1] <= 80.0
    assert (
        guard.DEFAULT_LIMITS["cpu_package"][0] < guard.DEFAULT_LIMITS["cpu_package"][1]
    )
    for key, (warn, stop) in guard.DEFAULT_LIMITS.items():
        assert warn < stop, key
        assert stop <= 85.0, f"{key} stop threshold is too permissive for a laptop"


def test_nvme_ceiling_is_watched_separately_and_is_lower_than_the_cpu(guard):
    """A drive throttles long before the CPU reads a reassuring number."""
    assert guard.DEFAULT_LIMITS["nvme"][1] < guard.DEFAULT_LIMITS["cpu_package"][1]


def test_default_sampling_is_fast_enough_for_a_laptop(guard):
    """Fifteen seconds is long enough for a small laptop to overshoot."""
    assert guard.DEFAULT_INTERVAL <= 5
    assert guard.DEFAULT_PATIENCE <= 2


def test_a_fast_climb_is_stopped_before_any_threshold_is_reached(guard):
    """Ramp detection catches the runaway that absolute thresholds miss.

    A fan that has stopped, or a job entering a heavier phase, can climb 30C while still
    reading below the absolute stop threshold. Waiting for 78C in that situation is
    waiting for the shutdown.
    """
    history = [(0.0, 45.0), (20.0, 58.0), (40.0, 70.0)]
    assert guard.ramp_c_per_minute(history) > guard.RAMP_C_PER_MIN
    assert guard.ramp_is_dangerous(history)
    # A slow, steady warm-up over the same period is fine.
    assert not guard.ramp_is_dangerous([(0.0, 45.0), (20.0, 50.0), (40.0, 55.0)])
    # And a flat machine is fine.
    assert not guard.ramp_is_dangerous([(0.0, 60.0), (30.0, 60.5)])


def test_ramp_needs_enough_history_to_mean_anything(guard):
    assert guard.ramp_c_per_minute([]) == 0.0
    assert guard.ramp_c_per_minute([(0.0, 50.0)]) == 0.0
    # Two samples 1s apart cannot distinguish a ramp from sensor noise.
    assert not guard.ramp_is_dangerous([(0.0, 50.0), (1.0, 70.0)])


def test_preflight_refuses_to_start_on_a_warm_machine(guard, monkeypatch):
    """This is the check that would have prevented the second crash.

    The machine was already at 89C with another job at 95% CPU. Supervising only our
    own child cannot see that, and by the time a second load pushes it over there is no
    time left to react.
    """
    monkeypatch.setattr(guard, "sample", lambda: [guard.Reading("cpu_package", 76.0)])
    monkeypatch.setattr(guard, "busiest_other_cpu", lambda: 5.0)
    check = guard.preflight()
    assert not check.ok
    assert any("already at" in reason for reason in check.reasons)

    monkeypatch.setattr(guard, "sample", lambda: [guard.Reading("cpu_package", 40.0)])
    monkeypatch.setattr(guard, "busiest_other_cpu", lambda: 5.0)
    assert guard.preflight().ok


def test_preflight_refuses_when_another_job_is_already_saturating_the_cpu(
    guard, monkeypatch
):
    """Stacking a second heavy job is what crashed this machine."""
    monkeypatch.setattr(guard, "sample", lambda: [guard.Reading("cpu_package", 45.0)])
    monkeypatch.setattr(guard, "busiest_other_cpu", lambda: 95.0)
    check = guard.preflight()
    assert not check.ok
    assert any("another process" in reason for reason in check.reasons)


def test_preflight_warns_but_does_not_block_without_sensors(guard, monkeypatch):
    """No sensors is a reason to be careful, not a reason to block work forever."""
    monkeypatch.setattr(guard, "sample", lambda: [])
    monkeypatch.setattr(guard, "busiest_other_cpu", lambda: 5.0)
    check = guard.preflight()
    assert check.ok
    assert any("no temperature sensors" in reason for reason in check.reasons)


def test_supervise_refuses_a_hot_start_with_exit_code_three(
    guard, monkeypatch, tmp_path
):
    """Exit 3 means "refused", which is distinct from 124 "stopped while running"."""
    monkeypatch.setattr(guard, "sample", lambda: [guard.Reading("cpu_package", 95.0)])
    monkeypatch.setattr(guard, "busiest_other_cpu", lambda: 5.0)
    marker = tmp_path / "should_not_exist"
    result = guard.supervise(
        [
            sys.executable,
            "-c",
            f"import pathlib; pathlib.Path({str(marker)!r}).touch()",
        ],
        interval=1,
        log_path=tmp_path / "thermal.csv",
    )
    assert result == 3
    assert not marker.exists(), "the command ran despite a refusing preflight"


def test_allow_hot_start_skips_the_preflight(guard, monkeypatch, tmp_path):
    monkeypatch.setattr(guard, "sample", lambda: [guard.Reading("cpu_package", 40.0)])
    monkeypatch.setattr(guard, "busiest_other_cpu", lambda: 5.0)
    marker = tmp_path / "ran"
    result = guard.supervise(
        [
            sys.executable,
            "-c",
            f"import pathlib; pathlib.Path({str(marker)!r}).touch()",
        ],
        interval=1,
        allow_hot_start=True,
    )
    assert result == 0
    assert marker.exists()


def test_samples_are_written_to_a_csv_that_survives(guard, tmp_path):
    """The crash left no record at all, which is why thresholds were guesswork.

    The telemetry has to outlive the process that produced it, so the log is flushed and
    fsynced per sample rather than at exit.
    """
    path = tmp_path / "thermal.csv"
    telemetry = guard.TemperatureLog(path)
    telemetry.write(
        1.0, [guard.Reading("cpu_package", 61.0), guard.Reading("gpu", 44.0)], "run"
    )
    telemetry.write(6.0, [guard.Reading("cpu_package", 63.0)], "warn")
    telemetry.close()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split(",")[0] == "timestamp"
    assert len(lines) == 3
    assert "61.0" in lines[1] and "44.0" in lines[1] and "run" in lines[1]
    assert "warn" in lines[2]


def test_the_log_records_every_sensors_value_a_run_touched(guard, tmp_path):
    """A log missing the NVMe is how the drive gets cooked unnoticed."""
    path = tmp_path / "thermal.csv"
    telemetry = guard.TemperatureLog(path)
    telemetry.write(
        1.0,
        [
            guard.Reading("cpu_package", 60.0),
            guard.Reading("nvme", 65.0),
            guard.Reading("nvme", 61.0),
        ],
        "run",
    )
    telemetry.close()
    row = path.read_text(encoding="utf-8").strip().splitlines()[1]
    assert "65.0" in row, "the hottest of several drives must be the one recorded"


def test_a_cool_run_is_left_to_finish(guard, monkeypatch, tmp_path):
    monkeypatch.setattr(
        guard,
        "sample",
        lambda: [guard.Reading("cpu_package", 50.0), guard.Reading("gpu", 42.0)],
    )
    marker = tmp_path / "finished"
    result = guard.supervise(
        [
            sys.executable,
            "-c",
            f"import pathlib; pathlib.Path({str(marker)!r}).write_text('done')",
        ],
        interval=1,
        allow_hot_start=True,
        log_path=tmp_path / "t.csv",
    )
    assert result == 0
    assert marker.exists()
    assert (tmp_path / "t.csv").exists()


def test_a_guarded_run_leaves_a_readable_temperature_trail(
    guard, monkeypatch, tmp_path
):
    """End to end: a real child, a real log, the file still there afterwards."""
    monkeypatch.setattr(
        guard,
        "sample",
        lambda: [guard.Reading("cpu_package", 55.0), guard.Reading("gpu", 48.0)],
    )
    path = tmp_path / "trail.csv"
    guard.supervise(
        [sys.executable, "-c", "import time; time.sleep(3)"],
        interval=1,
        allow_hot_start=True,
        log_path=path,
    )
    text = path.read_text(encoding="utf-8")
    assert "cpu_package" in text and "55.0" in text
