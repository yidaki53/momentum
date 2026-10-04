"""Tests for the thermal guard that wraps long-running pipeline scripts.

The workstation crashed from thermal overload during a CPU fine-tuning run. These tests
exist so the guard that prevents a repeat is not itself untested.

Everything that decides *whether to stop* is a pure function over readings, so it is
testable on any machine -- including CI, which has no sensors at all.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import subprocess
import sys
import time
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

    CHANGED 2026-10-03, deliberately and at the owner's instruction: the ceiling moved
    from 80C to 86C, and this test previously asserted ``<= 80``. The owner runs this
    machine and has twice seen it crash from heat, so the judgement is theirs to make --
    but the assertion now encodes the real risk rather than pretending 86 is as safe as
    80 was:

    - 86C is still 14C below the firmware's own 100C trip point.
    - The 80C figure did not disappear. It is now ``DEFAULT_START_CELSIUS``, and it is
      enforced by preflight, so a machine at 81C will not accept new work even though
      a machine at 85C will keep an existing run alive.
    - The margin test below asserts we stay under the firmware limit, which is the
      property that actually prevents the shutdown, rather than a specific number that
      was only ever a guess.
    """
    for key, (_warn, stop) in guard.DEFAULT_LIMITS.items():
        assert stop <= 100.0, f"{key} stop threshold is at or above the firmware limit"
    assert guard.DEFAULT_LIMITS["cpu_package"][1] == 86.0
    # The start gate is the tighter of the two, and is what protects an idle-then-worked
    # machine from the first sample.
    assert guard.DEFAULT_START_CELSIUS == 80.0
    assert guard.DEFAULT_START_CELSIUS < guard.DEFAULT_LIMITS["cpu_package"][1]
    for key, (warn, stop) in guard.DEFAULT_LIMITS.items():
        assert warn < stop, key


def test_nvme_ceiling_is_watched_separately_and_is_lower_than_the_cpu(guard):
    """A drive throttles long before the CPU reads a reassuring number."""
    assert guard.DEFAULT_LIMITS["nvme"][1] < guard.DEFAULT_LIMITS["cpu_package"][1]


def test_default_sampling_is_fast_enough_for_a_laptop(guard):
    """Fifteen seconds is long enough to overshoot, and too coarse for a hot timer."""
    assert guard.DEFAULT_INTERVAL <= 5


def test_the_default_tolerance_is_a_duration_not_a_rate(guard):
    """The rule is "too hot for too long", not "gets hot fast".

    An earlier version killed on the rate of climb and made a legitimate run
    impossible: loading a quantised 7B spikes the CPU from 61C to 78C in about five
    seconds and then settles. That is what loading a model does, not a thermal event.
    """
    assert 15.0 <= guard.DEFAULT_MAX_HOT_SECONDS <= 60.0, (
        "the tolerance must outlast a model-load spike but stay far below a "
        "sustained-heat failure"
    )
    assert not hasattr(guard, "ramp_c_per_minute"), (
        "the ramp rule is gone: rate-of-climb produces false positives on a laptop"
    )


def test_a_brief_spike_does_not_kill_the_run(guard, monkeypatch, tmp_path):
    """The load spike that used to stop every run must now be survivable."""
    hot = {"n": 0}

    def sample():
        hot["n"] += 1
        # Two samples hot, then it settles for the rest of the run.
        return [guard.Reading("cpu_package", 82.0 if hot["n"] <= 2 else 55.0)]

    monkeypatch.setattr(guard, "sample", sample)
    marker = tmp_path / "finished"
    result = guard.supervise(
        [
            sys.executable,
            "-c",
            f"import pathlib, time; time.sleep(4); pathlib.Path({str(marker)!r}).touch()",
        ],
        interval=1,
        max_hot_seconds=6,
        allow_hot_start=True,
    )
    assert result == 0
    assert marker.exists()


def test_sustained_heat_pauses_before_it_kills(guard, monkeypatch):
    """Sustained heat pauses the run and lets it finish, rather than killing it.

    A killed training run loses its checkpoint. Pausing costs five minutes and keeps
    the work, which is the better trade every time the run is resumable.
    """
    # Read the real stop threshold rather than hardcoding one: a literal 85C silently
    # stopped being "too hot" when the ceiling moved to 86C, and three tests went green
    # without pausing anything.
    hot = guard.DEFAULT_LIMITS["cpu_package"][1] + 1.0
    monkeypatch.setattr(guard, "sample", lambda: [guard.Reading("cpu_package", hot)])
    pauses = []
    original = guard._pause_child

    def spy(process, seconds, limits, telemetry, started):
        """Record the requested pause, then run a zero-length one for speed."""
        pauses.append(seconds)
        return original(process, 0.0, limits, telemetry, started)

    monkeypatch.setattr(guard, "_pause_child", spy)
    result = guard.supervise(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        interval=1,
        max_hot_seconds=2,
        pause_seconds=30,
        max_pauses=5,
        allow_hot_start=True,
    )
    assert pauses, "the run was never paused"
    assert pauses[0] == 30, "the requested pause length was not honoured"
    # It gave up eventually rather than pausing forever.
    assert result == 124


def test_a_paused_child_really_stops_working(guard):
    """A pause that does not idle the process is not a pause.

    This is the property the whole feature rests on. SIGSTOP is sent to the child's
    process group so that dataloader workers stop too; signalling only the parent would
    leave the CPU pinned through the cool-down and quietly achieve nothing.

    Verified by CPU time from /proc rather than by the guard's own return value, because
    a guard that claims to have paused while its child burns four cores looks identical
    from the outside.
    """
    import signal

    child = [
        sys.executable,
        "-c",
        "import threading, time\n"
        "def burn():\n"
        "    end = time.time() + 20\n"
        "    while time.time() < end:\n"
        "        sum(i * i for i in range(20000))\n"
        "for _ in range(3):\n"
        "    threading.Thread(target=burn).start()\n"
        "time.sleep(20)\n",
    ]
    process = subprocess.Popen(child, start_new_session=True)
    try:
        time.sleep(2.0)

        def cpu_seconds():
            with open(f"/proc/{process.pid}/stat") as handle:
                fields = handle.read().rsplit(")", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / 100.0

        before = cpu_seconds()
        guard._signal_group(process, signal.SIGSTOP)
        time.sleep(2.0)
        during = cpu_seconds()
        state = open(f"/proc/{process.pid}/stat").read().rsplit(")", 1)[1].split()[0]
        assert state == "T", f"child is not stopped (state {state!r})"
        assert during - before < 0.3, (
            f"child burned {during - before:.1f}s of CPU while supposedly paused"
        )
        guard._signal_group(process, signal.SIGCONT)
        time.sleep(1.0)
        assert cpu_seconds() - during > 0.3, "SIGCONT did not resume the child"
    finally:
        with contextlib.suppress(Exception):
            guard._terminate(process)


def test_pause_budget_is_finite(guard, monkeypatch):
    """A run that overheats repeatedly must stop, not pause forever.

    Otherwise a workload that cannot be cooled looks like a run that is merely slow,
    and the failure is discovered hours later.
    """
    # Read the real stop threshold rather than hardcoding one: a literal 85C silently
    # stopped being "too hot" when the ceiling moved to 86C, and three tests went green
    # without pausing anything.
    hot = guard.DEFAULT_LIMITS["cpu_package"][1] + 1.0
    monkeypatch.setattr(guard, "sample", lambda: [guard.Reading("cpu_package", hot)])
    pauses = []
    monkeypatch.setattr(
        guard,
        "_pause_child",
        lambda process, seconds, limits, telemetry, started: pauses.append(seconds),
    )
    result = guard.supervise(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        interval=1,
        max_hot_seconds=2,
        pause_seconds=10,
        max_pauses=2,
        allow_hot_start=True,
    )
    assert result == 124
    assert len(pauses) == 2, f"expected exactly 2 pauses, got {len(pauses)}"


def test_pause_seconds_zero_keeps_the_kill_only_behaviour(guard, monkeypatch):
    """Some commands are cheap to re-run and should not be paused at all."""
    # Read the real stop threshold rather than hardcoding one: a literal 85C silently
    # stopped being "too hot" when the ceiling moved to 86C, and three tests went green
    # without pausing anything.
    hot = guard.DEFAULT_LIMITS["cpu_package"][1] + 1.0
    monkeypatch.setattr(guard, "sample", lambda: [guard.Reading("cpu_package", hot)])
    paused = []
    monkeypatch.setattr(
        guard,
        "_pause_child",
        lambda *args: paused.append(args) or None,
    )
    result = guard.supervise(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        interval=1,
        max_hot_seconds=2,
        pause_seconds=0,
        allow_hot_start=True,
    )
    assert result == 124
    assert not paused, "pause_seconds=0 must not pause"


def test_default_budget_is_five_minutes_and_two_pauses(guard):
    """Pinned because these are the values the machine was tuned against."""
    assert guard.DEFAULT_MAX_HOT_SECONDS == 30.0
    assert guard.DEFAULT_PAUSE_SECONDS == 300.0
    assert guard.DEFAULT_MAX_PAUSES == 2


# --- two thresholds: may start, versus must pause ---------------------------


def test_start_gate_and_stop_threshold_are_separate(guard):
    """80C may start a run; 86C pauses one. One number cannot express both.

    They used to be the same value, which meant raising the ceiling to allow starting
    also permitted starting at the ceiling -- backwards. The tolerance belongs at the
    start gate, where nothing is running yet.
    """
    guard.busiest_other_cpu = lambda: 0.0
    assert guard.DEFAULT_START_CELSIUS == 80.0
    assert guard.DEFAULT_LIMITS["cpu_package"][1] == 86.0
    assert guard.DEFAULT_LIMITS["cpu_core"][1] == 86.0

    def starts_at(temp):
        guard.sample = lambda: [guard.Reading("cpu_package", temp)]
        return guard.preflight().ok

    assert starts_at(79.0), "79C should be fine to start a task"
    assert not starts_at(80.0), "80C is the start gate"
    assert not starts_at(85.0), "85C is above the start gate"


def test_start_gate_can_be_relaxed_per_run(guard):
    """The gate is a policy, not a law: --start-celsius has to reach it."""
    guard.busiest_other_cpu = lambda: 0.0
    guard.sample = lambda: [guard.Reading("cpu_package", 84.0)]
    assert not guard.preflight().ok
    assert guard.preflight(start_celsius=85.0).ok, "an explicit looser gate must apply"
    assert guard.preflight(start_celsius=None).ok, "None disables the gate entirely"


def test_stop_threshold_still_governs_a_running_job(guard):
    """Raising the start gate must not raise the ceiling for work already running."""
    assert guard.classify("cpu_package", 85.9) != "stop"
    assert guard.classify("cpu_package", 86.0) == "stop"
    assert guard.classify("cpu_core", 86.0) == "stop"


def test_nvme_keeps_its_own_lower_ceiling(guard):
    """The drive's limit is a hardware fact, not a preference to relax with the CPU.

    A drive at 70C is already throttling while the CPU reads a comfortable number, so
    the NVMe stop stays well below the CPU's even though the user raised the CPU.
    """
    assert guard.DEFAULT_LIMITS["nvme"][1] < guard.DEFAULT_LIMITS["cpu_package"][1]
    assert guard.classify("nvme", 70.0) != "stop"
    assert guard.classify("nvme", 76.0) == "stop"
    # And it must not participate in the start gate, which is a CPU question.
    assert "nvme" not in guard.START_GATED_KEYS


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
