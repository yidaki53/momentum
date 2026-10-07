#!/usr/bin/env python3
"""Run a command, but stop it if the machine gets too hot.

This machine has now crashed from thermal overload TWICE. The second time, with the
first version of this guard already written, we still could not say what had happened --
because the guard's output went to a terminal that died with the machine. That lesson
shaped every design decision below.

What is different from the first version:

1. **Durable telemetry.** Samples are appended to a CSV with flush+fsync, so the
   temperature history survives the crash. After two crashes we had no record at all;
   now the run that heats the machine leaves a trail either way.
2. **Laptop thresholds, not desktop ones.** The CPU stop threshold moved from 90C to
   78C. Sustained 89C on a laptop is not "hot", it is a countdown.
3. **A duration rule, not a rate rule.** An earlier version killed on the *rate* of
   climb. That was wrong: loading a quantised 7B spikes the CPU from 61C to 78C in about
   five seconds and then settles, and the guard made a legitimate run impossible. What
   kills a laptop is sustained heat, so a short spike is tolerated and only time-above-
   threshold decides.
4. **A preflight gate.** The guard now REFUSES to start when the machine is already warm
   or already busy. The crash happened with two heavy jobs sharing the box; stacking a
   second one is exactly the failure mode, and a guard that only supervises its own child
   cannot see it.
5. **Five-second sampling.** Fifteen seconds is long enough for a laptop to overshoot,
   and it sets the resolution of the hot-duration timer.
6. **System-load awareness.** Other processes' CPU use is read and reported, because the
   heat is the machine's, not the guarded job's.

Every decision is a pure function over readings, so all of this is testable on CI
hardware with no sensors at all.

Usage::

    python scripts/thermal_guard.py --check
    python scripts/thermal_guard.py --log out/thermal.csv -- python scripts/train.py ...
    python scripts/thermal_guard.py --max-hot-seconds 60 -- ...   # tolerate more
    python scripts/thermal_guard.py --daemon --log out/thermal.csv   # detached logger

Exit codes: the command's own status, 124 if stopped for heat, 3 if preflight refused.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger("thermal_guard")

# --- thresholds --------------------------------------------------------------
# Laptop-appropriate, with deliberate margin below the firmware's own limits.
#
# The firmware reports high=100C / crit=100C for the CPU package and ~89C max for the
# GPU. Those are the temperatures at which the hardware defends ITSELF -- by throttling,
# or by shutting down. Waiting for them means letting the machine decide. Everything
# here stops well before that.
#
# The NVMe ceiling is much lower than the CPU's and is the one most easily missed: a
# drive at 70C is throttling even while the CPU reads a reassuring number.
DEFAULT_LIMITS: dict[str, tuple[float, float]] = {
    "cpu_package": (70.0, 86.0),
    "cpu_core": (75.0, 86.0),
    "gpu": (72.0, 86.0),
    # The NVMe ceiling stays well below the CPU's. This is a hardware fact rather than
    # a preference: a drive at 70C is already throttling even while the CPU reads a
    # reassuring number, and throttling a drive under sustained write load is its own
    # failure mode. 76 is above the throttle point but far below the CPU's ceiling.
    "nvme": (60.0, 76.0),
}

# Fallback for a sensor we sample but have no rating for. Coarse but not a free pass.
UNRATED_LIMITS: tuple[float, float] = (100.0, 120.0)

# The bar a run must clear BEFORE it starts. Deliberately lower than the stop threshold.
#
# These are two different questions and the guard now answers them separately:
#   "may I start?"  -- answered by this, at 80C. A machine at 79C is fine to work on.
#   "must I stop?"  -- answered by DEFAULT_LIMITS above, at 86C. Nothing runs past that.
#
# Before this existed there was only one number doing both jobs, so raising it to 86 to
# allow starting would ALSO have permitted starting at 85C. That is the wrong way round:
# the tolerance belongs at the start gate, where it is cheap to be generous because
# nothing is running yet, and not at the ceiling, where it is a crash.
DEFAULT_START_CELSIUS = 80.0

# Sensors the start gate applies to. NVMe is absent deliberately: it has its own, much
# lower ceiling in DEFAULT_LIMITS, and gating a start on it would refuse work for a
# reason the CPU limits do not explain.
START_GATED_KEYS: frozenset[str] = frozenset({"cpu_package", "cpu_core", "gpu"})

# How long the machine may sit ABOVE its stop threshold before the run is killed.
#
# This is a duration rule, not a rate rule, and the distinction is deliberate. An
# earlier version killed on the *rate* of climb, which turned out to be wrong: loading
# a quantised 7B spikes the CPU from 61C to 78C in about five seconds and then settles.
# That is what loading a model does. It is a brief excursion, not a thermal event, and
# killing it meant a legitimate run could never start.
#
# What actually kills a laptop is sustained heat. So: a short spike is tolerated, and
# only time-above-threshold decides. Thirty seconds is long enough for a load spike and
# far too short for anything that is genuinely cooking.
DEFAULT_MAX_HOT_SECONDS = 30.0

# Sampling cadence. Five seconds gives a resolution of 5s on the hot-duration timer.
DEFAULT_INTERVAL = 5

# When the run exceeds --max-hot-seconds we do not kill it outright: we SIGSTOP the
# child's process group, wait out the cool-down, then SIGCONT and carry on. Killing a
# half-finished training run loses the checkpoint; pausing keeps it.
#
# Stopping (rather than continuing) matters because a paused process holds its GPU
# memory but does no work, so the machine actually gets to cool. The default is 300s,
# which is long enough to drop several degrees on this chassis.
DEFAULT_PAUSE_SECONDS = 60.0

# How many times one run may pause before we conclude that pausing is not enough and
# the workload has to change.
#
# The OWNER'S RULE is "pause for a minute, let it cool, then carry on" -- an unlimited
# budget, not a retry limit. So the default is 0, meaning "pause as often as it takes".
# Pausing is a genuinely safe response: the child is SIGSTOPped, does no work, holds no
# extra load, and resumes when the machine is cool. Nothing is lost by pausing often.
#
# Killing is what loses work, so the kill now requires an explicit budget. A previous
# default of 2 pauses killed v8 training twice -- at epoch 2 of 3, with the loss curve
# still falling -- purely because the workload is heavy for this chassis, not because
# pausing had stopped working. Each pause did exactly what it was supposed to: the
# temperature came back down every time.
#
# Set --max-pauses N to keep the old fail-fast behaviour: useful for a long unattended
# run where pausing forever would hide a workload that genuinely needs reducing.
DEFAULT_MAX_PAUSES = 0

# Refuse to start if some *other* process is already using this much of the CPU.
CONCURRENT_CPU_PERCENT = 60.0

CSV_HEADER = (
    "timestamp",
    "elapsed_s",
    "cpu_package",
    "cpu_core",
    "gpu",
    "nvme_max",
    "load1",
    "decision",
)


@dataclass(frozen=True)
class Reading:
    """One sensor sample."""

    key: str
    celsius: float


# --------------------------------------------------------------------------
# Decisions -- pure functions over readings
# --------------------------------------------------------------------------


def classify(
    key: str, celsius: float, limits: Optional[dict[str, tuple[float, float]]] = None
) -> str:
    """Return ``"ok"``, ``"warn"`` or ``"stop"`` for one reading."""
    table = limits if limits is not None else DEFAULT_LIMITS
    warn, stop = table.get(key, UNRATED_LIMITS)
    if celsius >= stop:
        return "stop"
    if celsius >= warn:
        return "warn"
    return "ok"


def hottest(readings: Iterable[Reading]) -> Optional[Reading]:
    return max(readings, key=lambda r: r.celsius, default=None)


def decide(
    readings: Iterable[Reading], limits: Optional[dict[str, tuple[float, float]]] = None
) -> tuple[str, Optional[Reading]]:
    """Fold a sample into ``("run"|"warn"|"stop", offender)``.

    Any single stop-level reading stops the run. A warn never does: a brief excursion is
    normal under load, and a guard that reacts to every spike trains you to ignore it.
    """
    worst: Optional[Reading] = None
    state = "run"
    for reading in readings:
        reading_state = classify(reading.key, reading.celsius, limits)
        if reading_state == "stop":
            return ("stop", reading)
        if reading_state == "warn" and worst is None:
            state, worst = "warn", reading
    return (state, worst)


# --------------------------------------------------------------------------
# Preflight -- refuse to start rather than regret starting
# --------------------------------------------------------------------------


@dataclass
class Preflight:
    """Why a run may not start, or that it may.

    ``reasons`` mixes blocking and advisory findings, which is a trap: an earlier
    version collected a busy-CPU reason and still returned ``ok=True``, so the single
    most dangerous case -- the machine already saturated -- was the one that did not
    block. Blocking status is computed explicitly from ``blocking``.
    """

    ok: bool
    reasons: list[str]
    readings: list["Reading"] = field(default_factory=list)
    blocking: bool = False


def preflight(
    limits: Optional[dict[str, tuple[float, float]]] = None,
    start_celsius: Optional[float] = DEFAULT_START_CELSIUS,
) -> Preflight:
    """Decide whether it is safe to start a heavy run right now.

    This is the check that would have prevented the second crash. The machine was
    already at 89C with another job at 95% CPU; a guard that only supervises its own
    child cannot see that, and by the time temperatures climb under a second load there
    may be no time left to react.

    ``start_celsius`` is the gate for *beginning* work, defaulting to
    ``DEFAULT_START_CELSIUS`` (80). Two thresholds, not one: a machine at 79C may start
    a run, a run at 86C pauses, and nothing runs at 87C.
    """
    reasons: list[str] = []
    blocking = False
    readings = sample()

    if not readings:
        # Not a refusal: an unreadable sensor is a reason to be careful, not a reason to
        # block work forever. But it is worth saying out loud.
        reasons.append("no temperature sensors readable; running unprotected")

    for reading in readings:
        # The start gate, checked before classify() because it is the looser of the two
        # thresholds and therefore decides more often. A machine at 81C is not in danger,
        # but it is not somewhere to add a new 7B teacher either: "wait a little", not
        # "danger". NVMe is excluded by START_GATED_KEYS -- it has its own lower ceiling.
        if (
            start_celsius is not None
            and reading.key in START_GATED_KEYS
            and reading.celsius >= start_celsius
        ):
            blocking = True
            reasons.append(
                f"{reading.key} already at {reading.celsius:.1f}C, at or above the "
                f"{start_celsius:.0f}C start threshold -- let it cool a little first"
            )
        state = classify(reading.key, reading.celsius, limits)
        if state == "stop":
            blocking = True
            reasons.append(
                f"{reading.key} already at {reading.celsius:.1f}C, over its stop "
                f"threshold -- let it cool first"
            )
        elif state == "warn":
            # Advisory only. This machine idles near 55C and touches the low 70s just
            # from editor activity, so a warn-level reading is normal rather than
            # alarming. Blocking here made the guard refuse to start most of the time,
            # which trains people to reach for --allow-hot-start -- and a guard people
            # disable protects nothing.
            reasons.append(
                f"NOTE {reading.key} already at {reading.celsius:.1f}C, over its warn "
                f"threshold -- watch the first samples closely"
            )

    busy = busiest_other_cpu()
    if busy is not None and busy >= CONCURRENT_CPU_PERCENT:
        blocking = True
        pid = top_cpu_pid()
        reasons.append(
            f"another process is using {busy:.0f}% CPU{f' (pid {pid})' if pid else ''}"
            f" -- stacking a second heavy job is what crashed this machine"
        )

    return Preflight(
        ok=not blocking, reasons=reasons, readings=readings, blocking=blocking
    )


def load_average_1m() -> float:
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):  # pragma: no cover - platform dependent
        return 0.0


def cpu_usage_percent() -> float:
    """Rough system-wide CPU pressure, from the load average relative to core count.

    Load average is the honest cheap signal here. Reading /proc/stat deltas would need
    mutable module state across samples, and a subtly wrong version of that is worse
    than a coarse one -- it would report "quiet" while the box is saturated.
    """
    try:
        cores = os.cpu_count() or 1
    except Exception:  # pragma: no cover
        cores = 1
    return min(100.0, 100.0 * load_average_1m() / cores)


def busiest_other_cpu() -> Optional[float]:
    """Highest CPU% used by any single process other than this one."""
    best = 0.0
    me = os.getpid()
    try:
        import psutil  # optional
    except ImportError:
        return None
    for proc in psutil.process_iter(["pid", "cpu_percent"]):
        try:
            if proc.info["pid"] == me:
                continue
            value = float(proc.info["cpu_percent"] or 0.0)
            best = max(best, value)
        except Exception:  # pragma: no cover - processes come and go
            continue
    return best


def top_cpu_pid() -> Optional[int]:
    best_pid, best = None, 0.0
    me = os.getpid()
    try:
        import psutil
    except ImportError:
        return None
    for proc in psutil.process_iter(["pid", "cpu_percent"]):
        try:
            if proc.info["pid"] == me:
                continue
            value = float(proc.info["cpu_percent"] or 0.0)
            if value > best:
                best_pid, best = proc.info["pid"], value
        except Exception:  # pragma: no cover
            continue
    return best_pid


# --------------------------------------------------------------------------
# Sensors
# --------------------------------------------------------------------------


def parse_sensors_json(payload: str) -> list[Reading]:
    """Turn ``sensors -j`` output into readings.

    The JSON shape is ``{adapter: {feature_name: {"tempN_input": value}}}``, and which
    features exist varies by machine: coretemp exposes "Package id 0" plus one entry per
    core, while each NVMe adapter exposes "Composite". So this walks features by name
    rather than assuming an index, and ignores adapters we have no rule for.
    """
    import json

    payload = payload[payload.index("{") :] if "{" in payload else payload
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []

    packages: list[float] = []
    cores: list[float] = []
    disks: list[float] = []
    for adapter, features in data.items():
        if not isinstance(features, dict):
            continue
        values: list[float] = []
        for body in features.values():
            if not isinstance(body, dict):
                continue
            for key, raw in body.items():
                if key.endswith("_input"):
                    try:
                        values.append(float(raw))
                    except (TypeError, ValueError):
                        continue
        if not values:
            continue
        if adapter.startswith("coretemp"):
            package = [
                float(features[name]["temp1_input"])  # type: ignore[index]
                for name in features
                if name.startswith("Package")
                and isinstance(features.get(name), dict)
                and "temp1_input" in features[name]
            ]
            if package:
                packages.append(max(package))
            # The hottest core is what throttles the package.
            cores.append(max(values))
        elif adapter.startswith("nvme"):
            disks.append(max(values))

    readings = [Reading("cpu_package", v) for v in packages]
    readings += [Reading("cpu_core", v) for v in cores]
    readings += [Reading("nvme", v) for v in disks]
    return readings


def read_sensors() -> list[Reading]:
    try:
        result = subprocess.run(
            ["sensors", "-j"], capture_output=True, text=True, timeout=10, check=False
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return parse_sensors_json(result.stdout)


def read_gpu() -> list[Reading]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    readings: list[Reading] = []
    for line in output.splitlines():
        line = line.strip()
        try:
            readings.append(Reading("gpu", float(line)))
        except ValueError:
            continue
    return readings


def sample() -> list[Reading]:
    """One combined sample of everything readable."""
    return read_sensors() + read_gpu()


# --------------------------------------------------------------------------
# Durable telemetry
# --------------------------------------------------------------------------


class TemperatureLog:
    """Append-only CSV of samples, flushed and fsynced every write.

    The point is survivability. The first guard printed to a terminal, and when the
    machine crashed there was no record of what the temperature had been doing -- which
    is exactly the information needed to set a threshold. fsync per sample is cheap at
    five-second intervals and is what makes the file survive a hard power-off.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._handle)
        if self.path.stat().st_size == 0:
            self._writer.writerow(CSV_HEADER)
            self._flush()

    def _flush(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def write(self, elapsed: float, readings: Iterable[Reading], decision: str) -> None:
        by_key: dict[str, float] = {}
        for reading in readings:
            by_key[reading.key] = max(by_key.get(reading.key, 0.0), reading.celsius)
        nvme = [r.celsius for r in readings if r.key == "nvme"]
        self._writer.writerow(
            [
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                f"{elapsed:.1f}",
                f"{by_key.get('cpu_package', float('nan')):.1f}",
                f"{by_key.get('cpu_core', float('nan')):.1f}",
                f"{by_key.get('gpu', float('nan')):.1f}",
                f"{max(nvme) if nvme else float('nan'):.1f}",
                f"{load_average_1m():.2f}",
                decision,
            ]
        )
        self._flush()

    def close(self) -> None:
        try:
            self._flush()
            self._handle.close()
        except Exception:  # pragma: no cover
            pass


def describe(
    reading: Reading, limits: Optional[dict[str, tuple[float, float]]] = None
) -> str:
    table = limits if limits is not None else DEFAULT_LIMITS
    warn, stop = table.get(reading.key, UNRATED_LIMITS)
    return f"{reading.key} {reading.celsius:.1f}C (warn {warn:.0f}, stop {stop:.0f})"


# --------------------------------------------------------------------------
# Supervision
# --------------------------------------------------------------------------


def _signal_group(process: subprocess.Popen, sig: int) -> None:
    """Send *sig* to the child's whole process group, falling back to the child alone.

    ``start_new_session`` puts the child in its own group, so signalling the group
    reaches the Python process *and* anything it spawned -- a training run spawns a
    dataloader worker, and pausing only the parent leaves the workers hammering the CPU
    through the cool-down, which defeats the entire point.
    """
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            process.send_signal(sig)
        except (ProcessLookupError, ValueError):
            pass


def _pause_child(
    process: subprocess.Popen,
    pause_seconds: float,
    limits: Optional[dict[str, tuple[float, float]]],
    telemetry: Optional["TemperatureLog"],
    started: float,
) -> None:
    """SIGSTOP the child, wait for *pause_seconds*, then SIGCONT.

    Keeps sampling through the cool-down and logs it, so the CSV shows the pause as the
    flat, cooling segment it is rather than a silent gap. If the child dies while
    stopped (a checkpoint flush, or an OOM the pause cannot help with) we do not
    SIGCONT a corpse: there is nothing to resume, and the caller sees the real exit code
    on its next poll.
    """
    _signal_group(process, signal.SIGSTOP)
    deadline = time.time() + pause_seconds
    while time.time() < deadline:
        time.sleep(min(DEFAULT_INTERVAL, max(0.0, deadline - time.time())))
        if process.poll() is not None:
            print(
                "[thermal] child exited during the cool-down; not resuming",
                flush=True,
            )
            return
        readings = sample()
        elapsed = time.time() - started
        action, _offender = decide(readings, limits)
        telemetry and telemetry.write(elapsed, readings, f"paused_{action}")
    cooled = sample()
    hottest = max((r.celsius for r in cooled), default=float("nan"))
    print(
        f"[thermal] resuming after {pause_seconds:.0f}s; hottest sensor now "
        f"{hottest:.1f}C",
        flush=True,
    )
    _signal_group(process, signal.SIGCONT)


def _terminate(process: subprocess.Popen) -> None:
    """SIGTERM the child's whole process group, then SIGKILL if it lingers.

    start_new_session puts the child in its own group; without that this kills the
    guard too, which is the one version of "protection" that is actively harmful.
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()


def supervise(
    command: list[str],
    *,
    interval: int = DEFAULT_INTERVAL,
    limits: Optional[dict[str, tuple[float, float]]] = None,
    log_path: Optional[Path] = None,
    allow_hot_start: bool = False,
    start_celsius: Optional[float] = DEFAULT_START_CELSIUS,
    max_hot_seconds: float = DEFAULT_MAX_HOT_SECONDS,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    max_pauses: int = DEFAULT_MAX_PAUSES,
    dry_run: bool = False,
) -> int:
    """Run *command*, polling temperature, pausing it if the machine overheats.

    ``max_hot_seconds`` is how long the machine may stay continuously above a stop
    threshold before the run is paused. A brief spike costs nothing and the timer resets
    the moment it settles; only sustained heat counts.

    On sustained heat the child's process group is SIGSTOPped for ``pause_seconds`` and
    then resumed, up to ``max_pauses`` times. Pausing rather than killing keeps a
    half-finished run alive -- it still holds its VRAM but does no work, so the machine
    actually cools. Exceeding the pause budget stops the run for good, because a
    workload that overheats three times is a workload that needs to get smaller, not
    one that needs more patience.

    Returns the command's status, 124 if stopped for heat, 3 if preflight refused.
    """
    if dry_run:
        print("would run:", " ".join(command))
        print(
            f"sampling every {interval}s; pausing after {max_hot_seconds:.0f}s "
            f"continuously above a stop threshold, for {pause_seconds:.0f}s, "
            f"up to {max_pauses} time(s)"
        )
        return 0

    if not allow_hot_start:
        check = preflight(limits, start_celsius)
        for reason in check.reasons:
            print(f"[thermal] PREFLIGHT: {reason}", flush=True)
        if not check.ok:
            print(
                "[thermal] refusing to start. Wait for the machine to cool, or run "
                "with --allow-hot-start if you have checked the other jobs yourself.",
                flush=True,
            )
            return 3

    telemetry = TemperatureLog(log_path) if log_path else None
    print(f"[thermal] guarding: {' '.join(command)}", flush=True)
    if pause_seconds > 0:
        budget = (
            f"up to {max_pauses} time(s)" if max_pauses > 0 else "as often as needed"
        )
        print(
            f"[thermal] sampling every {interval}s; pausing for {pause_seconds:.0f}s "
            f"after {max_hot_seconds:.0f}s continuously above a stop threshold "
            f"({budget})",
            flush=True,
        )
    else:
        print(
            f"[thermal] sampling every {interval}s; killing after "
            f"{max_hot_seconds:.0f}s continuously above a stop threshold",
            flush=True,
        )
    if telemetry:
        print(f"[thermal] logging to {telemetry.path}", flush=True)

    process = subprocess.Popen(command, start_new_session=True)
    started = time.time()
    # Seconds the machine has been continuously above a stop threshold. Reset as soon
    # as it drops back, so a brief spike costs nothing and only sustained heat counts.
    hot_since: Optional[float] = None
    pauses = 0
    try:
        while process.poll() is None:
            time.sleep(interval)
            if process.poll() is not None:
                break
            readings = sample()
            elapsed = time.time() - started
            action, offender = decide(readings, limits)
            telemetry and telemetry.write(elapsed, readings, action)

            if action == "stop" and offender is not None:
                if hot_since is None:
                    hot_since = elapsed
                    print(
                        f"[thermal] HOT {describe(offender, limits)} -- tolerating "
                        f"up to {max_hot_seconds:.0f}s while it settles",
                        flush=True,
                    )
                hot_for = elapsed - hot_since
                if hot_for >= max_hot_seconds:
                    # --pause-seconds 0 keeps the original kill-only behaviour, which is
                    # still right for a cheap command that can simply be re-run.
                    # max_pauses == 0 means unlimited: the owner's rule is to pause and
                    # carry on, so the run only stops when a budget was explicitly asked for.
                    budget_spent = max_pauses > 0 and pauses >= max_pauses
                    if pause_seconds <= 0 or budget_spent:
                        if pause_seconds > 0:
                            print(
                                f"[thermal] stopping: {hot_for:.0f}s above the stop "
                                f"threshold for the {pauses + 1}th time, and the pause "
                                f"budget of {max_pauses} is spent. Pausing is not keeping "
                                f"this workload cool, so continuing would just repeat "
                                f"this. This machine has crashed from heat twice.",
                                flush=True,
                            )
                        else:
                            print(
                                f"[thermal] stopping: {hot_for:.0f}s continuously above "
                                f"the stop threshold. This machine has crashed from heat "
                                f"twice.",
                                flush=True,
                            )
                        _terminate(process)
                        return 124
                    pauses += 1
                    shown = f"{pauses}/{max_pauses}" if max_pauses > 0 else str(pauses)
                    print(
                        f"[thermal] PAUSING (pause {shown}) for "
                        f"{pause_seconds:.0f}s: {hot_for:.0f}s above the stop threshold "
                        f"({describe(offender, limits)}). The run resumes afterwards.",
                        flush=True,
                    )
                    _pause_child(process, pause_seconds, limits, telemetry, started)
                    # The cool-down is not the run's fault, so it must not count towards
                    # elapsed time in the telemetry, and the hot timer restarts from zero
                    # once work resumes.
                    started += pause_seconds
                    hot_since = None
                    continue
                continue

            if hot_since is not None:
                print(
                    f"[thermal] cooled after {elapsed - hot_since:.0f}s hot; "
                    f"continuing",
                    flush=True,
                )
                hot_since = None
            if action == "warn" and offender is not None:
                print(f"[thermal] WARN {describe(offender, limits)}", flush=True)
    except KeyboardInterrupt:
        print("\n[thermal] interrupted; stopping the child", flush=True)
        _terminate(process)
        return 130
    finally:
        if telemetry:
            telemetry.close()
    return process.returncode


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="*", help="the command to guard")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help="seconds between samples (default 5)",
    )
    parser.add_argument(
        "--max-hot-seconds",
        type=float,
        default=DEFAULT_MAX_HOT_SECONDS,
        help="seconds continuously above a stop threshold before the run is paused "
        f"(default {DEFAULT_MAX_HOT_SECONDS:.0f})",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=DEFAULT_PAUSE_SECONDS,
        help="how long to pause the run for when it gets too hot, 0 to kill instead "
        f"(default {DEFAULT_PAUSE_SECONDS:.0f})",
    )
    parser.add_argument(
        "--max-pauses",
        type=int,
        default=DEFAULT_MAX_PAUSES,
        help="how many times one run may pause before it is stopped outright; "
        "0 (the default) means pause as often as it takes, per the owner's rule "
        f"(default {DEFAULT_MAX_PAUSES})",
    )
    parser.add_argument(
        "--start-celsius",
        type=float,
        default=DEFAULT_START_CELSIUS,
        help=(
            "refuse to START above this temperature; the stop threshold is separate. "
            f"(default {DEFAULT_START_CELSIUS:.0f})"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="print one sample and the preflight decision, then exit",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-hot-start",
        action="store_true",
        help="skip the preflight refusal (you checked the other jobs)",
    )
    parser.add_argument(
        "--log", default=None, help="append samples to this CSV; survives a crash"
    )
    args = parser.parse_args(argv)

    if args.check:
        readings = sample()
        if not readings:
            print("No temperature sensors readable (install lm-sensors).")
            return 2
        for reading in sorted(readings, key=lambda r: r.celsius, reverse=True):
            print(
                f"  {describe(reading):<46} {classify(reading.key, reading.celsius).upper()}"
            )
        print(
            f"[thermal] load1={load_average_1m():.2f} "
            f"cpu_pressure={cpu_usage_percent():.0f}%"
        )
        check = preflight(start_celsius=args.start_celsius)
        for reason in check.reasons:
            print(f"[thermal] PREFLIGHT: {reason}")
        print(f"[thermal] decision: {'RUN' if check.ok else 'REFUSE'}")
        return 0

    if not args.command:
        print(__doc__)
        return 2

    return supervise(
        args.command,
        interval=args.interval,
        log_path=Path(args.log) if args.log else None,
        allow_hot_start=args.allow_hot_start,
        max_hot_seconds=args.max_hot_seconds,
        pause_seconds=args.pause_seconds,
        max_pauses=args.max_pauses,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
