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
3. **Ramp detection.** The dangerous thing is often the *rate* of climb, not the
   absolute value. A sensor gaining 20C in a minute is stopped even if it has not yet
   reached any threshold -- that is a thermal runaway, not a hot machine.
4. **A preflight gate.** The guard now REFUSES to start when the machine is already warm
   or already busy. The crash happened with two heavy jobs sharing the box; stacking a
   second one is exactly the failure mode, and a guard that only supervises its own child
   cannot see it.
5. **Five-second sampling.** Fifteen seconds is long enough for a laptop to overshoot.
6. **System-load awareness.** Other processes' CPU use is read and reported, because the
   heat is the machine's, not the guarded job's.

Every decision is a pure function over readings, so all of this is testable on CI
hardware with no sensors at all.

Usage::

    python scripts/thermal_guard.py --check
    python scripts/thermal_guard.py --log out/thermal.csv -- python scripts/train.py ...
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
from dataclasses import dataclass
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
    "cpu_package": (70.0, 78.0),
    "cpu_core": (75.0, 85.0),
    "gpu": (72.0, 80.0),
    "nvme": (60.0, 70.0),
}

# Fallback for a sensor we sample but have no rating for. Coarse but not a free pass.
UNRATED_LIMITS: tuple[float, float] = (100.0, 120.0)

# A sensor gaining this many degrees per minute is a runaway, whatever its absolute
# reading. Detecting slope catches the failure that absolute thresholds miss: a fan that
# has stopped, or a job that has started a much heavier phase.
RAMP_C_PER_MIN = 20.0

# Sampling cadence. Five seconds, not fifteen: a small laptop can overshoot a
# threshold in the time a 15s gap takes to notice, and two consecutive hot samples at
# 5s means a sustained event rather than a single spike.
DEFAULT_INTERVAL = 5
DEFAULT_PATIENCE = 2

# A rate computed over less than this is noise, not a ramp.
MIN_RAMP_SPAN_SECONDS = 10.0

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


def ramp_c_per_minute(
    history: list[tuple[float, float]], window_seconds: float = 60.0
) -> float:
    """Peak rate of change over the last *window_seconds* of ``(elapsed, celsius)``.

    Returns degrees per minute, or 0.0 when there is not enough history. Comparing
    endpoints rather than consecutive samples keeps this stable against noise.
    """
    if len(history) < 2:
        return 0.0
    newest_time, newest_value = history[-1]
    reference = [entry for entry in history if newest_time - entry[0] <= window_seconds]
    if len(reference) < 2:
        return 0.0
    oldest_time, oldest_value = reference[0]
    span = newest_time - oldest_time
    # Two samples a second apart produce an enormous, meaningless rate. Requiring a
    # minimum span stops sensor noise from being reported as a thermal runaway.
    if span < MIN_RAMP_SPAN_SECONDS:
        return 0.0
    return max(0.0, (newest_value - oldest_value) / span * 60.0)


def ramp_is_dangerous(
    history: list[tuple[float, float]], threshold: float = RAMP_C_PER_MIN
) -> bool:
    return ramp_c_per_minute(history) >= threshold


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
    readings: list[Reading]
    blocking: bool = False


def preflight(limits: Optional[dict[str, tuple[float, float]]] = None) -> Preflight:
    """Decide whether it is safe to start a heavy run right now.

    This is the check that would have prevented the second crash. The machine was
    already at 89C with another job at 95% CPU; a guard that only supervises its own
    child cannot see that, and by the time temperatures climb under a second load there
    may be no time left to react.
    """
    reasons: list[str] = []
    blocking = False
    readings = sample()

    if not readings:
        # Not a refusal: an unreadable sensor is a reason to be careful, not a reason to
        # block work forever. But it is worth saying out loud.
        reasons.append("no temperature sensors readable; running unprotected")

    for reading in readings:
        state = classify(reading.key, reading.celsius, limits)
        if state == "stop":
            blocking = True
            reasons.append(
                f"{reading.key} already at {reading.celsius:.1f}C, over its stop "
                f"threshold -- let it cool first"
            )
        elif state == "warn":
            # Warn blocks a *start* even though it does not stop a running job.
            # Starting with the CPU already at 70C leaves no margin for the heat the
            # run itself will produce, and this machine has crashed twice already.
            blocking = True
            reasons.append(
                f"{reading.key} already at {reading.celsius:.1f}C, over its warn "
                f"threshold -- starting hot leaves no headroom for the run itself"
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
    patience: int = DEFAULT_PATIENCE,
    limits: Optional[dict[str, tuple[float, float]]] = None,
    log_path: Optional[Path] = None,
    allow_hot_start: bool = False,
    dry_run: bool = False,
) -> int:
    """Run *command*, polling temperature, stopping it if the machine overheats.

    ``patience`` is how many consecutive stop-level samples are required. Two at the
    5-second default means a real, sustained heat event, which no single spike survives.

    Returns the command's status, 124 if stopped for heat, 3 if preflight refused.
    """
    if dry_run:
        print("would run:", " ".join(command))
        print(f"sampling every {interval}s; stopping after {patience} hot samples")
        return 0

    if not allow_hot_start:
        check = preflight(limits)
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
    print(
        f"[thermal] sampling every {interval}s; stop after {patience} consecutive "
        f"hot samples or a {RAMP_C_PER_MIN:.0f}C/min climb",
        flush=True,
    )
    if telemetry:
        print(f"[thermal] logging to {telemetry.path}", flush=True)

    process = subprocess.Popen(command, start_new_session=True)
    started = time.time()
    breaches = 0
    # Per-sensor temperature history, for ramp detection.
    histories: dict[str, list[tuple[float, float]]] = {}
    try:
        while process.poll() is None:
            time.sleep(interval)
            if process.poll() is not None:
                break
            readings = sample()
            elapsed = time.time() - started
            action, offender = decide(readings, limits)

            for reading in readings:
                history = histories.setdefault(reading.key, [])
                history.append((elapsed, reading.celsius))
                del history[:-40]  # a two-minute tail is plenty

            # A runaway climb is stopped even when no absolute threshold is reached
            # yet. This is the failure an absolute threshold misses: a fan that has
            # stopped, or a job that has entered a much heavier phase.
            climbing = [
                reading
                for reading in readings
                if ramp_is_dangerous(histories[reading.key])
            ]
            telemetry and telemetry.write(elapsed, readings, action)

            if climbing:
                worst_climb = max(
                    climbing, key=lambda r: ramp_c_per_minute(histories[r.key])
                )
                rate = ramp_c_per_minute(histories[worst_climb.key])
                breaches += 1
                print(
                    f"[thermal] CLIMBING {worst_climb.key} "
                    f"{worst_climb.celsius:.1f}C and gaining {rate:.0f}C/min "
                    f"({breaches}/{patience})",
                    flush=True,
                )
                if breaches >= patience:
                    print(
                        "[thermal] stopping: thermal runaway. This machine has "
                        "crashed from heat twice.",
                        flush=True,
                    )
                    _terminate(process)
                    return 124
                continue

            if action == "warn" and offender is not None:
                print(f"[thermal] WARN {describe(offender, limits)}", flush=True)
                continue
            if action == "stop" and offender is not None:
                breaches += 1
                print(
                    f"[thermal] HOT {describe(offender, limits)} ({breaches}/{patience})",
                    flush=True,
                )
                if breaches >= patience:
                    print(
                        "[thermal] stopping: sustained overheat. This machine has "
                        "crashed from this twice.",
                        flush=True,
                    )
                    _terminate(process)
                    return 124
                continue
            breaches = 0
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
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="consecutive over-temp samples before stopping (default 2)",
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
        check = preflight()
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
        patience=args.patience,
        log_path=Path(args.log) if args.log else None,
        allow_hot_start=args.allow_hot_start,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
