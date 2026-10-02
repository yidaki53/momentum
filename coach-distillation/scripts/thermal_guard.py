#!/usr/bin/env python3
"""Run a command, but stop it if the machine gets too hot.

Why this exists: a CPU fine-tuning run pinned every core for half an hour and this
workstation shut down from thermal overload. That is a real cost -- a dead machine, a
lost run, and a rule that was never written down. So the rule is now written down, and
this is its enforcement.

The design point is that *all the logic is a pure function over readings*. Which sensor
matters, what a reading means, and whether to keep going are decided by functions that
never touch the hardware, so they can be unit-tested without a thermocouple. The only
code that knows about ``sensors`` and ``nvidia-smi`` is the two reader functions at the
bottom, and a missing sensor degrades to "no information" rather than to "safe".

Usage::

    python scripts/thermal_guard.py -- python scripts/train_distill.py ...
    python scripts/thermal_guard.py --check                 # one-shot reading
    python scripts/thermal_guard.py --watch --interval 10    # just monitor

Exit codes: the command's own status, or 124 if the guard stopped it.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Iterable, Optional

# --- thresholds --------------------------------------------------------------
# Chosen from this machine's own reported limits, with headroom:
#   CPU package high=100 crit=100 (coretemp); sustained 90 is where a workstation
#   starts throttling hard and where a long run is a gamble.
#   RTX 4080 Tj max is 89C; 85 stops before thermal throttling hurts throughput anyway.
#   NVMe high=82.8 crit=84.8; drives throttle hard well before that.
DEFAULT_LIMITS: dict[str, tuple[float, float]] = {
    # sensor key: (warn_c, stop_c)
    "cpu_package": (80.0, 90.0),
    "cpu_core": (85.0, 95.0),
    "gpu": (80.0, 85.0),
    "nvme": (70.0, 80.0),
}

# Fallback for a sensor we sample but have no rating for. Deliberately coarse: no
# consumer chip survives this, but nothing runs here normally.
UNRATED_LIMITS: tuple[float, float] = (100.0, 120.0)


@dataclass(frozen=True)
class Reading:
    """One sensor sample."""

    key: str
    celsius: float

    @property
    def state(self) -> str:
        return classify(self.key, self.celsius)


def classify(
    key: str, celsius: float, limits: Optional[dict[str, tuple[float, float]]] = None
) -> str:
    """Return ``"ok"``, ``"warn"`` or ``"stop"`` for one reading.

    An unrated sensor still gets a generic ceiling rather than being treated as safe
    forever. Absence of information is not evidence of safety -- but a chip reading
    150C is an emergency whatever its rating says, so ``UNRATED_LIMITS`` is a coarse
    backstop rather than an assumption that nothing is wrong.
    """
    table = limits if limits is not None else DEFAULT_LIMITS
    warn, stop = table.get(key, UNRATED_LIMITS)
    if celsius >= stop:
        return "stop"
    if celsius >= warn:
        return "warn"
    return "ok"


def decide(
    readings: Iterable[Reading], limits: Optional[dict[str, tuple[float, float]]] = None
) -> tuple[str, Optional[Reading]]:
    """Fold a sample into a decision.

    Returns ``(action, offender)`` where action is ``"run"``, ``"warn"`` or ``"stop"``.
    ``warn`` never stops the job -- it only makes noise, because a brief spike above the
    warning threshold under load is normal and killing on it would make the guard
    useless. Only a breach of a *stop* threshold, sustained for a few consecutive
    samples, ends the run; that hysteresis is what distinguishes a real thermal event
    from one noisy reading.
    """
    worst_state = "run"
    worst: Optional[Reading] = None
    for reading in readings:
        state = classify(reading.key, reading.celsius, limits)
        if state == "stop":
            return ("stop", reading)
        if state == "warn" and worst is None:
            worst_state = "warn"
            worst = reading
    return (worst_state, worst)


# --- readers -----------------------------------------------------------------

_CPU_PACKAGE = re.compile(r"^Package id 0:\s*\+?([\d.]+)", re.MULTILINE)
_CPU_CORE = re.compile(r"^Core \d+:\s*\+?([\d.]+)", re.MULTILINE)
_NVME = re.compile(r"^Composite:\s*\+?([\d.]+)", re.MULTILINE)


def parse_sensors_json(payload: str) -> list[Reading]:
    """Turn ``sensors -j`` output into readings.

    The JSON shape is ``{adapter: {feature_name: {"tempN_input": value}}}``, and which
    features exist varies by machine: coretemp exposes "Package id 0" plus one entry per
    core, while each NVMe adapter exposes "Composite". So this walks the features by name
    rather than assuming a fixed index, and simply ignores adapters we have no rule for
    (wifi radios, virtual devices, ACPI zones, battery).
    """
    import json

    # lm-sensors writes diagnostics to stderr, but a machine with a partially readable
    # sensor can still emit them on stdout ahead of the JSON.
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
        for feature, body in features.items():
            if not isinstance(body, dict):
                continue
            for key, raw in body.items():
                if not key.endswith("_input"):
                    continue
                try:
                    values.append(float(raw))
                except (TypeError, ValueError):
                    continue
        if not values:
            continue
        if adapter.startswith("coretemp"):
            package = [
                float(features[name].get("temp1_input"))  # type: ignore[union-attr]
                for name in features
                if name.startswith("Package") and "temp1_input" in features[name]
            ]
            if package:
                packages.append(max(package))
            # The hottest core is what actually throttles the package.
            cores.append(max(values))
        elif adapter.startswith("nvme"):
            disks.append(max(values))

    readings = [Reading("cpu_package", value) for value in packages]
    readings += [Reading("cpu_core", value) for value in cores]
    readings += [Reading("nvme", value) for value in disks]
    return readings


def read_sensors() -> list[Reading]:
    """Parse ``sensors`` output. Returns [] when lm-sensors is absent."""
    try:
        result = subprocess.run(
            ["sensors", "-j"], capture_output=True, text=True, timeout=10, check=False
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return _read_sensors_plain()
    parsed = parse_sensors_json(result.stdout)
    return parsed if parsed else _read_sensors_plain()


def _read_sensors_plain() -> list[Reading]:
    """Fallback parser for the human-readable ``sensors`` format."""
    try:
        output = subprocess.run(
            ["sensors"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    package = _CPU_PACKAGE.search(output)
    readings: list[Reading] = []
    if package:
        readings.append(Reading("cpu_package", float(package.group(1))))
    cores = _CPU_CORE.findall(output)
    if cores:
        # The hottest core is what throttles the package.
        readings.append(Reading("cpu_core", max(float(value) for value in cores)))
    nvme = _NVME.search(output)
    if nvme:
        readings.append(Reading("nvme", float(nvme.group(1))))
    return readings


def read_gpu() -> list[Reading]:
    """Parse ``nvidia-smi`` for GPU core temperature."""
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
        if not line:
            continue
        try:
            readings.append(Reading("gpu", float(line)))
        except ValueError:
            continue
    return readings


def sample() -> list[Reading]:
    """One combined sample of everything we can read."""
    return read_sensors() + read_gpu()


# --- supervision -------------------------------------------------------------


def describe(
    reading: Reading, limits: Optional[dict[str, tuple[float, float]]] = None
) -> str:
    """Human-readable reading plus the limits actually being enforced.

    The limits must be threaded through rather than read from DEFAULT_LIMITS: a caller
    that tightened them (as the self-test does) would otherwise be told one threshold and
    have another enforced, which is worse than no output.
    """
    table = limits if limits is not None else DEFAULT_LIMITS
    warn, stop = table.get(reading.key, UNRATED_LIMITS)
    return f"{reading.key} {reading.celsius:.1f}C (warn {warn:.0f}, stop {stop:.0f})"


def supervise(
    command: list[str],
    *,
    interval: int = 15,
    patience: int = 3,
    limits: Optional[dict[str, tuple[float, float]]] = None,
    dry_run: bool = False,
) -> int:
    """Run *command*, polling temperature, and stop it if it overheats.

    ``patience`` is how many consecutive stop-level samples are required before killing
    anything. Three samples at 15s means a run must be genuinely hot for 45 seconds,
    which no single spike survives.
    """
    if dry_run:
        print("would run:", " ".join(command))
        print(f"would sample every {interval}s, stopping after {patience} hot samples")
        return 0

    print(f"[thermal] guarding: {' '.join(command)}", flush=True)
    print(
        f"[thermal] polling every {interval}s; "
        f"stopping after {patience} consecutive samples over a stop threshold",
        flush=True,
    )

    # start_new_session puts the child in its own process group. Without it the child
    # shares ours, and the kill below takes the guard down with it -- which is exactly
    # what happened the first time this was tested.
    process = subprocess.Popen(command, start_new_session=True)
    breaches = 0
    try:
        while process.poll() is None:
            time.sleep(interval)
            if process.poll() is not None:
                break
            action, offender = decide(sample(), limits)
            if action == "warn" and offender is not None:
                print(f"[thermal] WARN {describe(offender, limits)}", flush=True)
                continue
            if action == "stop" and offender is not None:
                breaches += 1
                print(
                    f"[thermal] HOT {describe(offender, limits)} "
                    f"({breaches}/{patience})",
                    flush=True,
                )
                if breaches >= patience:
                    print(
                        "[thermal] stopping the run: sustained overheat. "
                        "The machine has crashed from this before.",
                        flush=True,
                    )
                    _terminate(process)
                    return 124
            else:
                breaches = 0
    except KeyboardInterrupt:
        print("\n[thermal] interrupted; stopping the child", flush=True)
        _terminate(process)
        return 130
    return process.returncode


def _terminate(process: subprocess.Popen) -> None:
    """SIGTERM the whole process group, then SIGKILL if it lingers.

    Killing the group matters: torch spawns worker processes, and signalling only the
    parent would leave the actual compute running and the machine just as hot.
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="*", help="the command to guard")
    parser.add_argument(
        "--interval", type=int, default=15, help="seconds between samples (default 15)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="consecutive over-temp samples before stopping (default 3)",
    )
    parser.add_argument(
        "--check", action="store_true", help="print one temperature sample and exit"
    )
    parser.add_argument(
        "--watch", action="store_true", help="print samples without running anything"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        readings = sample()
        if not readings:
            print(
                "No temperature sensors readable (install lm-sensors). "
                "The guard cannot protect the machine."
            )
            return 2
        action, offender = decide(readings)
        for reading in sorted(readings, key=lambda r: r.celsius, reverse=True):
            print(f"  {describe(reading):<44} {reading.state.upper()}")
        print(f"[thermal] decision: {action.upper()}")
        return 0

    if args.watch or not args.command:
        if args.watch:
            try:
                while True:
                    action, offender = decide(sample())
                    stamp = time.strftime("%H:%M:%S")
                    hottest = max(sample(), key=lambda r: r.celsius, default=None)
                    detail = describe(hottest) if hottest else "no sensors"
                    print(f"{stamp} {action.upper():<5} {detail}", flush=True)
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                return 0
        print(__doc__)
        return 2

    return supervise(
        args.command,
        interval=args.interval,
        patience=args.patience,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
