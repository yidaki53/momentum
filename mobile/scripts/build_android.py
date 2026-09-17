"""Run local Android builds with an explicit Python and Java environment."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def java_major(home: Path) -> int:
    try:
        result = subprocess.run(
            [str(home / "bin/java"), "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    match = re.search(r'version "(\d+)(?:\.(\d+))?', result.stderr + result.stdout)
    if not match:
        return 0
    major = int(match[1])
    return int(match[2] or 0) if major == 1 else major


def build_environment(root: Path = ROOT) -> dict[str, str]:
    env = os.environ.copy()
    venv = root / ".venv"
    if not (venv / "bin/buildozer").is_file():
        raise RuntimeError(
            "Install the project dev dependencies with poetry install --with dev first."
        )
    candidates = []
    if env.get("JAVA_HOME"):
        candidates.append(Path(env["JAVA_HOME"]))
    java = shutil.which("javac")
    if java:
        candidates.append(Path(java).resolve().parents[1])
    candidates.extend(sorted(Path("/usr/lib/jvm").glob("*")))
    home = next(
        (p for p in candidates if (p / "bin/javac").is_file() and java_major(p) >= 17),
        None,
    )
    if home is None:
        raise RuntimeError(
            "Install JDK 17 or newer and set JAVA_HOME to its directory."
        )
    env["JAVA_HOME"] = str(home)
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = os.pathsep.join(
        [str(venv / "bin"), str(home / "bin"), env.get("PATH", "")]
    )
    env.pop("PYTHONHOME", None)
    env["PIP_FIND_LINKS"] = (root / "mobile/p4a-wheels").as_uri()
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Validate environment without building"
    )
    parser.add_argument(
        "--aab",
        action="store_true",
        help="Build release AAB and restore the spec afterward",
    )
    args = parser.parse_args()
    env = build_environment()
    for key in ("JAVA_HOME", "VIRTUAL_ENV", "PIP_FIND_LINKS"):
        print(f"{key}={env[key]}", flush=True)
    if args.check:
        return 0
    spec = ROOT / "mobile/buildozer.spec"
    original = spec.read_bytes()
    try:
        if args.aab:
            text = original.decode()
            text, count = re.subn(
                r"(?m)^android.release_artifact\s*=.*$",
                "android.release_artifact = aab",
                text,
            )
            if count != 1:
                raise RuntimeError("Expected one android.release_artifact setting")
            spec.write_text(text)
        command = [
            str(ROOT / ".venv/bin/buildozer"),
            "-v",
            "android",
            "release" if args.aab else "debug",
        ]
        return subprocess.run(
            command, cwd=ROOT / "mobile", env=env, check=False
        ).returncode
    finally:
        if args.aab:
            spec.write_bytes(original)


if __name__ == "__main__":
    raise SystemExit(main())
