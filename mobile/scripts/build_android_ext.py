"""Cross-compile momentum's own Cython extensions for the Android ABI.

python-for-android cythonizes *recipe* sources (``CythonRecipe``) but never an
application's own ``.pyx`` files, and ``mobile/buildozer.spec`` ships ``momentum``
as plain Python. ``momentum._assessments_cy`` and ``momentum._charts_cy`` were
therefore never importable on Android, and the ``except ImportError`` fallbacks in
``momentum/domain/assessments/scoring.py`` and ``momentum/ui/charts.py`` quietly
selected the pure-Python code the Cython modules exist to replace.

This script closes that gap. It compiles the Cython-generated ``.c`` files that
are already committed to the repository (so neither Cython nor a host Python is
needed) against the *target* CPython that python-for-android built, using the NDK
toolchain it downloaded. Run it after buildozer's first pass -- which builds
hostpython3 and the target CPython and therefore creates the headers and
``_sysconfigdata`` this reads -- and before the second pass, so the ``.so`` files
are picked up by the app bundle.

It never raises: anything missing is reported as a warning and the build continues
with the pure-Python fallbacks (the same graceful-degradation contract the
``except ImportError`` guards implement).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (Cython source, compiled module name). Mirrors momentum/setup.py's ext_modules.
EXTENSIONS: tuple[tuple[str, str], ...] = (
    ("assessments_cy.pyx", "_assessments_cy"),
    ("charts_cy.pyx", "_charts_cy"),
    ("timer_cy.pyx", "_timer_cy"),
)

# p4a's android CPython reports EXT_SUFFIX as ``.cpython-<major><minor>.so``. Only
# used when the real value cannot be read from the target _sysconfigdata.
FALLBACK_EXT_SUFFIX = ".cpython-311.so"

_EXT_SUFFIX_RE = re.compile(r"""['"]EXT_SUFFIX['"]\s*:\s*['"]([^'"]+)['"]""")


def find_target_python(build_dir: Path, arch: str) -> Path | None:
    """Locate the target CPython install python-for-android built for ``arch``.

    p4a lays this out as ``<build dir>/python-installs/<distribution>/<arch>`` but
    the distribution name varies (it is the app package name for buildozer), so
    the directory is discovered rather than assumed.
    """
    patterns = (
        f"python-installs/*/{arch}",
        f"**/python-installs/*/{arch}",
    )
    seen: set[Path] = set()
    for pattern in patterns:
        for candidate in sorted(build_dir.glob(pattern)):
            if candidate in seen or not candidate.is_dir():
                continue
            seen.add(candidate)
            if list(candidate.glob("include/python3*")) or list(
                candidate.glob("lib/python3*")
            ):
                return candidate
    return None


def read_ext_suffix(target_dir: Path) -> str:
    """Read ``EXT_SUFFIX`` from the target interpreter's ``_sysconfigdata``.

    The Android interpreter cannot execute on the build host, so the value is
    parsed out of the generated module source instead of being computed.
    """
    for sysconfig_data in sorted(target_dir.glob("lib/python3*/_sysconfigdata*.py")):
        try:
            text = sysconfig_data.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = _EXT_SUFFIX_RE.search(text)
        if match:
            return match.group(1)
    return FALLBACK_EXT_SUFFIX


def find_include_dirs(target_dir: Path) -> list[Path]:
    """Return the target interpreter's C header directories."""
    dirs = [path for path in target_dir.glob("include/python3*") if path.is_dir()]
    return dirs


def find_ndk_clang(api: int, env: dict[str, str] | None = None) -> Path | None:
    """Return the NDK ``aarch64-linux-android<api>-clang`` driver, if present."""
    env = os.environ if env is None else env
    roots = [
        env.get("ANDROID_NDK_HOME"),
        env.get("ANDROID_NDK_ROOT"),
        env.get("ANDROID_NDK"),
    ]
    roots.extend(
        str(path) for path in sorted(Path.home().glob(".buildozer/android/platform/android-ndk-r*"))
    )
    for root in (Path(r) for r in roots if r):
        for host in root.glob("toolchains/llvm/prebuilt/*"):
            for candidate_api in (api, 21):
                driver = host / "bin" / f"aarch64-linux-android{candidate_api}-clang"
                if driver.is_file():
                    return driver
    return None


def build_command(
    clang: Path,
    c_source: Path,
    out_path: Path,
    include_dirs: list[Path],
    target_dir: Path,
) -> list[str]:
    """Return the clang command line that produces one extension module."""
    command = [str(clang), "-shared", "-fPIC", "-O2", "-DNDEBUG"]
    command += [f"-I{directory}" for directory in include_dirs]
    # Link the target libpython when one is present, matching how p4a's own
    # recipes build. The dynamic linker reuses the already-mapped soname.
    libpython = next(iter(sorted(target_dir.glob("libpython3*.so"))), None)
    if libpython is not None:
        command += [f"-L{target_dir}", f"-l{libpython.stem[3:]}"]
    command += ["-o", str(out_path), str(c_source)]
    return command


def warn(message: str) -> None:
    """Report a non-fatal problem (annotated for GitHub Actions when present)."""
    prefix = "::warning::" if os.environ.get("GITHUB_ACTIONS") else "WARNING: "
    print(f"{prefix}{message}", file=sys.stderr, flush=True)


def info(message: str) -> None:
    """Report progress."""
    print(message, flush=True)


def build_all(
    build_dir: Path,
    arch: str,
    min_api: int,
    dry_run: bool = False,
    root: Path = ROOT,
) -> int:
    """Compile every extension for ``arch``. Returns the number of failures."""
    target_dir = find_target_python(build_dir, arch)
    if target_dir is None:
        warn(f"no target CPython install for {arch} under {build_dir}; skipping")
        return 1

    include_dirs = find_include_dirs(target_dir)
    if not include_dirs:
        warn(f"no C headers under {target_dir}; skipping")
        return 1

    clang = find_ndk_clang(min_api)
    if clang is None:
        warn(f"no NDK clang driver found for API {min_api}; skipping")
        return 1

    ext_suffix = read_ext_suffix(target_dir)
    info(f"target python : {target_dir}")
    info(f"ext suffix    : {ext_suffix}")
    info(f"clang         : {clang}")

    failures = 0
    for pyx_name, module in EXTENSIONS:
        c_source = root / "momentum" / f"{Path(pyx_name).stem}.c"
        if not c_source.is_file():
            warn(f"{c_source.name} is missing (run 'python setup.py build_ext --inplace')")
            failures += 1
            continue
        out_path = root / "momentum" / f"{module}{ext_suffix}"
        command = build_command(clang, c_source, out_path, include_dirs, target_dir)
        info("  " + " ".join(command))
        if dry_run:
            continue
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            warn(f"compiling {module} failed: {result.stderr.strip()}")
            failures += 1
            continue
        info(f"  built {out_path.name}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        default="/tmp/buildozer-build/android/platform",
        help="python-for-android build directory created by pass 1",
    )
    parser.add_argument("--arch", default="arm64-v8a", help="target ABI directory")
    parser.add_argument(
        "--min-api",
        type=int,
        default=26,
        help="Android API level for the clang driver (android.minapi)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the compile commands without running them",
    )
    args = parser.parse_args(argv)

    failures = build_all(
        build_dir=Path(args.build_dir),
        arch=args.arch,
        min_api=args.min_api,
        dry_run=args.dry_run,
    )
    if failures:
        warn(
            f"{failures} extension(s) not compiled for {args.arch}; "
            "the APK keeps its pure-Python fallbacks"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

