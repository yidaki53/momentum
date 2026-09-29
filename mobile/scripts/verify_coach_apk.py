"""Check required coach payloads without executing Android machine code."""

from __future__ import annotations

import argparse
import io
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

# Compiled Cython modules built by mobile/scripts/build_android_ext.py. Their
# absence is not fatal -- momentum falls back to pure Python -- so the check below
# is opt-in via --require-cython.
CYTHON_MODULES = ("_assessments_cy", "_charts_cy")

# Byte strings that must appear in the bundled engine.py for the native loader to
# be wired up correctly. ``libllama.so`` declares bare-soname ``DT_NEEDED``
# entries and carries no ``DT_RUNPATH``, so the engine must point llama_cpp at the
# APK native library directory -- the only place that also holds the
# ``libc++_shared.so`` the wheel never ships -- and preload the chain itself. An
# APK missing any of these builds, installs and launches fine while the coach
# stays permanently unavailable, because packaging checks cannot see it.
# Exported so test fixtures derive from it instead of duplicating it.
ENGINE_REQUIRED_MARKERS: tuple[bytes, ...] = (
    b"LLAMA_CPP_LIB_PATH",
    b"llama_cpp",
    b"nativeLibraryDir",
    b"_preload_android_libraries",
    b"c++_shared",
)

# Buildozer splits the APK into two tarballs: third-party packages and the
# interpreter go inside ``libpybundle.so``, while the app's own sources -- the
# ``momentum`` package -- go inside ``assets/private.tar``. A real APK therefore
# contains no ``momentum`` entry at all in the bundle, so a check that looks only
# in the bundle finds nothing, and if it is written as "check the markers when the
# file was found" it silently skips itself on every APK. That is exactly how builds
# with an unwired engine passed CI release after release.
ARCHIVE_ENTRIES: tuple[str, ...] = (
    "lib/arm64-v8a/libpybundle.so",
    "assets/private.tar",
)


def is_engine_source(name: str) -> bool:
    """Return whether *name* is a packaged copy of the engine module.

    The module is matched by directory and base name rather than by an exact
    path-and-extension string, because the packager may ship it compiled and
    interpreter-tagged, either beside the package (``momentum/llm/engine.pyc``) or
    under a ``__pycache__`` directory (``engine.cpython-311.pyc``). Every form
    carries the marker constants, and an over-narrow match would report a good APK
    as missing its engine -- a failure that stops a release, so it must only fire
    when the module is genuinely absent from both archives.
    """
    parts = tuple(part for part in PurePosixPath(name).parts if part != "__pycache__")
    return (
        len(parts) >= 3
        and parts[-3] == "momentum"
        and parts[-2] == "llm"
        and parts[-1].startswith("engine.")
        and PurePosixPath(parts[-1]).suffix in (".py", ".pyc")
    )


def engine_sources(apk: zipfile.ZipFile) -> list[bytes]:
    """Return every copy of the engine module shipped inside *apk*.

    Bytecode is matched as well as source: the APK ships ``momentum`` compiled,
    and the markers are string constants that survive into the ``.pyc``.
    """
    sources: list[bytes] = []
    for entry in ARCHIVE_ENTRIES:
        if entry not in apk.namelist():
            continue
        with tarfile.open(fileobj=io.BytesIO(apk.read(entry))) as archive:
            for member in archive.getmembers():
                if is_engine_source(member.name):
                    handle = archive.extractfile(member)
                    if handle is not None:
                        sources.append(handle.read())
    return sources


def archive_members(apk: zipfile.ZipFile) -> set[str]:
    """Return every file name inside every payload tarball the APK ships.

    Looking only in ``libpybundle.so`` means the app's own ``momentum`` package --
    which buildozer puts in ``assets/private.tar`` -- is invisible, so checks
    against it pass or fail for the wrong reason.
    """
    names: set[str] = set()
    for entry in ARCHIVE_ENTRIES:
        if entry not in apk.namelist():
            continue
        with tarfile.open(fileobj=io.BytesIO(apk.read(entry))) as archive:
            names.update(member.name for member in archive.getmembers())
    return names


def missing_cython_modules(members: Iterable[str]) -> list[str]:
    """Return compiled Cython modules that are absent from a bundle listing."""
    names = [Path(name).name for name in members]
    return [
        module
        for module in CYTHON_MODULES
        if not any(
            name.startswith(f"{module}.") and name.endswith(".so") for name in names
        )
    ]


def verify_apk(path: Path, vulkan: bool = False, require_cython: bool = False) -> None:
    with zipfile.ZipFile(path) as apk:
        names = set(apk.namelist())
        prefix = "lib/arm64-v8a/"
        libraries = ["llama", "ggml", "ggml-base", "ggml-cpu", "c++_shared"]
        if vulkan:
            libraries.append("ggml-vulkan")
        for library in libraries:
            name = f"{prefix}lib{library}.so"
            if name not in names:
                raise ValueError(f"Missing native library: {name}")
        members = archive_members(apk)
        if require_cython:
            missing = missing_cython_modules(members)
            if missing:
                raise ValueError(
                    "Missing compiled Cython module(s): "
                    + ", ".join(missing)
                    + " -- run mobile/scripts/build_android_ext.py before packaging"
                )
        package_lib_prefix = "site-packages/llama_cpp/lib/"
        for library in libraries:
            if library == "c++_shared":
                continue
            package_name = f"{package_lib_prefix}lib{library}.so"
            if not any(name.endswith(package_name) for name in members):
                raise ValueError(f"Missing package-side native library: {package_name}")
        # libllama.so's DT_NEEDED entries list bare sonames (libggml*.so and
        # libc++_shared.so) and it carries no DT_RUNPATH, so it only links when
        # loaded from the APK native dir -- the one location that also holds
        # libc++_shared.so, which the wheel never ships. Preferring the
        # package-side llama_cpp/lib therefore fails at runtime with
        # 'dlopen failed: library "libc++_shared.so" not found' even though every
        # file above is present. The engine module is what implements that
        # preference, so its absence anywhere in the APK is a hard failure: the
        # previous "check it if you find it" wording skipped the check entirely
        # on real APKs, where the engine lives in assets/private.tar.
        engines = engine_sources(apk)
        if not engines:
            raise ValueError(
                "engine module not found in "
                + " or ".join(ARCHIVE_ENTRIES)
                + "; the native-lib wiring check must never be skipped"
            )
        missing = [
            token.decode()
            for token in ENGINE_REQUIRED_MARKERS
            if not any(token in source for source in engines)
        ]
        if missing:
            raise ValueError(
                "engine.py is missing native-lib wiring: " + ", ".join(missing)
            )
        for module in [
            "llama_cpp/__init__",
            "numpy/__init__",
            "diskcache/__init__",
            "jinja2/__init__",
            "markupsafe/__init__",
            "markupsafe/_native",
            "typing_extensions",
        ]:
            if not any(
                n.endswith(f"site-packages/{module}{ext}")
                for n in members
                for ext in (".py", ".pyc")
            ):
                raise ValueError(f"Missing Python module: {module}")
    print(f"Coach packaging verified: {path.name} (runtime test still required)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("apk", type=Path)
    parser.add_argument("--vulkan", action="store_true")
    parser.add_argument(
        "--require-cython",
        action="store_true",
        help="fail when the compiled Cython modules are absent from the bundle",
    )
    args = parser.parse_args()
    verify_apk(args.apk, args.vulkan, args.require_cython)
