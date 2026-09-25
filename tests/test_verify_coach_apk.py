"""Tests for the APK packaging verifier's optional Cython check.

``missing_cython_modules`` backs ``verify_coach_apk.py --require-cython``, which
confirms that mobile/scripts/build_android_ext.py's output reached the bundle.
The check is opt-in, so the default verification path is unchanged.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from mobile.scripts.verify_coach_apk import (
    CYTHON_MODULES,
    missing_cython_modules,
    verify_apk,
)

_LIBRARIES = ["llama", "ggml", "ggml-base", "ggml-cpu", "c++_shared"]
_ENGINE_MARKERS = (
    "LLAMA_CPP_LIB_PATH",
    "llama_cpp",
    "package_root",
    "_preload_android_libraries",
)


def _bundle_members(include_cython: bool) -> list[str]:
    """Member names for a libpybundle tar that satisfies the base checks."""
    members = [
        "site-packages/llama_cpp/lib/libllama.so",
        "site-packages/llama_cpp/lib/libggml.so",
        "site-packages/llama_cpp/lib/libggml-base.so",
        "site-packages/llama_cpp/lib/libggml-cpu.so",
        "momentum/llm/engine.py",
    ]
    members += [
        f"site-packages/{name}/__init__.py"
        for name in ("llama_cpp", "diskcache", "jinja2", "markupsafe")
    ]
    members += [
        "site-packages/markupsafe/_native.py",
        "site-packages/typing_extensions.py",
    ]
    if include_cython:
        members += [
            f"momentum/{module}.cpython-311-aarch64-linux-android.so"
            for module in CYTHON_MODULES
        ]
    return members


def _write_apk(path: Path, include_cython: bool) -> Path:
    """Build a minimal APK holding the tar bundle the verifier inspects."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        for name in _bundle_members(include_cython):
            payload = " ".join(_ENGINE_MARKERS) if name.endswith("engine.py") else ""
            data = payload.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    with zipfile.ZipFile(path, "w") as apk:
        for library in _LIBRARIES:
            apk.writestr(f"lib/arm64-v8a/lib{library}.so", b"")
        apk.writestr("lib/arm64-v8a/libpybundle.so", buffer.getvalue())
    return path


@pytest.mark.parametrize(
    ("members", "expected"),
    [
        ([f"momentum/{module}.cpython-311.so" for module in CYTHON_MODULES], []),
        (
            [
                f"a/b/momentum/{module}.cpython-311-aarch64-linux-android.so"
                for module in CYTHON_MODULES
            ],
            [],
        ),
        (["momentum/_assessments_cy.cpython-311.so"], ["_charts_cy"]),
        ([], list(CYTHON_MODULES)),
    ],
)
def test_missing_cython_modules(members: list[str], expected: list[str]) -> None:
    assert missing_cython_modules(members) == expected


def test_verify_apk_ignores_cython_by_default(tmp_path: Path) -> None:
    """The default verification path must not start requiring the extensions."""
    verify_apk(_write_apk(tmp_path / "momentum.apk", include_cython=False))


def test_verify_apk_require_cython_flags_a_missing_extension(tmp_path: Path) -> None:
    apk = _write_apk(tmp_path / "momentum.apk", include_cython=False)
    with pytest.raises(ValueError, match="Missing compiled Cython module"):
        verify_apk(apk, require_cython=True)


def test_verify_apk_require_cython_accepts_a_bundle_with_extensions(
    tmp_path: Path,
) -> None:
    verify_apk(
        _write_apk(tmp_path / "momentum.apk", include_cython=True), require_cython=True
    )
