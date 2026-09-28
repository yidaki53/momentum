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
    ENGINE_REQUIRED_MARKERS,
    missing_cython_modules,
    verify_apk,
)

_LIBRARIES = ["llama", "ggml", "ggml-base", "ggml-cpu", "c++_shared"]
# Derived from the verifier's own contract so this fixture can never drift from
# the real requirement (it previously duplicated the list and silently did).
_ENGINE_MARKERS = tuple(marker.decode() for marker in ENGINE_REQUIRED_MARKERS)


# buildozer splits the payload in two: third-party packages go into
# ``libpybundle.so`` while the app's own sources go into ``assets/private.tar``.
# A fixture that ignores that split validates a packaging layout no real APK has
# ever had -- which is precisely how the engine check stopped checking anything.
_BUNDLE_MEMBERS = [
    "site-packages/llama_cpp/lib/libllama.so",
    "site-packages/llama_cpp/lib/libggml.so",
    "site-packages/llama_cpp/lib/libggml-base.so",
    "site-packages/llama_cpp/lib/libggml-cpu.so",
    "site-packages/llama_cpp/__init__.py",
    "site-packages/diskcache/__init__.py",
    "site-packages/jinja2/__init__.py",
    "site-packages/markupsafe/__init__.py",
    "site-packages/markupsafe/_native.py",
    "site-packages/typing_extensions.py",
]


def _tar(entries: list[tuple[str, bytes]]) -> bytes:
    """Pack ``name -> contents`` pairs the way buildozer packs its payload tars."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _write_apk(
    path: Path,
    include_cython: bool,
    engine_payload: "str | None" = None,
    engine_location: str = "private",
    engine_name: str = "momentum/llm/engine.pyc",
) -> Path:
    """Build a minimal APK carrying the same two payload archives as a real one.

    ``engine_location`` selects where the engine module is shipped: ``private``
    matches today's APKs, ``bundle`` the layout older verifier fixtures assumed,
    ``none`` an APK that ships no engine at all -- the case the verifier used to
    wave through. ``engine_name`` is the member name it is shipped under, which
    varies with how the packager compiles the package.
    """
    if engine_location not in ("private", "bundle", "none"):  # pragma: no cover
        raise ValueError(f"unknown engine_location {engine_location!r}")
    engine_data = (
        " ".join(_ENGINE_MARKERS) if engine_payload is None else engine_payload
    ).encode()
    engine: tuple[str, bytes] = (engine_name, engine_data)

    private: list[tuple[str, bytes]] = []
    if engine_location == "private":
        private.append(engine)
    if include_cython:
        private += [
            (f"momentum/{module}.cpython-311-aarch64-linux-android.so", b"\x7fELF")
            for module in CYTHON_MODULES
        ]
    bundle: list[tuple[str, bytes]] = [(name, b"") for name in _BUNDLE_MEMBERS]
    if engine_location == "bundle":
        bundle.append(engine)

    with zipfile.ZipFile(path, "w") as apk:
        for library in _LIBRARIES:
            apk.writestr(f"lib/arm64-v8a/lib{library}.so", b"")
        apk.writestr("lib/arm64-v8a/libpybundle.so", _tar(bundle))
        apk.writestr("assets/private.tar", _tar(private))
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


def test_verify_apk_flags_an_engine_that_never_preloads_the_stl(
    tmp_path: Path,
) -> None:
    """Guard the regression that left the coach unavailable on-device.

    Packaging can be perfect -- every ``.so`` present in the APK -- and the
    import still fails, because ``libllama.so`` cannot pull ``libc++_shared.so``
    in from the wheel directory. An engine.py that drops the STL preload must be
    rejected even though all the libraries above are where they belong.
    """
    stripped = " ".join(marker for marker in _ENGINE_MARKERS if marker != "c++_shared")
    apk = _write_apk(
        tmp_path / "momentum.apk", include_cython=False, engine_payload=stripped
    )
    with pytest.raises(ValueError, match="missing native-lib wiring"):
        verify_apk(apk)


def test_verify_apk_accepts_the_layout_real_apks_ship(tmp_path: Path) -> None:
    """The engine is shipped in ``assets/private.tar``, not in the Python bundle.

    Released APKs contain no ``momentum`` entry inside ``libpybundle.so``, so a
    verifier that reads only the bundle finds no engine and -- when the check is
    written as "verify the markers if the file was found" -- reports success for
    every build regardless of what it contains.
    """
    verify_apk(_write_apk(tmp_path / "momentum.apk", include_cython=False))


def test_verify_apk_flags_an_apk_with_no_engine_anywhere(tmp_path: Path) -> None:
    """A missing engine must fail the build, not silently skip the check."""
    apk = _write_apk(
        tmp_path / "momentum.apk", include_cython=False, engine_location="none"
    )
    with pytest.raises(ValueError, match="engine module not found"):
        verify_apk(apk)


def test_verify_apk_still_accepts_an_engine_in_the_bundle(tmp_path: Path) -> None:
    """The older layout stays accepted so the check is about content, not place."""
    verify_apk(
        _write_apk(
            tmp_path / "momentum.apk", include_cython=False, engine_location="bundle"
        )
    )


@pytest.mark.parametrize(
    "engine_name",
    [
        "momentum/llm/engine.py",
        "momentum/llm/engine.pyc",
        "momentum/llm/__pycache__/engine.cpython-311.pyc",
        "python3.11/site-packages/momentum/llm/__pycache__/engine.cpython-311.pyc",
    ],
)
def test_verify_apk_finds_the_engine_in_any_shipped_form(
    tmp_path: Path, engine_name: str
) -> None:
    """Tagged bytecode under ``__pycache__`` is still the engine module.

    Matching one exact file name would report a correctly packaged APK as
    engine-less, and because that finding is fatal it would stop a good release.
    The markers are string constants that survive compilation, so every one of
    these spellings carries the same evidence.
    """
    verify_apk(
        _write_apk(
            tmp_path / "momentum.apk", include_cython=False, engine_name=engine_name
        )
    )


@pytest.mark.parametrize(
    "engine_name",
    [
        "momentum/llm/engine_config.pyc",
        "momentum/llm/engine.so",
        "tools/llm/engine.pyc",
    ],
)
def test_verify_apk_does_not_mistake_another_file_for_the_engine(
    tmp_path: Path, engine_name: str
) -> None:
    """The matcher must stay narrow enough to mean something.

    Accepting any file whose name merely starts with ``engine`` would let an APK
    with no engine at all pass, which is the failure this check exists to catch.
    """
    apk = _write_apk(
        tmp_path / "momentum.apk",
        include_cython=False,
        engine_name=engine_name,
    )
    with pytest.raises(ValueError, match="engine module not found"):
        verify_apk(apk)
