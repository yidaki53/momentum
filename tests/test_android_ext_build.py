"""Tests for the Android cross-compile helper for momentum's Cython modules.

``mobile/scripts/build_android_ext.py`` runs on the CI runner between
buildozer's two passes and cannot be exercised end-to-end here, so these tests
cover the parts that decide what gets compiled: locating python-for-android's
target CPython, reading the target ``EXT_SUFFIX``, finding the NDK driver, and
the shape of the clang command line.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from mobile.scripts import build_android_ext as ext


def _fake_target(
    build_dir: Path, arch: str = "arm64-v8a", suffix: str | None = None
) -> Path:
    """Create a python-for-android-shaped target CPython install."""
    target = build_dir / "python-installs" / "momentum" / arch
    headers = target / "include" / "python3.11"
    headers.mkdir(parents=True)
    (headers / "Python.h").touch()
    lib_dir = target / "lib" / "python3.11"
    lib_dir.mkdir(parents=True)
    if suffix is not None:
        (lib_dir / "_sysconfigdata__linux_aarch64-linux-android.py").write_text(
            f"build_time_vars = {{\n    'EXT_SUFFIX': '{suffix}',\n}}\n",
            encoding="utf-8",
        )
    return target


def _fake_ndk(root: Path, apis: tuple[int, ...] = (26,)) -> Path:
    """Create an NDK toolchain tree containing the given clang drivers."""
    ndk = root / "android-ndk-r25b"
    binaries = ndk / "toolchains" / "llvm" / "prebuilt" / "linux-x86_64" / "bin"
    binaries.mkdir(parents=True)
    for api in apis:
        (binaries / f"aarch64-linux-android{api}-clang").touch()
    return ndk


def test_target_python_found_from_p4a_layout(tmp_path: Path) -> None:
    target = _fake_target(tmp_path)
    assert ext.find_target_python(tmp_path, "arm64-v8a") == target


def test_target_python_ignores_directories_without_a_python(tmp_path: Path) -> None:
    (tmp_path / "python-installs" / "momentum" / "arm64-v8a").mkdir(parents=True)
    assert ext.find_target_python(tmp_path, "arm64-v8a") is None


def test_target_python_missing_returns_none(tmp_path: Path) -> None:
    assert ext.find_target_python(tmp_path, "arm64-v8a") is None


def test_ext_suffix_read_from_target_sysconfigdata(tmp_path: Path) -> None:
    target = _fake_target(tmp_path, suffix=".cpython-311-aarch64-linux-android.so")
    assert ext.read_ext_suffix(target) == ".cpython-311-aarch64-linux-android.so"


def test_ext_suffix_falls_back_without_sysconfigdata(tmp_path: Path) -> None:
    target = _fake_target(tmp_path)
    assert ext.read_ext_suffix(target) == ext.FALLBACK_EXT_SUFFIX


@pytest.mark.parametrize(
    ("apis", "min_api", "expected"),
    [
        ((26,), 26, "aarch64-linux-android26-clang"),
        ((21,), 26, "aarch64-linux-android21-clang"),
    ],
)
def test_ndk_clang_selected_for_min_api(
    tmp_path: Path, apis: tuple[int, ...], min_api: int, expected: str
) -> None:
    ndk = _fake_ndk(tmp_path, apis)
    found = ext.find_ndk_clang(min_api, env={"ANDROID_NDK_HOME": str(ndk)})
    assert found is not None
    assert found.name == expected


def test_ndk_clang_absent_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert ext.find_ndk_clang(26, env={}) is None


def test_build_command_links_target_libpython(tmp_path: Path) -> None:
    target = _fake_target(tmp_path)
    (target / "libpython3.11.so").touch()
    include_dir = target / "include" / "python3.11"
    command = ext.build_command(
        clang=Path("/ndk/clang"),
        c_source=Path("/src/assessments_cy.c"),
        out_path=Path("/out/_assessments_cy.so"),
        include_dirs=[include_dir],
        target_dir=target,
    )
    assert command[0] == "/ndk/clang"
    assert "-shared" in command and "-fPIC" in command
    assert f"-I{include_dir}" in command
    assert f"-L{target}" in command and "-lpython3.11" in command
    assert command[command.index("-o") + 1] == "/out/_assessments_cy.so"
    assert command[-1] == "/src/assessments_cy.c"


def test_build_all_is_non_fatal_without_ndk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _fake_target(tmp_path)
    monkeypatch.setattr(ext, "find_ndk_clang", lambda *a, **k: None)
    assert ext.build_all(tmp_path, "arm64-v8a", 26) == 1
    assert "skipping" in capsys.readouterr().err


def test_build_all_is_non_fatal_without_target_python(tmp_path: Path, capsys) -> None:
    assert ext.build_all(tmp_path, "arm64-v8a", 26) == 1
    assert "skipping" in capsys.readouterr().err


def test_build_all_compiles_every_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_target(tmp_path, suffix=".cpython-311.so")
    package = tmp_path / "momentum"
    package.mkdir()
    for pyx_name, _module in ext.EXTENSIONS:
        (package / f"{Path(pyx_name).stem}.c").touch()

    commands: list[list[str]] = []

    def record(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ext, "find_ndk_clang", lambda *a, **k: Path("/ndk/clang"))
    monkeypatch.setattr(ext.subprocess, "run", record)

    assert ext.build_all(tmp_path, "arm64-v8a", 26, root=tmp_path) == 0
    assert len(commands) == len(ext.EXTENSIONS)
    built = {Path(command[command.index("-o") + 1]).name for command in commands}
    assert built == {f"{module}.cpython-311.so" for _pyx, module in ext.EXTENSIONS}


def test_build_all_dry_run_does_not_invoke_the_compiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_target(tmp_path)
    package = tmp_path / "momentum"
    package.mkdir()
    for pyx_name, _module in ext.EXTENSIONS:
        (package / f"{Path(pyx_name).stem}.c").touch()

    def explode(*args, **kwargs):
        raise AssertionError("dry run must not invoke the compiler")

    monkeypatch.setattr(ext, "find_ndk_clang", lambda *a, **k: Path("/ndk/clang"))
    monkeypatch.setattr(ext.subprocess, "run", explode)
    assert ext.build_all(tmp_path, "arm64-v8a", 26, dry_run=True, root=tmp_path) == 0


def test_extensions_match_setup_py() -> None:
    """Adding an extension to setup.py must also add it to the Android build."""
    setup_py = (ext.ROOT / "setup.py").read_text(encoding="utf-8")
    declared = set(re.findall(r'name="momentum\.(\w+)"', setup_py))
    assert declared == {module for _pyx, module in ext.EXTENSIONS}


def test_cython_generated_sources_are_committed() -> None:
    """The script compiles the committed .c files, so they must exist."""
    for pyx_name, _module in ext.EXTENSIONS:
        c_source = ext.ROOT / "momentum" / f"{Path(pyx_name).stem}.c"
        assert c_source.is_file(), f"{c_source.name} must be committed"


def test_buildozer_spec_bundles_shared_objects() -> None:
    """The compiled .so files must survive buildozer's source filter."""
    spec = (ext.ROOT / "mobile" / "buildozer.spec").read_text(encoding="utf-8")
    line = next(
        text for text in spec.splitlines() if text.startswith("source.include_exts")
    )
    extensions = [item.strip() for item in line.split("=", 1)[1].split(",")]
    assert "so" in extensions
