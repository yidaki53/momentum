from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mobile.scripts import build_android as build


def test_environment_repairs_inactive_venv_and_java8(tmp_path, monkeypatch):
    root = tmp_path / "project with spaces"
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/buildozer").touch()
    jdk = tmp_path / "jdk17"
    (jdk / "bin").mkdir(parents=True)
    (jdk / "bin/javac").touch()
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "jdk8"))
    monkeypatch.setenv("PYTHONHOME", "incorrect")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(build.shutil, "which", lambda name: str(jdk / "bin/javac"))
    monkeypatch.setattr(build, "java_major", lambda home: 17 if home == jdk else 8)
    env = build.build_environment(root)
    assert env["JAVA_HOME"] == str(jdk)
    assert env["VIRTUAL_ENV"] == str(root / ".venv")
    assert "PYTHONHOME" not in env
    assert "%20" in env["PIP_FIND_LINKS"]
    assert env["PATH"].startswith(str(root / ".venv/bin"))


@pytest.mark.parametrize(
    ("version", "expected"), [("1.8.0", 8), ("17.0.20", 17), ("21.0.1", 21)]
)
def test_java_version_parsing(monkeypatch, version, expected):
    monkeypatch.setattr(
        build.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stderr=f'openjdk version "{version}"', stdout=""
        ),
    )
    assert build.java_major(Path("/fake")) == expected


def test_aab_failure_restores_spec(tmp_path, monkeypatch):
    (tmp_path / "mobile").mkdir()
    spec = tmp_path / "mobile/buildozer.spec"
    original = b"android.release_artifact = apk\n"
    spec.write_bytes(original)
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(
        build,
        "build_environment",
        lambda: {key: "test" for key in ("JAVA_HOME", "VIRTUAL_ENV", "PIP_FIND_LINKS")},
    )
    monkeypatch.setattr("sys.argv", ["build_android.py", "--aab"])

    def fail(*args, **kwargs):
        assert "= aab" in spec.read_text()
        raise OSError("build failed")

    monkeypatch.setattr(build.subprocess, "run", fail)
    with pytest.raises(OSError):
        build.main()
    assert spec.read_bytes() == original
