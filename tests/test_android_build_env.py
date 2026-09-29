from __future__ import annotations

import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from mobile import p4a_hooks
from mobile.scripts import build_android as build


def test_bundle_version_tracks_contents_not_archive_timestamp(tmp_path):
    def write_bundle(path: Path, content: bytes, timestamp: int) -> None:
        with tarfile.open(path, "w:gz") as tar:
            entry = tarfile.TarInfo("_python_bundle/site-packages/example.pyc")
            entry.size = len(content)
            entry.mtime = timestamp
            tar.addfile(entry, io.BytesIO(content))

    first = tmp_path / "first.so"
    rebuilt = tmp_path / "rebuilt.so"
    changed = tmp_path / "changed.so"
    write_bundle(first, b"same module", 100)
    write_bundle(rebuilt, b"same module", 200)
    write_bundle(changed, b"changed module", 200)

    first_version, entries = p4a_hooks._archive_contents(first)
    assert entries == 1
    assert p4a_hooks._archive_contents(rebuilt) == (first_version, entries)
    assert p4a_hooks._archive_contents(changed)[0] != first_version


def test_after_apk_build_uses_current_distribution(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET

    monkeypatch.chdir(tmp_path)
    resources = tmp_path / "src/main/res/values/strings.xml"
    resources.parent.mkdir(parents=True)
    resources.write_text(
        '<resources><string name="private_version">old</string></resources>'
    )
    for archive in (
        tmp_path / "src/main/assets/private.tar",
        tmp_path / "libs/arm64-v8a/libpybundle.so",
    ):
        archive.parent.mkdir(parents=True)
        with tarfile.open(archive, "w:gz") as tar:
            entry = tarfile.TarInfo("example")
            entry.size = 4
            tar.addfile(entry, io.BytesIO(b"data"))
    monkeypatch.setattr(
        p4a_hooks.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    p4a_hooks.after_apk_build(
        SimpleNamespace(ctx=SimpleNamespace(dist_dir=tmp_path.parent))
    )

    root = ET.parse(resources).getroot()
    assert root.find("string[@name='pybundle_version']") is not None
    assert root.find("integer[@name='pybundle_entries']").text == "1"


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
