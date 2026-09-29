"""Build-time pruning for the Android Python bundle."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

_REMOVE_DIR_NAMES = {
    "__pycache__",
    "tests",
    "testing",
    "test",
    "benchmarks",
    "docs",
    "doc",
    "examples",
    "example",
}
_REMOVE_SUFFIXES = {".pyi", ".md", ".rst"}


def _prune_tree(root: Path) -> None:
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and path.name in _REMOVE_DIR_NAMES:
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file() and path.suffix in _REMOVE_SUFFIXES:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def before_apk_build(toolchain) -> None:
    """Remove development-only payload before p4a copies the bundle into APK assets."""
    distribution_root = Path(toolchain.ctx.dist_dir)
    site_package_dirs = list(distribution_root.rglob("site-packages"))
    for site_packages in site_package_dirs:
        for package_name in ("matplotlib", "numpy", "PIL", "fontTools", "kivy"):
            _prune_tree(site_packages / package_name)

    for modules_dir in distribution_root.rglob("_python_bundle/_python_bundle/modules"):
        for path in modules_dir.iterdir():
            if path.name.startswith(("_test", "xx")):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)

    print("[p4a-hooks] pruned development-only scientific Python payload")


def _archive_contents(archive: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    entries = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            entries += 1
            digest.update(member.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(member.size).encode("ascii"))
            digest.update(b"\0")
            if member.isfile():
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError(f"Missing archive entry: {member.name}")
                with stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
    return digest.hexdigest(), entries


def after_apk_build(toolchain) -> None:
    """Version each unpacked archive by its contents, independent of build time."""
    distribution_root = Path.cwd()
    resources = distribution_root / "src/main/res/values/strings.xml"
    tree = ET.parse(resources)
    root = tree.getroot()
    archives = {
        "private": distribution_root / "src/main/assets/private.tar",
        "pybundle": distribution_root / "libs/arm64-v8a/libpybundle.so",
    }
    for name, archive in archives.items():
        version, count = _archive_contents(archive)
        version_key = "private_version" if name == "private" else "pybundle_version"
        element = root.find(f"string[@name='{version_key}']")
        if element is None:
            element = ET.SubElement(root, "string", {"name": version_key})
        element.text = version
        count_key = f"{name}_entries"
        count_element = root.find(f"integer[@name='{count_key}']")
        if count_element is None:
            count_element = ET.SubElement(root, "integer", {"name": count_key})
        count_element.text = str(count)
    tree.write(resources, encoding="utf-8", xml_declaration=True)

    patch_file = Path(__file__).resolve().parent / "patches/p4a-extraction-progress.patch"
    dry_run = subprocess.run(
        ["patch", "-p1", "--batch", "--forward", "--dry-run", "-i", str(patch_file)],
        cwd=distribution_root,
        capture_output=True,
        text=True,
    )
    if dry_run.returncode == 0:
        subprocess.run(
            ["patch", "-p1", "--batch", "--forward", "-i", str(patch_file)],
            cwd=distribution_root,
            check=True,
        )
    else:
        reverse = subprocess.run(
            ["patch", "-p1", "--batch", "--reverse", "--dry-run", "-i", str(patch_file)],
            cwd=distribution_root,
            capture_output=True,
            text=True,
        )
        if reverse.returncode != 0:
            raise RuntimeError(f"Cannot apply Android extraction patch: {dry_run.stdout} {dry_run.stderr}")
