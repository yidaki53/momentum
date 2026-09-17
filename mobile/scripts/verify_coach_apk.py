"""Check required coach payloads without executing Android machine code."""

from __future__ import annotations

import argparse
import io
import tarfile
import zipfile
from pathlib import Path


def verify_apk(path: Path, vulkan: bool = False) -> None:
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
        with tarfile.open(fileobj=io.BytesIO(apk.read(prefix + "libpybundle.so"))) as bundle:
            members = set(bundle.getnames())
        for module in ["llama_cpp/__init__", "diskcache/__init__", "jinja2/__init__",
                       "markupsafe/__init__", "markupsafe/_native", "typing_extensions"]:
            if not any(n.endswith(f"site-packages/{module}{ext}")
                       for n in members for ext in (".py", ".pyc")):
                raise ValueError(f"Missing Python module: {module}")
    print(f"Coach packaging verified: {path.name} (runtime test still required)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("apk", type=Path)
    parser.add_argument("--vulkan", action="store_true")
    args = parser.parse_args()
    verify_apk(args.apk, args.vulkan)
