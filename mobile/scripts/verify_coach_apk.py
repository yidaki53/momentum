"""Check required coach payloads without executing Android machine code."""

from __future__ import annotations

import argparse
import io
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable

# Compiled Cython modules built by mobile/scripts/build_android_ext.py. Their
# absence is not fatal -- momentum falls back to pure Python -- so the check below
# is opt-in via --require-cython.
CYTHON_MODULES = ("_assessments_cy", "_charts_cy")


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
        with tarfile.open(fileobj=io.BytesIO(apk.read(prefix + "libpybundle.so"))) as bundle:
            members = set(bundle.getnames())
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
            # The engine must prefer the complete package-side llama_cpp/lib
            # directory for both llama_cpp.py and _ggml.py.  Merely finding the
            # APK's lib/<abi> copies is insufficient: _ggml.py loads ggml from
            # the package directory and otherwise reports "engine unavailable".
            engine_srcs = [
                n for n in members
                if n.endswith("momentum/llm/engine.py")
                or n.endswith("momentum/llm/engine.pyc")
            ]
            if engine_srcs:
                extracted = bundle.extractfile(engine_srcs[0])
                raw = extracted.read() if extracted else b""
                if (
                    b"LLAMA_CPP_LIB_PATH" not in raw
                    or b"llama_cpp" not in raw
                    or b"package_root" not in raw
                    or b"_preload_android_libraries" not in raw
                ):
                    raise ValueError(
                        "engine.py is missing the package-side native-lib wiring"
                    )
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
    parser.add_argument(
        "--require-cython",
        action="store_true",
        help="fail when the compiled Cython modules are absent from the bundle",
    )
    args = parser.parse_args()
    verify_apk(args.apk, args.vulkan, args.require_cython)
