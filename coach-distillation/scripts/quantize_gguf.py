#!/usr/bin/env python3
"""Quantise the merged (or abliterated) student into GGUF, one file per candidate type.

llama.cpp is not a Python dependency of Momentum and not one of this pipeline's either:
``convert_hf_to_gguf.py`` and ``llama-quantize`` are run as subprocesses against a
pinned checkout. Quant formats change between llama.cpp releases, so the ref is pinned
below rather than tracking ``main`` -- a quantisation you cannot reproduce is not an
artefact you can ship.

The candidate list exists to answer a question with numbers rather than hope:
specialisation is supposed to buy headroom, so the smallest quantisation that still
clears the bench is the one to ship. Run ``eval_bench.py`` against each output.

Usage::

    python scripts/quantize_gguf.py --model out/student-merged
    python scripts/quantize_gguf.py --model out/student-abliterated --types Q4_K_M
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("quantize_gguf")

ROOT = Path(__file__).resolve().parent.parent
# Pinned llama.cpp revision. Bump deliberately, with a re-run of the bench.
LLAMA_CPP_REF = "b6100"
DEFAULT_LLAMA_CPP = ROOT / "llama.cpp"


def load_config(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as handle:
        return tomllib.load(handle)


def find_llama_cpp(explicit: Optional[Path]) -> Path:
    """Locate a llama.cpp checkout with both binaries present."""
    candidates = [explicit] if explicit else []
    candidates.append(DEFAULT_LLAMA_CPP)
    candidates.append(Path(os.environ.get("LLAMA_CPP_DIR", DEFAULT_LLAMA_CPP)))
    for candidate in candidates:
        if not candidate:
            continue
        if (candidate / "convert_hf_to_gguf.py").exists():
            return candidate
    raise RuntimeError(
        "No llama.cpp checkout found. Clone it inside this folder and check out "
        f"{LLAMA_CPP_REF}:\n"
        f"  git clone https://github.com/ggml-org/llama.cpp {DEFAULT_LLAMA_CPP}\n"
        f"  git -C {DEFAULT_LLAMA_CPP} checkout {LLAMA_CPP_REF}\n"
        "then build llama-quantize (cmake -B build && cmake --build build --target "
        "llama-quantize)"
    )


def find_quantize_binary(repo: Path) -> Path:
    """Return the llama-quantize binary, preferring the built one."""
    for candidate in (
        repo / "build" / "bin" / "llama-quantize",
        repo / "build" / "llama-quantize",
        repo / "llama-quantize",
    ):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        f"llama-quantize not found under {repo}. Build it with:\n"
        "  cmake -B build && cmake --build build --target llama-quantize"
    )


def convert(repo: Path, model: Path, output_dir: Path) -> Path:
    """Convert HF safetensors to an F16 GGUF. Quantisation happens next."""
    output_dir.mkdir(parents=True, exist_ok=True)
    f16 = output_dir / f"{model.name}-f16.gguf"
    command = [
        sys.executable,
        str(repo / "convert_hf_to_gguf.py"),
        "--outfile",
        str(f16),
        "--outtype",
        "f16",
        str(model),
    ]
    log.info("converting: %s", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0 or not f16.exists():
        raise RuntimeError("convert_hf_to_gguf.py failed")
    return f16


def quantize(
    binary: Path, f16: Path, out_type: str, output_dir: Path
) -> Optional[Path]:
    """Produce one quantised GGUF, tolerating formats llama.cpp does not build."""
    target = output_dir / f"{f16.name.split('-f16')[0]}-{out_type}.gguf"
    command = [str(binary), str(f16), str(target), out_type]
    log.info("quantising to %s", out_type)
    result = subprocess.run(command, check=False)
    if result.returncode != 0 or not target.exists():
        # IQ2_XXS and friends need a llama.cpp built with the right k-quants; report and
        # continue rather than aborting the remaining candidates.
        log.warning("quantisation to %s failed (exit %s)", out_type, result.returncode)
        return None
    size_mb = target.stat().st_size / (1024 * 1024)
    log.info("wrote %s (%.0f MB)", target, size_mb)
    return target


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distill.qwen05b.toml")
    parser.add_argument(
        "--model",
        default="out/student-merged",
        help="HF checkpoint (merged, or merged+abliterated)",
    )
    parser.add_argument(
        "--types",
        default="",
        help="comma-separated quant types (default: config candidates)",
    )
    parser.add_argument("--llama-cpp", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(Path(args.config))
    quant_cfg = config.get("quantize", {})
    types = [t.strip() for t in args.types.split(",") if t.strip()] or list(
        quant_cfg.get("candidates", ["Q4_K_M"])
    )
    output_dir = Path(args.output_dir or quant_cfg.get("output_dir", "out/gguf"))
    model = Path(args.model)

    if not model.exists():
        print(f"No checkpoint at {model}. Run train_distill.py first.", file=sys.stderr)
        return 1

    try:
        repo = find_llama_cpp(Path(args.llama_cpp) if args.llama_cpp else None)
        binary = find_quantize_binary(repo)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    f16 = convert(repo, model, output_dir)
    written: list[dict[str, Any]] = []
    for out_type in types:
        path = quantize(binary, f16, out_type, output_dir)
        if path is None:
            continue
        written.append(
            {
                "type": out_type,
                "path": str(path),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
            }
        )

    manifest = output_dir / "quantization.json"
    manifest.write_text(
        json.dumps({"model": str(model), "outputs": written}, indent=2) + "\n",
        encoding="utf-8",
    )
    if not written:
        print("No quantisation succeeded.", file=sys.stderr)
        return 1
    for entry in written:
        print(f"{entry['type']:>10}  {entry['size_mb']:>6.0f} MB  {entry['path']}")
    print(f"\nnext: python scripts/eval_bench.py --model {written[0]['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
