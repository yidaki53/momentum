#!/usr/bin/env python3
"""Measure what a candidate teacher model costs in heat, on this machine.

The teacher size trade-off is not obvious: a bigger model is a better teacher, but on a
laptop the CPU package sensor sits on the same cooling path as the GPU, so heavier
generation heats the whole machine. That trade was worth measuring rather than
guessing, especially after two thermal crashes.

This runs the SAME number of generations against each candidate and reports peak and
mean temperatures, VRAM, and wall-clock per pair. The numbers are directly comparable
because the workload is identical.

It samples temperatures itself rather than relying on the thermal guard, so the guard
can be run *around* this with a generous tolerance: the point is to observe the heat,
not to be killed by it. Keep --pairs small anyway. This machine has crashed.

Usage::

    python scripts/compare_teacher_heat.py --models 1.5B 3B --pairs 15
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
}


@dataclass
class Trace:
    """A temperature series around one model's run."""

    label: str
    samples: list[tuple[float, dict[str, float]]] = field(default_factory=list)

    def peak(self, key: str) -> float:
        values = [values[key] for _, values in self.samples if key in values]
        return max(values) if values else float("nan")

    def mean(self, key: str) -> float:
        values = [values[key] for _, values in self.samples if key in values]
        return statistics.fmean(values) if values else float("nan")


def sample_temps() -> dict[str, float]:
    import importlib.util
    import sys

    guard_path = Path(__file__).with_name("thermal_guard.py")
    spec = importlib.util.spec_from_file_location("thermal_guard_mod", guard_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["thermal_guard_mod"] = module
    spec.loader.exec_module(module)
    out: dict[str, float] = {}
    for reading in module.sample():
        out[reading.key] = max(out.get(reading.key, 0.0), reading.celsius)
    return out


def run_one(
    label: str, model_id: str, *, pairs: int, four_bit: bool, max_new_tokens: int = 120
) -> dict[str, Any]:
    """Generate *pairs* replies with one teacher, sampling heat throughout."""
    import importlib.util
    import sys
    import threading

    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "build_pairs_mod", here / "build_distill_pairs.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_pairs_mod"] = module
    spec.loader.exec_module(module)

    trace = Trace(label=label)
    sampling = True

    def sampler() -> None:
        while sampling:
            trace.samples.append((time.time(), sample_temps()))
            time.sleep(2.0)

    # Baseline before anything is loaded, so the reported rise is real.
    trace.samples.append((time.time(), sample_temps()))
    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()

    start = time.time()
    teacher = module.Teacher(model_id, load_in_4bit=four_bit)
    teacher._load()
    load_seconds = time.time() - start

    import torch

    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    situations = list(module.SITUATIONS[:pairs])
    prompts = [
        module.build_teacher_prompt(
            situation,
            "Give exactly one first action that takes under two minutes.",
            module.USER_POOL[0].digest(),
            "task decomposition into a sub-two-minute first action",
        )
        for situation in situations
    ]

    gen_start = time.time()
    accepted = 0
    for prompt in prompts:
        reply = teacher.generate(prompt, max_new_tokens=max_new_tokens)
        if module.quality_gate(
            reply,
            constraint="Give exactly one first action that takes under two minutes.",
        )[0]:
            accepted += 1
    generate_seconds = time.time() - gen_start

    sampling = False
    trace.samples.append((time.time(), sample_temps()))

    return {
        "model": label,
        "load_s": round(load_seconds, 1),
        "generate_s": round(generate_seconds, 1),
        "per_pair_s": round(generate_seconds / max(1, len(prompts)), 2),
        "vram_gb": round(vram, 1),
        "accepted": f"{accepted}/{len(prompts)}",
        "peak_cpu": round(trace.peak("cpu_package"), 1),
        "mean_cpu": round(trace.mean("cpu_package"), 1),
        "peak_gpu": round(trace.peak("gpu"), 1),
        "mean_gpu": round(trace.mean("gpu"), 1),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["1.5B"],
        choices=sorted(MODELS),
        help="sizes to compare",
    )
    parser.add_argument(
        "--pairs", type=int, default=15, help="generations per model (keep small)"
    )
    parser.add_argument(
        "--four-bit", nargs="*", default=[], help="model sizes to load in 4-bit"
    )
    parser.add_argument("--out", default="out/teacher-heat.json")
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for label in args.models:
        print(f"\n=== {label} ({MODELS[label]}) ===", flush=True)
        row = run_one(
            label, MODELS[label], pairs=args.pairs, four_bit=label in args.four_bit
        )
        rows.append(row)
        print(json.dumps(row), flush=True)

    print("\n" + "=" * 78)
    header = f"{'model':<7}{'per pair':>10}{'VRAM':>8}{'CPU peak':>10}{'CPU mean':>10}{'GPU peak':>10}{'gate':>10}"
    print(header)
    print("-" * 78)
    for row in rows:
        print(
            f"{row['model']:<7}{row['per_pair_s']:>9.2f}s{row['vram_gb']:>7.1f}G"
            f"{row['peak_cpu']:>9.0f}C{row['mean_cpu']:>9.0f}C"
            f"{row['peak_gpu']:>9.0f}C{row['accepted']:>10}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
