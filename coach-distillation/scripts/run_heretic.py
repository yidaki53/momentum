#!/usr/bin/env python3
"""Run Heretic abliteration against the merged student, as a subprocess.

Heretic is AGPL-3.0 and this repository is MIT, so it lives in ``heretic/`` as a pinned
submodule and is *invoked*, never imported. Copying its code into ``momentum/`` would
place an AGPL obligation on the whole app; a subprocess call does not.

Why run it at all: Heretic removes the base model's generic refusal direction so that
``docs/CHARTER.md`` governs behaviour instead. It is optional. If the supervised model
already refuses rarely enough to clear the charter axis, skip this step -- abliterating a
model that does not need it only risks capability.

How it is driven: Heretic has **no CLI flags**. It reads ``config.toml`` from its own
working directory via pydantic-settings, and it *prompts interactively* for the export
strategy and save directory unless ``export_strategy`` is already set. So this script
materialises a scratch working directory, writes a resolved ``config.toml`` into it, and
runs there.

Usage::

    python scripts/run_heretic.py --check-only
    python scripts/run_heretic.py --model out/student-merged
    python scripts/run_heretic.py --model out/student-merged --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("run_heretic")

ROOT = Path(__file__).resolve().parent.parent
HERETIC_DIR = ROOT / "heretic"
ENTRYPOINT = HERETIC_DIR / "src" / "heretic" / "main.py"


def check_submodule() -> Path:
    """Return the Heretic checkout, or explain why it cannot be used."""
    if not HERETIC_DIR.exists():
        raise RuntimeError(
            f"Heretic submodule missing at {HERETIC_DIR}. Run:\n"
            "  git submodule update --init --recursive"
        )
    if not (HERETIC_DIR / "src" / "heretic" / "main.py").exists():
        raise RuntimeError(
            f"{HERETIC_DIR} exists but is not a Heretic checkout "
            "(src/heretic/main.py missing)."
        )
    return HERETIC_DIR


def load_toml(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as handle:
        return tomllib.load(handle)


def inject_keys(text: str, scalars: dict[str, Any]) -> str:
    """Return *text* with *scalars* inserted before the first table header.

    The config is hand-maintained TOML and is far too nested to re-emit from a parsed
    dict without a TOML writer dependency (a naive writer turns
    ``[scorer.KeywordRate]`` into a stringified Python dict). Prepending scalar keys is
    both simpler and lossless, and it is valid TOML because scalars must precede tables
    anyway.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") or stripped.startswith("#"):
            # A comment before the first table is fine to place after our keys.
            if stripped.startswith("["):
                break
            continue
    insert_at = index if index < len(lines) else len(lines)

    def fmt(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

    injected = [f"{key} = {fmt(value)}" for key, value in scalars.items()]
    header = [
        "# Injected by scripts/run_heretic.py (see configs/heretic.momentum.toml)."
    ]
    return "\n".join(lines[:insert_at] + header + injected + lines[insert_at:]) + "\n"


def prepare_workdir(
    config_path: Path, model: Path, output: Path, workdir: Path
) -> Path:
    """Write a resolved config.toml into *workdir* and return it."""
    text = config_path.read_text(encoding="utf-8")
    scalars: dict[str, Any] = {
        "model": str(model.resolve()),
        "save_directory": str(output.resolve()),
    }
    # Heretic prompts interactively for the export strategy unless it is already set,
    # which would hang a scripted run. Only inject when the config is silent.
    if "export_strategy" not in text:
        scalars["export_strategy"] = "merge"
    target = workdir / "config.toml"
    target.write_text(inject_keys(text, scalars), encoding="utf-8")
    return target


def run(
    config_path: Path,
    model: Path,
    output: Path,
    *,
    dry_run: bool = False,
    keep_workdir: bool = False,
) -> int:
    """Execute Heretic, returning its exit status."""
    check_submodule()
    if not model.exists():
        raise RuntimeError(f"model checkpoint not found: {model}")

    workdir = Path(tempfile.mkdtemp(prefix="momentum-heretic-"))
    try:
        target = prepare_workdir(config_path, model, output, workdir)
        # Heretic must run with cwd = its own checkout (it imports `heretic.*`), but it
        # reads config.toml from cwd too. Resolve that by pointing PYTHONPATH at the
        # checkout and running from the scratch directory.
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(HERETIC_DIR / "src"), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        env.setdefault("HF_HOME", str(ROOT / ".hf"))

        command = [sys.executable, "-c", "from heretic.main import main; main()"]
        if dry_run:
            print(f"workdir: {workdir}")
            print(f"resolved config written to: {target}")
            print("would run:", " ".join(command))
            print("--- config.toml ---")
            print(target.read_text(encoding="utf-8"))
            return 0

        log.info("running heretic in %s", workdir)
        result = subprocess.run(command, cwd=str(workdir), env=env, check=False)
        if result.returncode != 0:
            log.error("heretic exited %s", result.returncode)
            print(
                "Abliteration failed. This step is OPTIONAL: the un-abliterated student "
                "is a valid candidate. Measure the charter axis on the merged model "
                "before deciding whether this step is needed at all.",
                file=sys.stderr,
            )
        return result.returncode
    finally:
        if dry_run or keep_workdir:
            log.info("workdir kept at %s", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heretic.momentum.toml")
    parser.add_argument("--model", default="out/student-merged")
    parser.add_argument("--output", default="out/student-abliterated")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify the submodule and the config, run nothing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved config without running Heretic",
    )
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.check_only:
        try:
            repo = check_submodule()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        config = load_toml(Path(args.config))
        print(f"heretic submodule present at {repo}")
        print(
            f"export_strategy: {config.get('export_strategy', '<unset: would prompt>')}"
        )
        print(f"n_trials: {config.get('n_trials')}")
        print(f"scorers: {[s.get('plugin') for s in config.get('scorers', [])]}")
        print(f"modifiers: {[m.get('plugin') for m in config.get('modifiers', [])]}")
        if config.get("export_strategy") is None:
            print(
                "WARNING: export_strategy unset; Heretic would prompt and hang.",
                file=sys.stderr,
            )
            return 1
        return 0

    try:
        return run(
            Path(args.config),
            Path(args.model),
            Path(args.output),
            dry_run=args.dry_run,
            keep_workdir=args.keep_workdir,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
