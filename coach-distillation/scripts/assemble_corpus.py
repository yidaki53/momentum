#!/usr/bin/env python3
"""Assemble a training corpus from accepted teacher rows plus the first-party tiers.

Why this exists: ``out/regen-7b.jsonl`` contains only ``kind: coaching`` rows. The
hand-written tiers -- ``bounded``, ``substantive``, ``repair``, ``charter`` -- are
emitted by ``plan_pairs`` with hand-written responses and are appended when a run
generates them, but a resumed run that skips already-accepted IDs writes back only
the teacher rows. Training on that file alone silently drops the first-party tiers,
and they are the only rows that cannot be acceptance-biased.

The ratio matters as much as the contents. Gold tiers are ~12% of the raw rows, and
``train.repeat_kinds`` exists to lift them; at 4x they are roughly 42% of the training
set. Dropping them entirely would push the student further towards the teacher's
phrasing than any run before it.

Usage: assemble_corpus.py --accepted out/regen-7b.jsonl --output data/synthetic/train-v4.jsonl
"""

import argparse
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
GOLD_KINDS = ("bounded", "substantive", "repair", "charter")
# Self-generated and verifier-approved. Kept separate from GOLD_KINDS because it has a
# different provenance: the student wrote these and the bench admitted them, so they are
# on-policy rather than first-party. Both are equally free of acceptance bias -- a reply
# that failed the scorer was never written down -- but they are not equally
# human-authored, and conflating the two would overstate what the corpus contains.
RFT_KIND = "rft"


def load_module():
    path = HERE / "build_distill_pairs.py"
    spec = importlib.util.spec_from_file_location("_assemble_pairs", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", required=True, help="accepted teacher rows")
    parser.add_argument(
        "--rft",
        default=None,
        help="verified on-policy rows from rejection_sample.py",
    )
    parser.add_argument(
        "--drop-teacher",
        action="store_true",
        help="omit the teacher tier entirely (v6: the 7B rows measured as a tax)",
    )
    parser.add_argument("--output", required=True, help="corpus to write")
    parser.add_argument("--config", default="configs/regen.7b.toml")
    args = parser.parse_args()

    module = load_module()
    config = module.load_config(pathlib.Path(args.config))
    planned = module.plan_pairs(config)

    # First-party tiers come from the plan, which carries hand-written responses.
    gold = [p for p in planned if p.kind in GOLD_KINDS]

    accepted = read_jsonl(pathlib.Path(args.accepted))
    if args.drop_teacher:
        accepted = []
    on_policy = read_jsonl(pathlib.Path(args.rft)) if args.rft else []

    seen: set[str] = set()
    merged: list[dict] = []
    for row in (
        accepted
        + on_policy
        + [
            {
                "id": p.id,
                "kind": p.kind,
                "situation": p.situation,
                "skill": p.skill,
                "prompt": p.prompt,
                "constraint": p.constraint,
                "response": p.response,
                "citation": p.citation,
                "user_data": p.user.digest(),
            }
            for p in gold
        ]
    ):
        if row["id"] in seen:
            continue
        if not (row.get("response") or "").strip():
            continue
        seen.add(row["id"])
        merged.append(row)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in merged:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    kinds: dict[str, int] = {}
    for row in merged:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    print(f"wrote {len(merged)} rows to {out}")
    print(f"  by kind: {kinds}")
    gold_n = sum(kinds.get(k, 0) for k in GOLD_KINDS)
    print(
        f"  first-party tiers: {gold_n}/{len(merged)} = {100 * gold_n / len(merged):.0f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
