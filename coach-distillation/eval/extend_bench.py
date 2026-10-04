#!/usr/bin/env python3
"""Extend the golden bench, refusing to leak new items into training data.

Why this exists: activation had a denominator of 13 items, which cannot resolve the
effect sizes we are chasing. v2 and v3 differed by one item and five of thirteen
flipped in both directions -- indistinguishable from noise, so a hyperparameter sweep
against that axis would select luck rather than quality.

Widening an eval set only helps if the new items measure something new, so:

- every prompt must be absent from every training corpus (checked; refuses to write
  otherwise);
- every reference must itself pass every axis it will be scored on, or the bench is
  unwinnable by construction;
- the items are activation-focused because that is the thin axis, and each is a
  distinct failure mode rather than a paraphrase of "cannot start writing".

Writes eval/golden_qa_extended.jsonl, which is deliberately NOT wired into the
release gate: a wider denominator raises confidence in a number nobody has baselined.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from extra_activation_items import DIGEST, ITEMS  # noqa: E402

CORPORA = [
    pathlib.Path("data/synthetic/train.jsonl"),
    pathlib.Path("data/synthetic/train-v2.jsonl"),
    pathlib.Path("data/synthetic/train-v3.jsonl"),
    pathlib.Path("out/regen-3b.jsonl"),
    pathlib.Path("out/regen-7b.jsonl"),
    pathlib.Path("eval/asimov_vignettes.jsonl"),
    pathlib.Path("eval/golden_qa.jsonl"),
]


def load(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def train_prompts():
    seen = {}
    for path in CORPORA:
        for row in load(path):
            prompt = (row.get("prompt") or row.get("situation") or "").strip().lower()
            if prompt:
                seen.setdefault(prompt, set()).add(path.name)
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="emit the extended file")
    args = parser.parse_args()

    bench = load(pathlib.Path("eval/golden_qa.jsonl"))
    counts = {}
    for r in bench:
        for a in r.get("axes") or [
            "activation",
            "dignity",
            "grounding",
            "charter",
            "brevity",
            "sanity",
        ]:
            counts[a] = counts.get(a, 0) + 1
    print("current axis denominators:")
    for k, v in sorted(counts.items()):
        print(f"  {k:12s} {v}")

    seen = train_prompts()
    print(f"\ndistinct prompts across bench + corpora: {len(seen)}")

    collisions = [
        (prompt, sorted(seen[prompt.strip().lower()]))
        for prompt, _ in ITEMS
        if prompt.strip().lower() in seen
    ]
    if collisions:
        print("\nREFUSING to write: new items collide with existing prompts:")
        for prompt, src in collisions:
            print(f"  {prompt[:50]:52s} <- {src}")
        return 1

    ids = {r["id"] for r in bench}
    rows = []
    for prompt, reference in ITEMS:
        key = prompt.strip().lower()
        if key in seen:
            continue
        index = len(bench) + len(rows) + 1
        while f"g{index:02d}" in ids:
            index += 1
        item_id = f"g{index:02d}"
        ids.add(item_id)
        rows.append(
            {
                "id": item_id,
                "kind": "golden",
                "prompt": prompt,
                "reference": reference,
                "user_data": DIGEST,
            }
        )
        seen[key] = {"golden_qa_extended.jsonl"}

    print(f"\n{len(rows)} new items, none colliding with any existing prompt")
    new_activation = counts.get("activation", 0) + len(rows)
    print(f"activation denominator: {counts.get('activation', 0)} -> {new_activation}")
    if args.write:
        out = pathlib.Path("eval/golden_qa_extended.jsonl")
        out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        print(f"wrote {out}")
    else:
        print("(pass --write to emit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
