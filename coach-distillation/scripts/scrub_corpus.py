#!/usr/bin/env python3
"""Remove rows that teach the model to invent files, paths and other artefacts.

Why this exists
---------------
151 of the 495 rejection-sampled rows in ``train-v7.jsonl`` (and 147 of 491 in v8) carry a
concrete file path in the TARGET reply that appears nowhere in the app data shown with
it::

    response: Open the file named Documents/Harvard/Biostatistics/Week1/Introduction.txt
    user_data: Active tasks: Finish literature review

So every model trained since has learned to name files that do not exist, with complete
confidence. Measured on the 120-item bench: v7 invents a path in 62% of its replies and
the best merged model in 36%. No axis could see it -- ``grounding`` checks that a reply
does not CONTRADICT or narrate the data, and an absent path is an addition, not a
contradiction -- which is why the ``specificity`` axis was added alongside this script.

The cause is half the seed's fault. ``BOUNDED_ACTIONS`` says "Open the file and type one
heading line for it"; the sampler completed the implied blank with something plausible.
Both ends are fixed: the seed no longer implies a filename, and ``rejection_sample.py``
now rejects replies that invent one. This script cleans the corpora already built.

Rows are **dropped, not rewritten.** Inventing a replacement would mean inventing coaching
text, and a synthetic "correct" answer written by a script is exactly the
first-party-but-fabricated material that caused the problem. Dropping costs ~30% of the
rft tier; that is the honest price.

``--audit`` reports without writing and ``--require-clean`` fails the exit code instead,
so this can gate a pipeline rather than quietly changing a corpus under two later runs
that are then compared against each other.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

log = logging.getLogger("scrub_corpus")

# The same expression the bench's `specificity` axis uses, deliberately. A corpus filter
# that disagreed with the release gate would clean rows the gate still rejects, or reject
# rows the gate accepts -- both worse than not running at all.
PATH_RE = re.compile(
    r"\b[A-Za-z]:?[\\/][\w.\-]+(?:[\\/][\w.\-]+)*|"
    r"\bDocuments/[^\s,.;:]*|"
    r"\b[\w\-]+\.(?:txt|html|md|csv|pdf|docx)\b"
)


def invented_paths(row: dict[str, Any]) -> list[str]:
    """Paths in the target reply that the app data never showed.

    Compared against the RAW user_data, not a tokenised form: a path can be entirely
    legitimate and still contribute no content words.
    """
    response = str(row.get("response", ""))
    data = str(row.get("user_data", "")).lower()
    return [p for p in sorted(set(PATH_RE.findall(response))) if p.lower() not in data]


def scrub(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split *rows* into kept and dropped. Pure: no IO, so it is directly testable."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    kinds_before: Counter[str] = Counter()
    kinds_after: Counter[str] = Counter()
    for row in rows:
        kind = str(row.get("kind", "unknown"))
        kinds_before[kind] += 1
        if invented_paths(row):
            dropped.append(row)
        else:
            kept.append(row)
            kinds_after[kind] += 1
    report = {
        "total": len(rows),
        "kept": len(kept),
        "dropped": len(dropped),
        "kinds_before": dict(kinds_before),
        "kinds_after": dict(kinds_after),
        "examples": [
            {"id": r.get("id"), "kind": r.get("kind"), "paths": invented_paths(r)}
            for r in dropped[:5]
        ],
    }
    return kept, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", help="input .jsonl")
    parser.add_argument(
        "--output",
        default=None,
        help="where to write the cleaned corpus (required unless --audit/--require-clean)",
    )
    parser.add_argument(
        "--audit", action="store_true", help="report only; write nothing"
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="exit 1 if any row would be dropped, without writing",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    source = Path(args.corpus)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept, report = scrub(rows)

    print(f"{source}: {report['total']} rows")
    print(f"  kept    : {report['kept']}")
    print(
        f"  dropped : {report['dropped']} (invented a file the app data never showed)"
    )
    for kind in sorted(report["kinds_before"]):
        before = report["kinds_before"][kind]
        after = report["kinds_after"].get(kind, 0)
        flag = "  <-- ENTIRELY REMOVED" if after == 0 and before else ""
        print(f"    {kind:12} {before:4} -> {after:4}{flag}")
    for example in report["examples"]:
        print(f"    e.g. {example['id']} {example['paths']}")

    if args.require_clean:
        if report["dropped"]:
            print(f"FAIL: {report['dropped']} rows would be dropped")
            return 1
        print("clean")
        return 0

    if args.audit:
        print("audit only; nothing written")
        return 0

    if not args.output:
        parser.error("--output is required unless --audit or --require-clean is given")

    # A tier wiped out entirely would silently remove a behaviour rather than fix it, and
    # the run would look healthy because the other tiers survived.
    wiped = [
        kind
        for kind, before in report["kinds_before"].items()
        if before and report["kinds_after"].get(kind, 0) == 0
    ]
    if wiped:
        print(f"refusing to write: these tiers would be removed entirely: {wiped}")
        print(
            "that deletes a behaviour rather than repairing it; fix those rows by hand."
        )
        return 1

    out = Path(args.output)
    if out.exists() and out.resolve() == source.resolve():
        parser.error("refusing to overwrite the input corpus; write to a new path")
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
        encoding="utf-8",
    )
    report_path = out.parent / f"{out.stem}.scrub_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {len(kept)} rows to {out}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
