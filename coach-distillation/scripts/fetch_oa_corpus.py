#!/usr/bin/env python3
"""Fetch an open-access corpus and write a provenance manifest.

Only records whose per-record licence is provably CC-BY / CC0 / public domain are
accepted. Everything else is rejected *in code*, because "we were careful" is not a
licence defence and this is the step where a project like this one most easily picks up
text it has no right to redistribute.

Accepted records are written as metadata only. The publisher's prose stays out of git;
the pipeline's job is to turn this corpus into teacher *context*, and only the
teacher-generated replies get committed (see ../SOURCES.md).

Usage::

    python scripts/fetch_oa_corpus.py --config configs/distill.qwen05b.toml
    python scripts/fetch_oa_corpus.py --config configs/distill.qwen05b.toml --dry-run

Stdlib only, so this is importable and testable without torch installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger("fetch_oa_corpus")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT = "momentum-coach-distillation/1.0 (research; mailto:{email})"

# Journals publish under many spellings of the same two licences. Rather than
# enumerate them, normalise to a small set of tokens and match on prefixes: a licence we
# do not recognise is *not* assumed permissive.
_CC_BY = re.compile(r"\bcc[\s\-_]?by\b", re.IGNORECASE)
_CC0 = re.compile(r"\bcc[\s\-_]?0\b", re.IGNORECASE)
_PUBLIC_DOMAIN = re.compile(r"\b(public\s*domain|pd\b|not\s+copyright)", re.IGNORECASE)
# "CC-BY-NC" and friends are free to *read* but explicitly forbid commercial use, so they
# are not on the allow-list for something we intend to redistribute. Caught before the
# generic CC-BY match, which would otherwise accept them.
_NONCOMMERCIAL = re.compile(r"\bcc[\s\-_]?by[\s\-_]?(nc|nd|ncsa|ncnd)\b", re.IGNORECASE)
# Copyleft is fine to *read* but is not on the allow-list, and that is deliberate.
_REJECT = re.compile(
    r"\b(elsevier|springer|wiley|sciencedirect|all rights reserved)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class Record:
    """One accepted corpus record. Provenance is mandatory, not optional."""

    id: str
    source: str
    title: str
    license: str
    doi: str
    pmcid: str
    year: str
    url: str
    sha256: str

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "license": self.license,
            "doi": self.doi,
            "pmcid": self.pmcid,
            "year": self.year,
            "url": self.url,
            "sha256": self.sha256,
        }


@dataclass
class Rejection:
    """A record we declined, and why. Kept so the manifest is auditable."""

    id: str
    reason: str
    license: str = ""
    title: str = ""


@dataclass
class FetchReport:
    """Outcome of one run, summarised for the operator."""

    accepted: list[Record] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    total_seen: int = 0

    @property
    def acceptance_rate(self) -> float:
        return (len(self.accepted) / self.total_seen) if self.total_seen else 0.0


def classify_license(raw: str) -> Optional[str]:
    """Return a normalised licence token, or None when not provably open.

    Returning None for anything unrecognised is the important behaviour: an unknown
    licence is treated as proprietary.
    """
    text = (raw or "").strip()
    if not text:
        return None
    if _REJECT.search(text) or _NONCOMMERCIAL.search(text):
        return None
    if _CC0.search(text):
        return "CC0"
    if _CC_BY.search(text):
        return "CC-BY"
    if _PUBLIC_DOMAIN.search(text):
        return "PD"
    return None


def is_allowed(normalized: Optional[str], allowed: Iterable[str]) -> bool:
    """Check a normalised licence against the configured allow-list."""
    if normalized is None:
        return False
    allowed_set = {item.upper() for item in allowed}
    if normalized.upper() in allowed_set:
        return True
    # "CC-BY" in the allow-list should admit "CC-BY-4.0" style variants.
    return any(
        normalized.upper().startswith(item)
        for item in allowed_set
        if item.startswith("CC")
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _http_get(url: str, *, email: str, api_key: str = "", timeout: float = 30.0) -> str:
    headers = {"User-Agent": USER_AGENT.format(email=email)}
    if api_key:
        headers["api-key"] = api_key
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def search_ids(query: str, *, retmax: int, email: str, api_key: str = "") -> list[str]:
    """Return PubMed IDs for *query* via E-utilities esearch."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(retmax),
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    url = f"{EUTILS}/esearch.fcgi?{urllib.parse.urlencode(params)}"
    payload = json.loads(_http_get(url, email=email, api_key=api_key))
    return list(payload.get("esearchresult", {}).get("idlist", []))


def fetch_summary(
    pmids: list[str], *, email: str, api_key: str = ""
) -> list[dict[str, Any]]:
    """Fetch metadata for *pmids* and flatten NCBI's response structure."""
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    url = f"{EUTILS}/esummary.fcgi?{urllib.parse.urlencode(params)}"
    payload = json.loads(_http_get(url, email=email, api_key=api_key))
    result = payload.get("result", {})
    out: list[dict[str, Any]] = []
    for uid in result.get("uids", []):
        entry = result.get(uid)
        if isinstance(entry, dict):
            out.append(entry)
    return out


def extract_license(entry: dict[str, Any]) -> str:
    """Pull licence signals out of an esummary record.

    esummary exposes licence information inconsistently across the PMC corpus, so this
    gathers every plausible field and returns them joined for :func:`classify_license`.
    """
    chunks: list[str] = []
    for article_id in entry.get("articleids", []) or []:
        if isinstance(article_id, dict):
            for key in ("idtype", "value"):
                value = article_id.get(key)
                if value:
                    chunks.append(str(value))
    for key in ("copyright", "license", "licensetype", "pubtype"):
        value = entry.get(key)
        if isinstance(value, list):
            chunks.extend(str(item) for item in value)
        elif value:
            chunks.append(str(value))
    return " | ".join(chunks)


def build_record(
    entry: dict[str, Any], *, query: str, allowed: Iterable[str]
) -> tuple[Optional[Record], Rejection]:
    """Convert one esummary entry into a Record, or explain the refusal."""
    pmid = str(entry.get("uid") or entry.get("id") or "")
    title = (entry.get("title") or "").strip()
    if not pmid:
        return None, Rejection(id="", reason="no identifier")

    raw_license = extract_license(entry)
    normalized = classify_license(raw_license)
    if normalized is None:
        return None, Rejection(
            id=f"pmid:{pmid}",
            reason="licence absent or unrecognised",
            license=raw_license[:120],
            title=title[:120],
        )
    if not is_allowed(normalized, allowed):
        return None, Rejection(
            id=f"pmid:{pmid}",
            reason=f"licence {normalized} not in allow-list",
            license=normalized,
            title=title[:120],
        )

    doi = ""
    pmcid = ""
    for article_id in entry.get("articleids", []) or []:
        if not isinstance(article_id, dict):
            continue
        idtype = str(article_id.get("idtype", "")).lower()
        value = str(article_id.get("value", ""))
        if idtype == "doi" and not doi:
            doi = value
        elif idtype in {"pmcid", "pmc"} and not pmcid:
            pmcid = value

    # A DOI is mandatory. Without one the record cannot be traced back to a specific
    # version of a specific paper, and provenance we cannot cite is provenance we
    # cannot defend.
    if not doi:
        return None, Rejection(
            id=f"pmid:{pmid}",
            reason="no DOI",
            license=normalized,
            title=title[:120],
        )

    record = Record(
        id=f"pmid:{pmid}",
        source="pmc-oa",
        title=title,
        license=normalized,
        doi=doi,
        pmcid=pmcid,
        year=str(entry.get("pubdate", ""))[:4],
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        # The hash covers the metadata we actually store, so tampering with the manifest
        # is detectable even though the source text lives outside git.
        sha256=_sha256(f"{title}|{doi}|{normalized}"),
    )
    log.info("accepted %s (%s) from query %r", record.id, record.license, query)
    return record, Rejection(id="", reason="")


def load_config(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as handle:
        return tomllib.load(handle)


def write_manifest(path: Path, report: FetchReport) -> None:
    """Write the manifest plus a sibling rejections log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # One line, not a pretty-printed object: this file is JSONL, and a multi-line
    # header makes every line-oriented reader choke on line 1.
    header = {
        "_comment": (
            "Provenance ledger for the Momentum coach corpus. One row per accepted "
            "source record. Publisher prose is NOT stored here: only metadata, so this "
            "file is redistributable. See ../LICENSES.md."
        ),
        "count": len(report.accepted),
    }
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for record in report.accepted:
            handle.write(json.dumps(record.to_row(), sort_keys=True) + "\n")

    rejections_path = path.with_name(path.stem + "_rejected.jsonl")
    with rejections_path.open("w", encoding="utf-8") as handle:
        for rejection in report.rejected:
            if rejection.id:
                handle.write(json.dumps(rejection.__dict__, sort_keys=True) + "\n")
    log.info(
        "wrote %d accepted, %d rejected", len(report.accepted), len(report.rejected)
    )


def fetch(config_path: Path, *, dry_run: bool = False) -> FetchReport:
    """Run every configured query and return the report."""
    config = load_config(config_path)
    corpus = config.get("corpus", {})
    email = corpus.get("email", "research@example.invalid")
    api_key = corpus.get("api_key", "")
    retmax = int(corpus.get("retmax_per_query", 200))
    allowed = corpus.get("allowed_licenses", ["CC0", "CC-BY"])
    queries = corpus.get("queries", [])

    report = FetchReport()
    seen: set[str] = set()
    for query in queries:
        try:
            pmids = search_ids(query, retmax=retmax, email=email, api_key=api_key)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            # A failed query is not a failed run: keep going and report the shortfall.
            log.error("query failed (%s): %s", query, exc)
            continue
        entries = fetch_summary(pmids, email=email, api_key=api_key)
        for entry in entries:
            report.total_seen += 1
            record, rejection = build_record(entry, query=query, allowed=allowed)
            if record is None:
                report.rejected.append(rejection)
                continue
            if record.id in seen:
                continue
            seen.add(record.id)
            report.accepted.append(record)
        # NCBI asks for ~3 requests/second without an API key.
        time.sleep(0.34)

    log.info(
        "acceptance %.1f%% (%d/%d)",
        100 * report.acceptance_rate,
        len(report.accepted),
        report.total_seen,
    )
    if not dry_run:
        write_manifest(Path("data/manifest.jsonl"), report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distill.qwen05b.toml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="query and classify without writing the manifest",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = fetch(Path(args.config), dry_run=args.dry_run)
    print(
        f"accepted {len(report.accepted)} / {report.total_seen} "
        f"({report.acceptance_rate:.1%})"
    )
    if not report.accepted:
        print(
            "Nothing accepted. Check connectivity before proceeding.", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
