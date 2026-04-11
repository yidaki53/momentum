from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

LATEST_RELEASE_URL = "https://api.github.com/repos/yidaki53/momentum/releases/latest"
FALLBACK_RELEASES_PAGE = "https://github.com/yidaki53/momentum/releases/latest"


@dataclass(frozen=True)
class ReleaseInfo:
    version: str
    url: str


def normalize_version(version: str) -> str:
    cleaned = version.strip()
    if cleaned.startswith("v"):
        cleaned = cleaned[1:]
    cleaned = cleaned.split("-", 1)[0]
    parts = cleaned.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid semantic version: {version}")
    return cleaned


def compare_versions(current: str, other: str) -> int:
    current_parts = tuple(int(part) for part in normalize_version(current).split("."))
    other_parts = tuple(int(part) for part in normalize_version(other).split("."))
    if current_parts < other_parts:
        return -1
    if current_parts > other_parts:
        return 1
    return 0


def fetch_latest_release(timeout: float = 5.0) -> ReleaseInfo:
    request = urllib.request.Request(
        LATEST_RELEASE_URL,
        headers={"User-Agent": "Momentum update checker"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    raw_version = payload.get("tag_name") or payload.get("name") or ""
    version = normalize_version(raw_version)
    url = payload.get("html_url") or FALLBACK_RELEASES_PAGE
    return ReleaseInfo(version=version, url=url)


def is_update_available(current: str, latest: str) -> bool:
    return compare_versions(current, latest) < 0
