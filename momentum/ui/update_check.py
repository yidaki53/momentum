from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, cast

LATEST_RELEASE_URL = "https://api.github.com/repos/yidaki53/momentum/releases/latest"
FALLBACK_RELEASES_PAGE = "https://github.com/yidaki53/momentum/releases/latest"


@dataclass(frozen=True)
class ReleaseInfo:
    version: str
    url: str
    assets: list[tuple[str, str]] = field(default_factory=list)
    # Each asset is a (name, browser_download_url) tuple.


def normalize_version(version: str) -> str:
    cleaned = version.strip()
    if cleaned.startswith("v"):
        cleaned = cleaned[1:]
    # Keep our CI build suffix: successive releases often share a base version.
    base, separator, suffix = cleaned.partition("-")
    parts = base.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid semantic version: {version}")
    if separator and suffix.startswith("build."):
        build = suffix.removeprefix("build.")
        if not build.isdigit():
            raise ValueError(f"Invalid build version: {version}")
        return f"{base}-build.{int(build)}"
    return base


def _version_key(version: str) -> tuple[int, ...]:
    base, _, suffix = normalize_version(version).partition("-build.")
    return (*(int(part) for part in base.split(".")), int(suffix or 0))


def compare_versions(current: str, other: str) -> int:
    current_parts = _version_key(current)
    other_parts = _version_key(other)
    if current_parts < other_parts:
        return -1
    if current_parts > other_parts:
        return 1
    return 0


def _read_release_payload(
    request: urllib.request.Request,
    timeout: float,
    ssl_context: ssl.SSLContext | None = None,
) -> dict[str, Any]:
    with urllib.request.urlopen(
        request, timeout=timeout, context=ssl_context
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return cast(dict[str, Any], payload)


def is_certificate_verification_error(exc: urllib.error.URLError) -> bool:
    reason = exc.reason
    if isinstance(reason, ssl.SSLCertVerificationError):
        return True
    reason_text = str(reason).lower()
    return (
        "certificate_verify_failed" in reason_text
        or "unable to get local issuer certificate" in reason_text
    )


def certifi_ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def fetch_latest_release(timeout: float = 5.0) -> ReleaseInfo:
    request = urllib.request.Request(
        LATEST_RELEASE_URL,
        headers={"User-Agent": "Momentum update checker"},
    )
    try:
        payload = _read_release_payload(request, timeout)
    except urllib.error.URLError as exc:
        if not is_certificate_verification_error(exc):
            raise
        certifi_context = certifi_ssl_context()
        if certifi_context is None:
            raise
        payload = _read_release_payload(request, timeout, ssl_context=certifi_context)

    raw_version = payload.get("tag_name") or payload.get("name") or ""
    version = normalize_version(raw_version)
    url = payload.get("html_url") or FALLBACK_RELEASES_PAGE
    assets: list[tuple[str, str]] = []
    for asset in payload.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        name = asset.get("name")
        download_url = asset.get("browser_download_url")
        if isinstance(name, str) and isinstance(download_url, str):
            assets.append((name, download_url))
    return ReleaseInfo(version=version, url=url, assets=assets)


def is_update_available(current: str, latest: str) -> bool:
    return compare_versions(current, latest) < 0
