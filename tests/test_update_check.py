from __future__ import annotations

import ssl
import urllib.error
from types import SimpleNamespace

import pytest

from momentum.ui import update_check
from momentum.ui.update_check import (
    compare_versions,
    fetch_latest_release,
    is_update_available,
    normalize_version,
)


def test_normalize_version_accepts_tag_prefix() -> None:
    assert normalize_version("v1.2.3") == "1.2.3"


def test_compare_versions_orders_semver_values() -> None:
    assert compare_versions("0.3.1", "0.4.0") == -1
    assert compare_versions("0.4.0", "0.3.1") == 1
    assert compare_versions("0.4.0", "0.4.0") == 0


def test_is_update_available_uses_semver_ordering() -> None:
    assert is_update_available("0.3.1", "0.4.0") is True
    assert is_update_available("0.4.0", "0.4.0") is False


def test_fetch_latest_release_retries_with_certifi_on_cert_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_bytes = b'{"tag_name":"v0.4.1","html_url":"https://github.com/yidaki53/momentum/releases/tag/v0.4.1"}'

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
            return None

        def read(self) -> bytes:
            return payload_bytes

    calls: list[ssl.SSLContext | None] = []

    def fake_urlopen(request, timeout, context=None):  # type: ignore[no-untyped-def]
        calls.append(context)
        if context is None:
            raise urllib.error.URLError(
                ssl.SSLCertVerificationError(
                    1,
                    "certificate verify failed: unable to get local issuer certificate",
                )
            )
        return _Response()

    sentinel_context = ssl.create_default_context()
    monkeypatch.setattr(update_check.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        update_check,
        "_certifi_ssl_context",
        lambda: sentinel_context,
    )

    release = fetch_latest_release(timeout=1.0)

    assert release.version == "0.4.1"
    assert release.url.endswith("/v0.4.1")
    assert calls == [None, sentinel_context]


def test_fetch_latest_release_raises_without_certifi_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cert_error = urllib.error.URLError(
        ssl.SSLCertVerificationError(1, "certificate verify failed")
    )

    def fake_urlopen(request, timeout, context=None):  # type: ignore[no-untyped-def]
        raise cert_error

    monkeypatch.setattr(update_check.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(update_check, "_certifi_ssl_context", lambda: None)

    with pytest.raises(urllib.error.URLError):
        fetch_latest_release(timeout=1.0)


def test_fetch_latest_release_does_not_retry_on_non_ssl_url_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = urllib.error.URLError("timed out")
    calls: list[ssl.SSLContext | None] = []

    def fake_urlopen(request, timeout, context=None):  # type: ignore[no-untyped-def]
        calls.append(context)
        raise failure

    marker = SimpleNamespace(called=False)

    def fake_certifi_context() -> ssl.SSLContext | None:
        marker.called = True
        return ssl.create_default_context()

    monkeypatch.setattr(update_check.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(update_check, "_certifi_ssl_context", fake_certifi_context)

    with pytest.raises(urllib.error.URLError):
        fetch_latest_release(timeout=1.0)

    assert calls == [None]
    assert marker.called is False
