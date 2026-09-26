"""Tests for the model registry, verification and mirror-fallback download."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest

from momentum.llm import downloader
from momentum.llm.downloader import (
    DEFAULT_MODEL,
    MODELS,
    ModelSpec,
    diagnose_model,
    ensure_model,
    get_spec,
    is_model_downloaded,
    model_size_mb,
    verify_model,
)

_SPEC = ModelSpec(
    name="demo",
    filename="demo.gguf",
    size_mb=42,
    license="Apache-2.0",
    sources=(("owner/demo", "demo.gguf"), ("mirror/demo", "demo.gguf")),
)


class _Resp(io.BytesIO):
    """A urlopen response stand-in exposing Content-Length like the real one."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _fake_gguf(size: int = 4096) -> bytes:
    return b"GGUF" + b"\x03\x00\x00\x00" + b"\x00" * (size - 8)


@pytest.fixture(autouse=True)
def _isolated_models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloader, "_models_dir", lambda: tmp_path)


class TestRegistry:
    def test_known_models_resolve(self) -> None:
        assert get_spec("tinyllama").name == "tinyllama"
        assert get_spec("qwen").name == "qwen"

    def test_unknown_model_falls_back_to_default(self) -> None:
        """A stale config value must not take the coach down."""
        assert get_spec("no-such-model").name == DEFAULT_MODEL

    def test_every_entry_has_a_source_and_licence(self) -> None:
        for spec in MODELS.values():
            assert spec.sources, spec.name
            assert spec.license, spec.name
            assert spec.size_mb > 0, spec.name

    def test_canonical_url_is_first_source(self) -> None:
        assert _SPEC.url == _SPEC.mirror_urls()[0]

    def test_mirror_urls_are_unique(self) -> None:
        assert len(set(_SPEC.mirror_urls())) == len(_SPEC.mirror_urls())

    def test_default_model_has_multiple_sources(self) -> None:
        """The whole point of the registry: no single point of failure."""
        assert len(get_spec(DEFAULT_MODEL).mirror_urls()) > 1

    def test_legacy_module_constants_still_exist(self) -> None:
        assert downloader.MODEL_FILENAME == get_spec("tinyllama").filename
        assert downloader.MODEL_URL.startswith("https://huggingface.co/")
        assert downloader.MODEL_SIZE_MB == get_spec("tinyllama").size_mb

    def test_sizes_differ_between_models(self) -> None:
        assert model_size_mb("tinyllama") != model_size_mb("qwen")


class TestVerifyModel:
    def test_accepts_a_plausible_gguf(self, tmp_path: Path) -> None:
        path = tmp_path / "m.gguf"
        path.write_bytes(_fake_gguf())
        assert verify_model(path, _SPEC)

    def test_rejects_an_html_error_page(self, tmp_path: Path) -> None:
        """A redirected or failed download yields HTML, not a model."""
        path = tmp_path / "m.gguf"
        path.write_bytes(b"<html><body>404 not found</body></html>" + b" " * 4000)
        assert not verify_model(path, _SPEC)

    def test_rejects_a_truncated_file(self, tmp_path: Path) -> None:
        path = tmp_path / "m.gguf"
        path.write_bytes(b"GGUF")
        assert not verify_model(path, _SPEC)

    def test_rejects_a_missing_file(self, tmp_path: Path) -> None:
        assert not verify_model(tmp_path / "absent.gguf", _SPEC)

    def test_enforces_sha256_when_published(self, tmp_path: Path) -> None:
        import hashlib

        payload = _fake_gguf()
        good_sha = hashlib.sha256(payload).hexdigest()
        path = tmp_path / "m.gguf"
        path.write_bytes(payload)
        assert verify_model(path, ModelSpec(**{**_SPEC.__dict__, "sha256": good_sha}))
        assert not verify_model(
            path, ModelSpec(**{**_SPEC.__dict__, "sha256": "00" * 32})
        )


class TestEnsureModel:
    def test_returns_cached_file_without_downloading(self) -> None:
        with (
            patch.object(downloader, "get_spec", lambda _n: _SPEC),
            patch.object(downloader, "_download_file") as dl,
        ):
            path = downloader.get_model_path(_SPEC.name)
            path.write_bytes(_fake_gguf())
            assert ensure_model(_SPEC.name) == path
            dl.assert_not_called()

    def test_downloads_and_verifies(self) -> None:
        with (
            patch.object(downloader, "get_spec", lambda _n: _SPEC),
            patch.object(downloader, "_download_file") as dl,
        ):

            def _write(url: str, dest: Path, cb: object = None) -> None:
                dest.write_bytes(_fake_gguf())

            dl.side_effect = _write
            assert ensure_model(_SPEC.name).exists()
            assert dl.call_count == 1

    def test_falls_back_to_the_next_mirror(self) -> None:
        """One dead repository must not leave the coach unavailable."""
        with (
            patch.object(downloader, "get_spec", lambda _n: _SPEC),
            patch.object(downloader, "_download_file") as dl,
        ):
            calls = {"n": 0}

            def _flaky(url: str, dest: Path, cb: object = None) -> None:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("404")
                dest.write_bytes(_fake_gguf())

            dl.side_effect = _flaky
            assert ensure_model(_SPEC.name).exists()
            assert dl.call_count == 2

    def test_raises_after_exhausting_every_source(self) -> None:
        with (
            patch.object(downloader, "get_spec", lambda _n: _SPEC),
            patch.object(
                downloader, "_download_file", side_effect=OSError("boom")
            ) as dl,
        ):
            with pytest.raises(OSError, match="any of"):
                ensure_model(_SPEC.name)
            assert dl.call_count == 2

    def test_rejects_a_corrupt_download_and_retries(self) -> None:
        """An HTML page saved as .gguf must not be accepted as the model."""
        with (
            patch.object(downloader, "get_spec", lambda _n: _SPEC),
            patch.object(downloader, "_download_file") as dl,
        ):
            calls = {"n": 0}

            def _sometimes_bad(url: str, dest: Path, cb: object = None) -> None:
                calls["n"] += 1
                dest.write_bytes(
                    b"<html>" + b"x" * 4000 if calls["n"] == 1 else _fake_gguf()
                )

            dl.side_effect = _sometimes_bad
            assert ensure_model(_SPEC.name).exists()
            assert dl.call_count == 2

    def test_leaves_no_partial_file_behind(self) -> None:
        with (
            patch.object(downloader, "get_spec", lambda _n: _SPEC),
            patch.object(downloader, "_download_file", side_effect=OSError("network")),
        ):
            with pytest.raises(OSError):
                ensure_model(_SPEC.name)
        assert not is_model_downloaded(_SPEC.name)


class TestDownloadFile:
    def test_removes_the_temp_file_on_failure(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.gguf"
        with patch.object(
            downloader.urllib.request, "urlopen", side_effect=OSError("reset")
        ):
            with pytest.raises(OSError):
                downloader._download_file("https://example.com/m", dest)
        assert not dest.exists()
        assert list(tmp_path.iterdir()) == []

    def test_rejects_a_short_body(self, tmp_path: Path) -> None:
        """A truncated transfer must not be published as the model.

        The response advertises more bytes than it actually delivers, which is
        how a connection drops mid-download.
        """
        dest = tmp_path / "out.gguf"
        resp = _Resp(b"abc")
        resp.headers = {"Content-Length": "999"}
        with patch.object(downloader.urllib.request, "urlopen", return_value=resp):
            with pytest.raises(OSError, match="Incomplete"):
                downloader._download_file("https://example.com/m", dest)
        assert not dest.exists()

    def test_reports_progress(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.gguf"
        seen: list[tuple[int, int]] = []
        with patch.object(
            downloader.urllib.request, "urlopen", return_value=_Resp(_fake_gguf())
        ):
            downloader._download_file(
                "https://example.com/m", dest, lambda d, t: seen.append((d, t))
            )
        assert dest.read_bytes() == _fake_gguf()
        assert seen and seen[-1][0] == seen[-1][1]


class TestDiagnoseModel:
    def test_reports_missing_model(self) -> None:
        report = diagnose_model("qwen")
        assert report["ok"] is False
        assert report["exists"] is False
        assert "Not downloaded" in str(report["error"])

    def test_reports_a_valid_cached_model(self) -> None:
        downloader.get_model_path("qwen").write_bytes(_fake_gguf())
        report = diagnose_model("qwen")
        assert report["ok"] is True
        assert report["verified"] is True
        assert report["error"] == ""

    def test_reports_a_corrupt_model(self) -> None:
        downloader.get_model_path("qwen").write_bytes(b"not a model" * 200)
        report = diagnose_model("qwen")
        assert report["ok"] is False
        assert "truncated" in str(report["error"])

    def test_never_loads_unless_asked(self) -> None:
        downloader.get_model_path("qwen").write_bytes(_fake_gguf())
        with patch("llama_cpp.Llama") as llama:
            diagnose_model("qwen")
            llama.assert_not_called()

    def test_reports_a_real_load(self) -> None:
        pytest.importorskip("llama_cpp", reason="native backend absent")
        from momentum.llm.gguf_writer import build_nano_llama

        build_nano_llama(downloader.get_model_path("qwen"))
        report = diagnose_model("qwen", load=True)
        assert report["ok"] is True
        assert report["loadable"] is True

    def test_surfaces_a_load_failure(self) -> None:
        downloader.get_model_path("qwen").write_bytes(_fake_gguf())
        with patch("llama_cpp.Llama", side_effect=ValueError("corrupt")):
            report = diagnose_model("qwen", load=True)
        assert report["ok"] is False
        assert "corrupt" in str(report["error"])
