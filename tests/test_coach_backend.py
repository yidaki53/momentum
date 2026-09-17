"""Host-side tests for native import failures and download integrity."""

from __future__ import annotations

import builtins
import io
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

from momentum.llm import downloader, engine


@pytest.mark.parametrize(
    "error", [ImportError("missing"), OSError("dlopen"), RuntimeError("loader")]
)
def test_native_import_error_is_retained(monkeypatch, error):
    original = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "llama_cpp":
            raise error
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    module = runpy.run_path(engine.__file__)
    assert module["LLM_AVAILABLE"] is False
    assert str(error) in module["LLM_IMPORT_ERROR"]


@pytest.mark.parametrize(("variant", "layers"), [("cpu", 0), ("vulkan", -1)])
def test_gpu_variant_requests_offload(monkeypatch, variant, layers):
    llama = Mock()
    monkeypatch.setattr(engine, "Llama", llama)
    monkeypatch.setattr(engine, "LLM_AVAILABLE", True)
    monkeypatch.setattr(engine, "BUILD_VARIANT", variant)
    engine.LlmEngine(Path("/fake.gguf")).load()
    assert llama.call_args.kwargs["n_gpu_layers"] == layers


@pytest.mark.parametrize("declared_size", [3, 20])
def test_download_flushes_and_rejects_truncation(tmp_path, monkeypatch, declared_size):
    class Response(io.BytesIO):
        headers = {"Content-Length": str(declared_size)}

    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", lambda *a, **kw: Response(b"abc")
    )
    monkeypatch.setattr(downloader, "certifi_ssl_context", lambda: None)
    dest = tmp_path / "model.gguf"
    if declared_size == 3:
        downloader._download_file("https://example.com/model", dest)
        assert dest.read_bytes() == b"abc"
    else:
        with pytest.raises(OSError, match="Incomplete"):
            downloader._download_file("https://example.com/model", dest)
        assert not dest.exists()
        assert not list(tmp_path.iterdir())
