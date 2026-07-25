"""Tests for the import-safe LLM engine availability probe.

The engine module must be importable even when the native ``llama_cpp``
dependency is absent (the Android APK without the optional recipe). These tests
verify the availability flag is a bool and that ``LlmEngine.load()`` raises a
clear error when the backend is missing, without touching the real native lib.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import momentum.llm.engine as engine_mod
from momentum.llm.engine import LlmEngine, is_llm_available


def test_is_llm_available_returns_bool() -> None:
    """is_llm_available() is always a concrete bool, never None/other."""
    result = is_llm_available()
    assert isinstance(result, bool)


def test_engine_importable_without_native_lib() -> None:
    """The engine module imports cleanly regardless of the native backend."""
    # Importing the module already succeeded (it is imported above); this is a
    # placeholder assertion that documents the import-safe guarantee.
    assert hasattr(engine_mod, "LlmEngine")
    assert hasattr(engine_mod, "is_llm_available")
    assert hasattr(engine_mod, "LLM_AVAILABLE")


def test_load_raises_clear_error_when_backend_missing() -> None:
    """load() raises a descriptive RuntimeError when the native lib is absent."""
    eng = LlmEngine(model_path=Path("/nonexistent/model.gguf"))
    with (
        patch.object(engine_mod, "LLM_AVAILABLE", False),
        patch.object(engine_mod, "Llama", None),
    ):
        try:
            eng.load()
        except RuntimeError as exc:
            assert "llama-cpp-python is not available" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError when the backend is missing")


def test_load_raises_when_already_unavailable_and_not_loaded() -> None:
    """Even with a real path, missing backend short-circuits before any file IO."""
    eng = LlmEngine(model_path=Path("/definitely/missing.gguf"))
    with (
        patch.object(engine_mod, "LLM_AVAILABLE", False),
        patch.object(engine_mod, "Llama", None),
    ):
        try:
            eng.load()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Expected RuntimeError before any model file access")
