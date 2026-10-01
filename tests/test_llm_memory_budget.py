"""Regression tests for the coach memory budget.

Both on-device failures traced back to one missing check. Loading a GGUF asks
llama.cpp to memory-map the weights and then allocate KV-cache and compute
scratch on top; when the device cannot satisfy that, llama.cpp aborts natively
with SIGSEGV, which no Python ``except`` can intercept. The observed symptoms
were an AI Coach that sat on "thinking" forever (the load ran inline on the Kivy
main thread) and an app that died outright when the help-page coach button was
tapped (a native crash at ~185 MB pss).

These tests pin the behaviour that converts both into ordinary, catchable
Python errors carrying a message the UI can render.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import momentum.llm.engine as engine_mod
from momentum.llm.engine import (
    InsufficientMemoryError,
    check_memory_budget,
    estimate_load_mb,
)


def _fake_model(tmp_path: Path, size_mb: int) -> Path:
    """Write a GGUF-shaped file of *size_mb* so size-based estimates are real."""
    path = tmp_path / "model.gguf"
    with path.open("wb") as handle:
        handle.write(b"GGUF")
        handle.truncate(size_mb * 1024 * 1024)
    return path


def test_estimate_scales_with_weights_and_context(tmp_path) -> None:
    """A bigger model and a bigger context window both cost more memory."""
    small = estimate_load_mb(_fake_model(tmp_path, 64), n_ctx=512)
    large_ctx = estimate_load_mb(_fake_model(tmp_path, 64), n_ctx=4096)
    assert large_ctx > small

    missing = estimate_load_mb(tmp_path / "missing.gguf", n_ctx=512)
    assert missing < small  # a missing file must not fabricate a size


def test_refuses_load_when_memory_is_insufficient(tmp_path, monkeypatch) -> None:
    """The headline fix: refuse before llama.cpp can segfault."""
    model = _fake_model(tmp_path, 720)
    monkeypatch.setattr(engine_mod, "_available_memory_mb", lambda: 400)

    refusal = check_memory_budget(model, 2048)
    assert refusal is not None
    assert "memory" in refusal.lower()


def test_permits_load_when_memory_is_sufficient(tmp_path, monkeypatch) -> None:
    """A device with room is not blocked by the guard."""
    model = _fake_model(tmp_path, 64)
    monkeypatch.setattr(engine_mod, "_available_memory_mb", lambda: 8192)

    assert check_memory_budget(model, 512) is None


def test_unmeasurable_memory_does_not_block(tmp_path, monkeypatch) -> None:
    """If free memory cannot be read, do not invent a failure."""
    model = _fake_model(tmp_path, 720)
    monkeypatch.setattr(engine_mod, "_available_memory_mb", lambda: None)

    assert check_memory_budget(model, 2048) is None


def test_tight_but_positive_headroom_is_still_refused(tmp_path, monkeypatch) -> None:
    """The LOW_MEMORY kills happened with memory left, so a thin margin refuses."""
    model = _fake_model(tmp_path, 700)
    needed = estimate_load_mb(model, 2048)
    # Leave a sliver: enough to technically fit, far too little to survive.
    monkeypatch.setattr(engine_mod, "_available_memory_mb", lambda: needed + 8)

    assert check_memory_budget(model, 2048) is not None


def test_load_raises_catchable_error_instead_of_segfaulting(
    tmp_path, monkeypatch
) -> None:
    """``load()`` must refuse in Python, never reach the native constructor.

    The fake ``Llama`` raises if constructed, which would be the point at which
    the real backend would take the process down with SIGSEGV.

    Symbols are resolved through :func:`sys.modules` rather than the imported
    ``engine_mod`` name: ``test_llm_engine_availability`` calls
    ``importlib.reload`` on the engine, which rebinds its globals. A module
    object captured at import time is then a stale copy that the code under
    test no longer consults.
    """
    import sys

    engine_mod = sys.modules["momentum.llm.engine"]

    model = _fake_model(tmp_path, 720)
    monkeypatch.setattr(engine_mod, "_available_memory_mb", lambda: 256)
    monkeypatch.setattr(engine_mod, "LLM_AVAILABLE", True)

    def _boom(*_args, **_kwargs):
        raise AssertionError("native Llama must not be constructed")

    monkeypatch.setattr(engine_mod, "Llama", _boom)

    # Resolve the class from the live module: a reload rebinds it, and the
    # file-scope name would then be a different class whose exception the
    # ``pytest.raises`` below cannot match.
    engine = engine_mod.LlmEngine(model_path=model, n_ctx=2048)
    with pytest.raises(engine_mod.InsufficientMemoryError):
        engine.load()


def test_insufficient_memory_error_is_a_runtime_error() -> None:
    """Existing ``except Exception`` handlers must already cover this."""
    assert issubclass(InsufficientMemoryError, RuntimeError)


def test_context_window_matches_prompt_requirements(monkeypatch) -> None:
    """The window must fit Momentum's prompts on every platform.

    A 512-token Android setting once looked like a memory saving but made every
    message fail, because the system prompt alone exceeds it.
    """
    monkeypatch.setenv("ANDROID_ARGUMENT", "1")
    assert engine_mod._default_n_ctx() == 2048

    monkeypatch.delenv("ANDROID_ARGUMENT", raising=False)
    assert engine_mod._default_n_ctx() == 2048


def test_assist_refuses_before_spawning_a_worker(monkeypatch) -> None:
    """``request_assistance`` must not start a thread it knows will fail.

    This is the help-page crash path: the load ran on a worker thread and died
    natively. The refusal has to happen on the calling thread so the message
    can be reported through ``on_error``.
    """
    from momentum.llm import assist

    monkeypatch.setattr(assist, "is_ready", lambda config=None: True)
    monkeypatch.setattr(
        assist, "_memory_refusal", lambda name: "not enough memory on this device"
    )

    def _no_thread(*_args, **_kwargs):
        raise AssertionError("must not spawn a worker when the budget is refused")

    monkeypatch.setattr(assist.threading, "Thread", _no_thread)

    errors: list[Exception] = []
    thread = assist.request_assistance("hello", on_error=errors.append, cache_key="k1")

    assert thread is None
    assert len(errors) == 1
    assert isinstance(errors[0], InsufficientMemoryError)
    assert "not enough memory" in str(errors[0])
