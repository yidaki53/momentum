"""Regression tests for the context-window budget.

Sending "hi" to the coach on Android failed with "Requested tokens (...)
exceed context window". The cause was a 512-token context window introduced to
save KV-cache memory, while ``CHAT_SYSTEM_PROMPT`` inlines the entire
executive-dysfunction knowledge base and is ~1450 tokens by itself. llama-cpp
rejects a request when the prompt alone fills the window, so every message
failed regardless of its length.

These tests pin the window size and the budget arithmetic that keeps the
failure unreachable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import momentum.llm.engine as engine_mod
from momentum.llm import prompts
from momentum.llm.engine import LlmEngine, _estimate_prompt_tokens, _fit_max_tokens


def _chat_messages(history_turns: int = 6) -> list[dict[str, str]]:
    """Build the same message list the coach screen sends."""
    history = [
        {"role": "user", "content": "I could not start my report today."},
        {"role": "assistant", "content": "That is common. Try two minutes."},
    ] * (history_turns // 2)
    return prompts.build_chat_prompt("hi", "Task focus: 25 minutes.", history)


def test_context_window_fits_the_system_prompt() -> None:
    """The bare system prompt plus a greeting must fit the default window.

    This is the exact regression: at 512 tokens the prompt alone overflowed, so
    every message was rejected before it was read.
    """
    n_ctx = engine_mod._default_n_ctx()
    messages = [
        {"role": "system", "content": prompts.CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": "hi"},
    ]

    assert _estimate_prompt_tokens(messages) < n_ctx, (
        "the system prompt alone must fit inside the context window"
    )


def test_context_window_is_not_starved_on_android(monkeypatch) -> None:
    """Android must not use a window smaller than the prompts require.

    The 512-token setting looked like a memory saving but made the coach
    unusable, so the window is now the same on every platform.
    """
    monkeypatch.setenv("ANDROID_ARGUMENT", "1")
    assert engine_mod._default_n_ctx() == 2048


def test_real_chat_request_gets_a_usable_budget() -> None:
    """A realistic conversation yields a budget instead of a hard failure."""
    engine = LlmEngine(model_path=Path("/nonexistent.gguf"), n_ctx=2048)
    messages = _chat_messages()

    fitted, budget = engine._prepare_request(messages, max_tokens=512)

    assert budget > 0, "a normal conversation must not be rejected"
    assert _estimate_prompt_tokens(fitted) + budget <= 2048


def test_budget_never_exceeds_the_requested_amount() -> None:
    """A short prompt keeps the full requested completion."""
    engine = LlmEngine(model_path=Path("/nonexistent.gguf"), n_ctx=2048)
    _, budget = engine._prepare_request(
        [{"role": "user", "content": "hi"}], max_tokens=256
    )

    assert budget == 256


def test_overlong_conversation_trims_oldest_turns() -> None:
    """A huge history is trimmed, keeping the system prompt and newest turn."""
    engine = LlmEngine(model_path=Path("/nonexistent.gguf"), n_ctx=2048)
    messages = [
        {"role": "system", "content": prompts.CHAT_SYSTEM_PROMPT},
        *({"role": "user", "content": "x" * 2000} for _ in range(8)),
        {"role": "user", "content": "hi"},
    ]

    fitted, budget = engine._prepare_request(messages, max_tokens=512)

    assert len(fitted) < len(messages), "old turns must be dropped"
    assert budget > 0
    assert fitted[0]["content"] == prompts.CHAT_SYSTEM_PROMPT, "system prompt kept"
    assert fitted[-1]["content"] == "hi", "newest message kept"


def test_fit_max_tokens_reports_zero_when_nothing_fits() -> None:
    """A prompt that cannot fit yields zero so the caller can explain."""
    huge = [{"role": "user", "content": "x" * 500_000}]

    assert _fit_max_tokens(2048, huge, 512) == 0


@pytest.mark.parametrize("n_ctx", [2048])
def test_prompt_plus_budget_never_exceeds_window(n_ctx: int) -> None:
    """The invariant that makes the llama-cpp error unreachable."""
    for turns in (0, 2, 6, 20):
        messages = _chat_messages(history_turns=max(2, turns))
        engine = LlmEngine(model_path=Path("/nonexistent.gguf"), n_ctx=n_ctx)
        fitted, budget = engine._prepare_request(messages, max_tokens=512)
        if budget > 0:
            assert _estimate_prompt_tokens(fitted) + budget <= n_ctx
